"""
ThinkMail Backend
- Google OAuth login
- Groq API proxy
- Supabase database for user tracking
- Zero email logging
"""

import os
import json
import hashlib
import httpx
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from jose import jwt, JWTError
from database import upsert_user, increment_usage, get_user_stats

load_dotenv()

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET           = os.getenv("JWT_SECRET", "change-this-in-production")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:8000")
REDIRECT_URI         = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")
FREE_TIER_LIMIT      = int(os.getenv("FREE_TIER_LIMIT", "50"))

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="ThinkMail API", version="1.0.0")
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too many requests."})


# ── Models ────────────────────────────────────────────────────────────────────

class FixRequest(BaseModel):
    thread: Optional[str] = ""
    draft: Optional[str] = ""

class FixResponse(BaseModel):
    result: str
    fixes_used: int
    fixes_remaining: int

class CoachRequest(BaseModel):
    thread: Optional[str] = ""
    draft: Optional[str] = ""

class CoachResponse(BaseModel):
    improved: str   # The polished version of the draft, matching thread format/style
    note: str       # One line: what was adjusted and why. Empty if nothing changed.


# ── Auth helpers ──────────────────────────────────────────────────────────────

def create_jwt(user_id: str, email: str, name: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": datetime.utcnow() + timedelta(days=30)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please sign in again.")

def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated. Please sign in.")
    return verify_jwt(auth.split(" ", 1)[1])


# ── Prompts ───────────────────────────────────────────────────────────────────

def build_prompt(thread: str, draft: str, today: str) -> list[dict]:
    no_draft_msg = "USER HAS NO DRAFT — Write the reply entirely from scratch based on the full thread context."
    has_draft_msg = "USER DRAFT REPLY:\n---\n" + draft + "\n---"
    draft_section = has_draft_msg if draft.strip() else no_draft_msg
    thread_section = thread.strip() if thread.strip() else "No thread found — only a draft is available."

    if not thread.strip() and not draft.strip():
        user_content = "Tell the user to open a Gmail thread first."
    else:
        user_content = (
            "Today: " + today + "\n\n"
            + thread_section + "\n\n"
            + draft_section + "\n\n"
            + """Read every single email in the thread above carefully before responding.
Understand who said what, when, what was promised, what changed, and what the current situation actually is.

Respond in EXACTLY this format — nothing outside these sections:

TONE: [one word: Formal / Casual / Tense / Friendly / Aggressive / Professional]
URGENCY: [one word: Low / Medium / High]
VIBE: [one word: Positive / Neutral / Tense / Hostile / Warm]
INTENT: [one short phrase — what does the other person actually want?]
RISK: [one word: Low / Medium / High — how risky is it to reply wrong here?]

SITUATION:
[3-4 plain English sentences. Summarize the ENTIRE thread history. Who are the people involved? What has been said across all emails? What is the actual current situation? Write like explaining it to a friend who hasn't read any of it.]

CONTEXT ANALYSIS:
[What is the deeper context? Is there a power dynamic? Has anything been promised and not delivered? Is the tone changing across the thread? What should the user be aware of before replying? Be specific and reference actual things said.]

CONFLICTS:
[Look across the ENTIRE thread. Did someone make a promise they are now contradicting? Is the draft ignoring something important said earlier? Is someone moving goalposts? Be specific. If nothing is wrong say exactly: No conflicts detected.]

SUGGESTED REPLY:
[Write the reply the user should actually send. Sound like a real human. Match the tone of the conversation. Never use filler openers. If the other person is wrong, address it directly but not rudely. If no draft was provided, write the complete reply from scratch. Be concise. Never be a pushover.]"""
        )

    system_content = (
        "You are ThinkMail — an email situational intelligence assistant.\n"
        "Today: " + today + "\n\n"
        "Your core job: Read the FULL email thread. Understand the complete history and context — not just the latest message.\n"
        "You are on the USER side. Always.\n"
        "You think like a smart friend who read every email in the chain.\n"
        "You catch timeline conflicts, broken promises, power dynamics, and meaning errors.\n"
        "You write replies that sound human — confident, direct, appropriate.\n"
        "You never generate generic AI text. You never suggest pushover replies.\n"
        "Always follow the exact output format — every section, every time."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def build_coach_prompt(thread: str, draft: str, today: str) -> list[dict]:
    """
    Takes the user's draft and returns ONE polished version of it.

    Critical behaviours:
    - Study the thread's FORMAT: does it end with 'Thanks', 'Regards', a phone number,
      sign-off, or nothing? Does it use bullet points? Short sentences or long paragraphs?
      Casual 'hey' openers or formal 'Dear'? Mirror all of that exactly.
    - Study the thread's TONE: is it a close colleague, a client, a boss, a stranger?
      Casual banter? Formal business? Tense escalation? Match it.
    - Keep the user's intent word-for-word where possible. Only change HOW it sounds.
    - Never add things that weren't in the draft (don't invent apologies, offers, or promises).
    - If the draft is already perfect for the situation — return it unchanged.
    """
    thread_section = thread.strip() if thread.strip() else "No prior thread — this is the opening message."

    system_content = (
        "You are ThinkMail's silent writing polish engine.\n"
        "Today: " + today + "\n\n"

        "YOUR JOB:\n"
        "Read the full email thread and the user's current draft. "
        "Return the best version of that draft — same message, better delivery.\n\n"

        "STEP 1 — READ THE THREAD FORMAT:\n"
        "Before touching the draft, study every email in the thread and extract:\n"
        "- Sign-off style: 'Thanks', 'Regards', 'Cheers', name only, phone number, nothing?\n"
        "- Opener style: 'Hey [name]', 'Hi', 'Dear', no opener at all?\n"
        "- Length: short 2-3 line replies, or long paragraphs?\n"
        "- Formality: casual friend, professional colleague, senior exec, client?\n"
        "- Any recurring elements: bullet points, numbered lists, specific phrases they always use?\n\n"

        "STEP 2 — POLISH THE DRAFT:\n"
        "Rewrite the user's draft to:\n"
        "- Match exactly the format and sign-off style you found in STEP 1.\n"
        "- Correct the tone if it's too weak, too aggressive, or off for the relationship.\n"
        "- Sound like a confident, real human. No filler, no corporate speak.\n"
        "- Keep every point the user made. Do not add new content or promises.\n\n"

        "STEP 3 — WRITE THE NOTE:\n"
        "One short sentence (max 12 words) saying what you changed and why.\n"
        "Examples: 'Matched casual tone and added sign-off from thread history'\n"
        "          'Removed apologetic opener — thread context doesn't warrant it'\n"
        "          'Kept formal format consistent with previous emails in thread'\n"
        "If nothing needed changing, leave note as empty string.\n\n"

        "OUTPUT: valid JSON only. No markdown, no explanation outside the JSON.\n"
        '{"improved": "the polished reply", "note": "one line or empty"}'
    )

    user_content = (
        "Today: " + today + "\n\n"
        "FULL THREAD:\n" + thread_section + "\n\n"
        "USER'S DRAFT:\n---\n" + draft.strip() + "\n---\n\n"
        "Return the polished version. JSON only."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# ── Groq caller ───────────────────────────────────────────────────────────────

async def call_groq(messages: list[dict], max_tokens: int = 1024) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + GROQ_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "messages": messages
            }
        )

    if response.status_code != 200:
        try:
            msg = response.json().get("error", {}).get("message", "Groq API error")
        except Exception:
            msg = "Groq API error " + str(response.status_code)
        raise HTTPException(status_code=502, detail=msg)

    return response.json()["choices"][0]["message"]["content"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ThinkMail API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/auth/google")
async def auth_google():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured.")
    params = "&".join([
        "response_type=code",
        "client_id=" + GOOGLE_CLIENT_ID,
        "redirect_uri=" + REDIRECT_URI,
        "scope=openid%20email%20profile",
        "access_type=offline",
        "prompt=consent"
    ])
    return RedirectResponse("https://accounts.google.com/o/oauth2/v2/auth?" + params)

@app.get("/auth/callback")
async def auth_callback(code: str, request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth not configured.")

    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code"
            }
        )
        if token_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange OAuth code.")

        access_token = token_response.json().get("access_token")
        user_response = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": "Bearer " + access_token}
        )
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info.")

        user_info = user_response.json()

    user_id = hashlib.sha256(user_info["email"].encode()).hexdigest()[:16]
    name = user_info.get("name", "")
    email = user_info["email"]

    await upsert_user(email=email, name=name, user_id=user_id)

    jwt_token = create_jwt(user_id, email, name)
    return RedirectResponse(
        FRONTEND_URL + "/auth/extension-callback?token=" + jwt_token + "&name=" + name + "&email=" + email
    )

@app.get("/auth/extension-callback")
async def extension_callback(token: str, name: str = "", email: str = ""):
    html = (
        "<!DOCTYPE html><html><head><title>ThinkMail</title>"
        "<style>*{box-sizing:border-box;margin:0;padding:0}"
        "body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;"
        "height:100vh;background:#0d0f10;color:#edf0f2;flex-direction:column;gap:14px;text-align:center;padding:24px}"
        ".icon{width:52px;height:52px;border-radius:14px;"
        "background:linear-gradient(135deg,#4285F4,#34A853);display:flex;"
        "align-items:center;justify-content:center;font-size:22px;margin-bottom:4px}"
        "h2{font-size:20px;font-weight:600}"
        "p{color:#8d9499;font-size:13px;line-height:1.6}"
        ".em{color:#4285F4;font-size:13px}</style></head><body>"
        "<div class='icon'>✦</div>"
        "<h2>Signed in!</h2>"
        "<div class='em'>" + email + "</div>"
        "<p>You can close this tab and go back to Gmail.</p>"
        "<script>"
        "try{localStorage.setItem('mailmind_token','" + token + "');"
        "localStorage.setItem('mailmind_name','" + name + "');"
        "localStorage.setItem('mailmind_email','" + email + "');}catch(e){}"
        "setTimeout(()=>{window.location.href='https://mail.google.com';},2000);"
        "</script></body></html>"
    )
    return HTMLResponse(html)


@app.post("/fix", response_model=FixResponse)
@limiter.limit("30/minute")
async def fix_email(
    request: Request,
    body: FixRequest,
    user: dict = Depends(get_current_user)
):
    user_id = user["sub"]
    try:
        fixes_used, fixes_remaining = await increment_usage(user_id)
    except Exception as e:
        raise HTTPException(status_code=429, detail=str(e))

    today = datetime.now().strftime("%A, %B %d %Y")
    messages = build_prompt(body.thread or "", body.draft or "", today)
    result = await call_groq(messages, max_tokens=1024)
    return FixResponse(result=result, fixes_used=fixes_used, fixes_remaining=fixes_remaining)


@app.post("/coach", response_model=CoachResponse)
@limiter.limit("60/minute")
async def coach_email(
    request: Request,
    body: CoachRequest,
    user: dict = Depends(get_current_user)
):
    """
    Live draft polish — called once per draft session (client locks after first call).
    Does NOT count against the user's daily fix quota.
    Returns { improved, note } — improved is always the best version of their draft.
    """
    if not body.draft or len(body.draft.strip()) < 15:
        return CoachResponse(improved=body.draft or "", note="")

    today = datetime.now().strftime("%A, %B %d %Y")
    messages = build_coach_prompt(body.thread or "", body.draft, today)
    raw = await call_groq(messages, max_tokens=400)

    try:
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(cleaned)
        improved = str(data.get("improved", body.draft)).strip()
        note = str(data.get("note", "")).strip()
        # If model returned the exact same text, note should be empty
        if improved == body.draft.strip():
            note = ""
        return CoachResponse(improved=improved, note=note)
    except (json.JSONDecodeError, KeyError):
        # Parsing failed — return original draft silently, don't show broken card
        return CoachResponse(improved=body.draft, note="")


@app.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    return await get_user_stats(user["sub"])

@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"email": user.get("email"), "name": user.get("name"), "user_id": user.get("sub")}

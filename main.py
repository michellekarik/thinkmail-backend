"""
ThinkMail Backend
- Google OAuth login
- Groq API proxy (70b for analysis, 8b-instant for coach)
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
    improved: str
    note: str
    what_was_wrong: str


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


# ── Groq caller ───────────────────────────────────────────────────────────────

async def call_groq(messages: list[dict], max_tokens: int = 1024, model: str = "llama-3.3-70b-versatile") -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured.")

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + GROQ_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "model": model,
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


# ── Fallback reply generator ──────────────────────────────────────────────────
# If the model skips SUGGESTED REPLY, call it again just for the reply.
# This guarantees a reply is ALWAYS returned — no exceptions.

async def get_fallback_reply(thread: str, draft: str, today: str) -> str:
    has_draft = draft.strip()
    prompt = (
        "You are ThinkMail. Today: " + today + "\n\n"
        "EMAIL THREAD:\n" + (thread.strip() or "No thread available.") + "\n\n"
        + ("USER DRAFT:\n" + draft.strip() + "\n\n" if has_draft else "")
        + "IMPORTANT: The user is the person who needs to REPLY — NOT the person who sent the last message. "
        + "The last message was sent TO the user. Write the reply FROM the user, TO the other person. "
        + "Open with the OTHER person's name. Sign off with the USER's name. "
        + "Match the format, opener, sign-off, length and tone of the thread history exactly. "
        + ("Improve and complete the user's draft. " if has_draft else "Write from scratch. ")
        + "Sound like a confident real human. Return ONLY the reply text, nothing else."
    )
    messages = [{"role": "user", "content": prompt}]
    return await call_groq(messages, max_tokens=512, model="llama-3.3-70b-versatile")


# ── /fix prompt ───────────────────────────────────────────────────────────────

def build_prompt(thread: str, draft: str, today: str) -> list[dict]:
    has_draft = draft.strip()
    draft_section = (
        "USER HAS ALREADY TYPED THIS DRAFT:\n---\n" + draft + "\n---\n\n"
        "The user has started writing. Understand what they are trying to say and give them "
        "the best version of it for this specific situation and relationship."
        if has_draft else
        "USER HAS NO DRAFT — write the reply entirely from scratch based on the full thread context."
    )
    thread_section = thread.strip() if thread.strip() else "No thread — only a draft is available."

    if not thread.strip() and not draft.strip():
        user_content = "Tell the user to open a Gmail thread first."
    else:
        user_content = (
            "Today: " + today + "\n\n"
            + thread_section + "\n\n"
            + draft_section + "\n\n"
            + """Read every email in the thread carefully before responding.

STEP 1 — EXTRACT THE EMAIL FORMAT:
Study every email in the thread and note exactly:
- Opener style: Hey [name], Hi, Dear, straight to point, or nothing?
- Sign-off: Thanks [name], Regards, Cheers, name only, phone number, or nothing?
- Length: short punchy lines or full paragraphs?
- Formality: close friend, colleague, boss, client, stranger?
- Any recurring patterns: bullet points, line breaks, specific phrases they always use?

FORMAT RULE — non-negotiable:
If there IS a back-and-forth history between these two people in the thread:
  → The SUGGESTED REPLY must mirror that exact format. Same opener, same sign-off, same length, same tone.
  → Example: if every email ends "Thanks, Priya" — end with that. If they write short 2-line replies — do that.
  → Never invent a format that does not already exist in their history.

If this is the FIRST email or there is NO prior reply history (only one email in the thread):
  → Use a clean professional default: opener with the person's name, clear body, sign off with the user's name.
  → Match the formality level of the incoming email — if they wrote formally, reply formally. If casual, casual.
  → Example default format:
     Hi [Name],
     [Body of reply]
     Thanks,
     [User's name]

STEP 2 — IDENTIFY WHO IS WHO:
The person who sent the LAST message in the thread is the OTHER person — not the user.
The user is whoever needs to REPLY. Look at earlier emails to find the user's name and sign-off style.
The SUGGESTED REPLY opens with the OTHER person's name and signs off with the USER's name.

STEP 3 — DETECT LANGUAGE:
If the user typed a draft — check what language it is in.
If the draft is in Hindi, Japanese, Spanish, Tamil, French, or any non-English language — write the SUGGESTED REPLY in that exact language.
If the draft is in English or there is no draft — write the SUGGESTED REPLY in English.
TONE, URGENCY, VIBE, INTENT, RISK, SITUATION, CONTEXT ANALYSIS, CONFLICTS are always in English.

STEP 4 — RESPOND IN THIS EXACT FORMAT. ALL SECTIONS ARE MANDATORY. DO NOT SKIP ANY:

TONE: [one word: Formal / Casual / Tense / Friendly / Aggressive / Professional]
URGENCY: [one word: Low / Medium / High]
VIBE: [one word: Positive / Neutral / Tense / Hostile / Warm]
INTENT: [one short phrase — what does the other person actually want?]
RISK: [one word: Low / Medium / High]

SITUATION:
[3-4 plain English sentences. What is happening? Who are these people, what is the history, what is the current moment?]

CONTEXT ANALYSIS:
[Power dynamic, tone shifts, promises made or broken, what the user must understand before replying.]

CONFLICTS:
[Did the user's draft miss or contradict something from the thread? Is their tone wrong for this relationship? If nothing is wrong: No conflicts detected.]

SUGGESTED REPLY:
[MANDATORY — you must always write this. Never skip it under any circumstances.
If they typed a draft: keep their intent, fix the delivery for this situation.
If no draft: write from scratch based on the full thread.
FORMAT RULE — non-negotiable: match the exact opener, sign-off, length and formality from the thread history. If emails end with "Thanks, Priya" use that. If casual with no sign-off, do that. Never invent a format that does not exist in the history.
Sound like a confident real human. No filler. No pushover replies.]"""
        )

    # Figure out who the user is from the last email they sent in the thread
    # The user is the person who RECEIVES the most recent message and needs to reply
    user_identity_instruction = (
        "CRITICAL — WHO IS THE USER:\n"
        "The USER is the person reading this right now who needs to send a reply. "
        "They are NOT the person who sent the last message. "
        "The last message in the thread was sent TO the user — the user needs to reply TO that person. "
        "Read the thread carefully to identify who the user is (they will appear as a sender in earlier emails). "
        "The SUGGESTED REPLY must be written FROM the user, TO the other person. "
        "Open with the OTHER person's name. Sign off with the USER's name. Never mix these up.\n\n"

        "LANGUAGE RULE:\n"
        "If the user has typed a draft, detect what language it is written in.\n"
        "- If the draft is in a non-English language (Hindi, Japanese, Spanish, French, Tamil, etc.) "
        "— write the SUGGESTED REPLY in that same language.\n"
        "- If the draft is in English or there is no draft — write the SUGGESTED REPLY in English.\n"
        "- All other sections (TONE, URGENCY, VIBE, INTENT, RISK, SITUATION, CONTEXT ANALYSIS, CONFLICTS) "
        "must always be in English regardless of the reply language.\n\n"
    )

    system_content = (
        "You are ThinkMail — email situational intelligence.\n"
        "Today: " + today + "\n\n"
        + user_identity_instruction +
        "Rules you never break:\n"
        "1. Read the FULL thread — every email, not just the last one.\n"
        "2. You are on the USER's side. Always. Write the reply AS the user, TO the other person.\n"
        "3. If the user typed a draft, understand what they meant and say it right for this situation.\n"
        "4. SUGGESTED REPLY is ALWAYS required. You must write it every single time without exception.\n"
        "5. FORMAT RULE: If there is back-and-forth history — mirror it exactly: same opener, "
        "same sign-off, same length, same formality. If it is the first email or no prior replies exist — "
        "use a clean professional default matching the formality of the incoming email: "
        "opener with their name, clear body, sign off with user's name.\n"
        "6. Sound like a real human. No generic AI text.\n"
        "Always follow the exact output format. All six sections are mandatory."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# ── /coach prompt ─────────────────────────────────────────────────────────────

def build_coach_prompt(thread: str, draft: str, today: str) -> list[dict]:
    thread_section = thread.strip() if thread.strip() else "No prior thread."

    system_content = (
        "You are ThinkMail, email situational coach. Today: " + today + "\n"
        "User typed a draft reply. Do three things:\n"
        "1. DIAGNOSE: Is tone/vibe wrong for this relationship and situation? "
        "Too aggressive, too weak, too formal, off-vibe? One sentence.\n"
        "2. FORMAT: From the thread extract opener style, sign-off, length, formality. "
        "The rewrite MUST match this exactly.\n"
        "3. REWRITE: Same message, right tone, correct format from step 2. "
        "Confident real human. No filler. No new content.\n\n"
        "Return ONLY valid JSON, no markdown, no explanation:\n"
        '{"what_was_wrong":"one sentence diagnosis or empty if fine",'
        '"improved":"full rewritten reply",'
        '"note":"what changed or empty"}'
    )

    user_content = (
        "THREAD:\n" + thread_section +
        "\n\nDRAFT:\n" + draft.strip() +
        "\n\nReturn JSON only."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content}
    ]


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
    name    = user_info.get("name", "")
    email   = user_info["email"]

    await upsert_user(email=email, name=name, user_id=user_id)
    jwt_token = create_jwt(user_id, email, name)
    return RedirectResponse(
        FRONTEND_URL + "/auth/extension-callback?token=" + jwt_token +
        "&name=" + name + "&email=" + email
    )

@app.get("/auth/extension-callback")
async def extension_callback(token: str, name: str = "", email: str = ""):
    html = (
        "<!DOCTYPE html><html><head><title>ThinkMail</title>"
        "<style>*{box-sizing:border-box;margin:0;padding:0}"
        "body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;"
        "height:100vh;background:#0d0f10;color:#edf0f2;flex-direction:column;gap:14px;"
        "text-align:center;padding:24px}"
        ".icon{width:52px;height:52px;border-radius:14px;"
        "background:linear-gradient(135deg,#4285F4,#34A853);display:flex;"
        "align-items:center;justify-content:center;font-size:22px;margin-bottom:4px}"
        "h2{font-size:20px;font-weight:600}p{color:#8d9499;font-size:13px;line-height:1.6}"
        ".em{color:#4285F4;font-size:13px}</style></head><body>"
        "<div class='icon'>✦</div><h2>Signed in!</h2>"
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

    today    = datetime.now().strftime("%A, %B %d %Y")
    messages = build_prompt(body.thread or "", body.draft or "", today)
    result   = await call_groq(messages, max_tokens=1200, model="llama-3.3-70b-versatile")

    # ── Guarantee a reply is always present ──────────────────────────────────
    # If the model skipped SUGGESTED REPLY for any reason, call again just for it
    reply_match = __import__('re').search(r'SUGGESTED REPLY:\s*([\s\S]*)', result, __import__('re').IGNORECASE)
    reply_text  = reply_match.group(1).strip() if reply_match else ""

    if not reply_text:
        fallback = await get_fallback_reply(body.thread or "", body.draft or "", today)
        result = result.rstrip() + "\n\nSUGGESTED REPLY:\n" + fallback

    return FixResponse(result=result, fixes_used=fixes_used, fixes_remaining=fixes_remaining)


@app.post("/coach", response_model=CoachResponse)
@limiter.limit("60/minute")
async def coach_email(
    request: Request,
    body: CoachRequest,
    user: dict = Depends(get_current_user)
):
    if not body.draft or len(body.draft.strip()) < 3:
        return CoachResponse(improved="", note="", what_was_wrong="")

    today    = datetime.now().strftime("%A, %B %d %Y")
    messages = build_coach_prompt(body.thread or "", body.draft, today)
    raw      = await call_groq(messages, max_tokens=350, model="llama-3.1-8b-instant")

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        data = json.loads(cleaned)
        return CoachResponse(
            what_was_wrong = str(data.get("what_was_wrong", "")).strip(),
            improved       = str(data.get("improved", body.draft)).strip(),
            note           = str(data.get("note", "")).strip()
        )
    except (json.JSONDecodeError, KeyError):
        return CoachResponse(improved=body.draft, note="", what_was_wrong="")


@app.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    return await get_user_stats(user["sub"])

@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"email": user.get("email"), "name": user.get("name"), "user_id": user.get("sub")}

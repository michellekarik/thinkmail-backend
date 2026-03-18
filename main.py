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
# model param lets /fix use the big accurate model,
# while /coach uses llama-3.1-8b-instant (5-10x faster, good enough for polish)

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


# ── /fix prompt ───────────────────────────────────────────────────────────────

def build_prompt(thread: str, draft: str, today: str) -> list[dict]:
    has_draft = draft.strip()
    draft_section = (
        "USER HAS ALREADY TYPED THIS DRAFT:\n---\n" + draft + "\n---\n\n"
        "Understand what they are trying to say and give them the best version for this situation."
        if has_draft else
        "USER HAS NO DRAFT — write the reply from scratch based on the full thread."
    )
    thread_section = thread.strip() if thread.strip() else "No thread — only a draft is available."

    if not thread.strip() and not draft.strip():
        user_content = "Tell the user to open a Gmail thread first."
    else:
        user_content = (
            "Today: " + today + "\n\n"
            + thread_section + "\n\n"
            + draft_section + "\n\n"
            + """Read every email in the thread carefully. Understand the full history, relationship, power dynamic.

Respond in EXACTLY this format:

TONE: [one word: Formal / Casual / Tense / Friendly / Aggressive / Professional]
URGENCY: [one word: Low / Medium / High]
VIBE: [one word: Positive / Neutral / Tense / Hostile / Warm]
INTENT: [one short phrase — what does the other person actually want?]
RISK: [one word: Low / Medium / High]

SITUATION:
[3-4 sentences. What is happening? Who are these people, what is the history, what is the current moment?]

CONTEXT ANALYSIS:
[Power dynamic, promises made, tone shifts, what the user needs to know before replying.]

CONFLICTS:
[Did the draft miss or contradict something from the thread? Is the user's tone wrong for the relationship? If nothing: No conflicts detected.]

SUGGESTED REPLY:
[The reply the user should send. If they wrote a draft — this is the corrected situationally-aware version. If no draft — write from scratch. Match the relationship format: opener, sign-off, length, formality from thread history. Sound like a real human.]"""
        )

    system_content = (
        "You are ThinkMail — email situational intelligence.\n"
        "Today: " + today + "\n\n"
        "You read the FULL thread and understand the relationship history.\n"
        "You are on the USER's side. Always.\n"
        "If the user wrote a draft, understand what they meant and help them say it right.\n"
        "SUGGESTED REPLY must match the format and style of this specific email relationship.\n"
        "Always follow the exact output format."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# ── /coach prompt ─────────────────────────────────────────────────────────────
# Compact prompt for llama-3.1-8b-instant — fast, focused, no fluff

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
        '"note":"what changed, e.g. matched casual tone added sign-off or empty"}'
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
    # Full analysis uses the big accurate model
    result   = await call_groq(messages, max_tokens=1024, model="llama-3.3-70b-versatile")
    return FixResponse(result=result, fixes_used=fixes_used, fixes_remaining=fixes_remaining)


@app.post("/coach", response_model=CoachResponse)
@limiter.limit("60/minute")
async def coach_email(
    request: Request,
    body: CoachRequest,
    user: dict = Depends(get_current_user)
):
    """
    Live draft coach — called once when user pauses typing.
    Uses llama-3.1-8b-instant: 5-10x faster than 70b, sub-2s responses.
    Does NOT count against daily fix quota.
    """
    if not body.draft or len(body.draft.strip()) < 3:
        return CoachResponse(improved="", note="", what_was_wrong="")

    today    = datetime.now().strftime("%A, %B %d %Y")
    messages = build_coach_prompt(body.thread or "", body.draft, today)
    # Fast model for real-time coaching
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

"""
MailMind Backend
- Google OAuth login
- Groq API proxy (your key, user never sees it)
- Rate limiting (20 fixes/day free tier)
- Zero email logging — email content is never stored
"""

import os
import time
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

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET           = os.getenv("JWT_SECRET", "change-this-in-production")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:8000")
REDIRECT_URI         = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")
FREE_TIER_LIMIT      = int(os.getenv("FREE_TIER_LIMIT", "20"))

usage_tracker: dict = {}

# ── App ───────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="MailMind API", version="1.0.0")
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

# ── Usage tracking ────────────────────────────────────────────────────────────
def check_and_increment_usage(user_id: str) -> tuple[int, int]:
    now = time.time()
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    reset_at = (midnight + timedelta(days=1)).timestamp()

    if user_id not in usage_tracker or usage_tracker[user_id]["reset_at"] <= now:
        usage_tracker[user_id] = {"count": 0, "reset_at": reset_at}

    current = usage_tracker[user_id]["count"]
    if current >= FREE_TIER_LIMIT:
        reset_time = datetime.fromtimestamp(reset_at).strftime("%H:%M")
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {FREE_TIER_LIMIT} fixes reached. Resets at midnight ({reset_time})."
        )

    usage_tracker[user_id]["count"] += 1
    new_count = usage_tracker[user_id]["count"]
    return new_count, FREE_TIER_LIMIT - new_count

# ── Groq helper ───────────────────────────────────────────────────────────────
def build_prompt(thread: str, draft: str, today: str) -> list[dict]:
    draft_section = (
        f"USER'S DRAFT REPLY:\n---\n{draft}\n---"
        if draft.strip()
        else "USER'S DRAFT REPLY: None — figure out the right reply entirely from context."
    )

    if not thread.strip() and not draft.strip():
        content = "Tell the user to open an email thread in Gmail first."
    elif not thread.strip():
        content = f"Review this draft and suggest improvements.\n\n{draft_section}"
    else:
        content = f"""Today's date: {today}

FULL EMAIL THREAD (oldest to newest):
---
{thread}
---

{draft_section}

Analyze this thread completely and respond in EXACTLY this format:

SITUATION:
[2-3 sentences: what is actually happening, who said what, what was agreed or promised]

CONFLICTS DETECTED:
[Any timeline violations, contradictions, broken promises, or unreasonable requests. Be specific. If none, say "No conflicts detected."]

SUGGESTED REPLY:
[The reply the user should send. If the other party is contradicting themselves or being unreasonable, call it out professionally. Never write a pushover reply. No filler openers. Match the user's tone.]"""

    return [
        {
            "role": "system",
            "content": f"""You are MailMind — an intelligent email context detective.
Today's date is {today}.
You are on the USER's side. Always.
Never suggest sycophantic replies like "Yes sir, understood."
If someone is contradicting a prior agreement, call it out clearly and professionally.
If a deadline was set and is being violated, reference it specifically.
Sound like a real confident human, not a corporate robot."""
        },
        {"role": "user", "content": content}
    ]

async def call_groq(messages: list[dict]) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured on server.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 1024,
                "temperature": 0.3,
                "messages": messages
            }
        )

    if response.status_code != 200:
        try:
            msg = response.json().get("error", {}).get("message", "Groq API error")
        except:
            msg = f"Groq API error ({response.status_code})"
        raise HTTPException(status_code=502, detail=msg)

    return response.json()["choices"][0]["message"]["content"]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "MailMind API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

# ── Google OAuth ──────────────────────────────────────────────────────────────

@app.get("/auth/google")
async def auth_google():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured.")
    params = "&".join([
        "response_type=code",
        f"client_id={GOOGLE_CLIENT_ID}",
        f"redirect_uri={REDIRECT_URI}",
        "scope=openid%20email%20profile",
        "access_type=offline",
        "prompt=consent"
    ])
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")

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
            headers={"Authorization": f"Bearer {access_token}"}
        )
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info.")

        user_info = user_response.json()

    user_id = hashlib.sha256(user_info["email"].encode()).hexdigest()[:16]
    jwt_token = create_jwt(user_id, user_info["email"], user_info.get("name", ""))

    name = user_info.get("name", "")
    email = user_info["email"]

    return RedirectResponse(
        f"{FRONTEND_URL}/auth/extension-callback?token={jwt_token}&name={name}&email={email}"
    )

# ── Extension callback page ───────────────────────────────────────────────────

@app.get("/auth/extension-callback")
async def extension_callback(token: str, name: str = "", email: str = ""):
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>MailMind — Signed in!</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Google Sans', sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    background: #111214;
    color: #e8eaed;
    flex-direction: column;
    gap: 14px;
    text-align: center;
    padding: 24px;
  }}
  .icon {{
    width: 56px; height: 56px; border-radius: 16px;
    background: linear-gradient(135deg, #EA4335, #4285F4);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; margin-bottom: 4px;
  }}
  h2 {{ font-size: 20px; font-weight: 600; }}
  p {{ color: #9aa0a6; font-size: 14px; line-height: 1.6; }}
  .email {{ color: #4285F4; font-size: 13px; }}
</style>
</head>
<body>
<div class="icon">✦</div>
<h2>Signed in successfully!</h2>
<div class="email">{email}</div>
<p>You can close this tab and go back to Gmail.<br>MailMind is ready to use.</p>
<script>
  // Pass token back to the Chrome extension background script
  // The background script is listening for this URL pattern
  console.log('MailMind auth callback loaded');
  
  // Store in localStorage as fallback
  try {{
    localStorage.setItem('mailmind_token', '{token}');
    localStorage.setItem('mailmind_name', '{name}');
    localStorage.setItem('mailmind_email', '{email}');
  }} catch(e) {{}}

  // Auto close after 3 seconds
  setTimeout(() => window.close(), 3000);
</script>
</body>
</html>"""
    return HTMLResponse(html)

# ── Fix endpoint ──────────────────────────────────────────────────────────────

@app.post("/fix", response_model=FixResponse)
@limiter.limit("30/minute")
async def fix_email(
    request: Request,
    body: FixRequest,
    user: dict = Depends(get_current_user)
):
    """PRIVACY: Email content is NEVER logged or stored."""
    user_id = user["sub"]
    fixes_used, fixes_remaining = check_and_increment_usage(user_id)
    today = datetime.now().strftime("%A, %B %d %Y")
    messages = build_prompt(body.thread or "", body.draft or "", today)
    result = await call_groq(messages)
    return FixResponse(result=result, fixes_used=fixes_used, fixes_remaining=fixes_remaining)

@app.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    user_id = user["sub"]
    now = time.time()
    if user_id not in usage_tracker or usage_tracker[user_id]["reset_at"] <= now:
        return {"fixes_used": 0, "fixes_remaining": FREE_TIER_LIMIT, "limit": FREE_TIER_LIMIT}
    count = usage_tracker[user_id]["count"]
    return {"fixes_used": count, "fixes_remaining": max(0, FREE_TIER_LIMIT - count), "limit": FREE_TIER_LIMIT}

@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"email": user.get("email"), "name": user.get("name"), "user_id": user.get("sub")}

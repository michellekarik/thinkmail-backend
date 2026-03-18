"""
MailMind Backend
- Google OAuth login
- Groq API proxy
- Supabase database for user tracking
- Zero email logging
"""

import os
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

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET           = os.getenv("JWT_SECRET", "change-this-in-production")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:8000")
REDIRECT_URI         = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")
FREE_TIER_LIMIT      = int(os.getenv("FREE_TIER_LIMIT", "20"))

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

# ── Groq helper ───────────────────────────────────────────────────────────────
def build_prompt(thread: str, draft: str, today: str) -> list[dict]:
    draft_section = (
        f"USER'S DRAFT REPLY:
---
{draft}
---"
        if draft.strip()
        else "USER'S DRAFT REPLY: None — the user hasn't typed anything. Write the reply from scratch based on the full thread context."
    )

    if not thread.strip() and not draft.strip():
        content = "Tell the user to open a Gmail thread first."
    else:
        content = f"""Today's date: {today}

{thread if thread.strip() else "No thread found — only a draft is available."}

{draft_section}

You are ThinkMail. Read every single email in the thread above carefully before responding.
Understand who said what, when, what was promised, what changed, and what the current situation actually is.

Respond in EXACTLY this format — nothing outside these sections:

TONE: [one word: Formal / Casual / Tense / Friendly / Aggressive / Professional]
URGENCY: [one word: Low / Medium / High]
VIBE: [one word: Positive / Neutral / Tense / Hostile / Warm]
INTENT: [one short phrase — what does the other person actually want from this email?]
RISK: [one word: Low / Medium / High — how risky is it to reply wrong here?]

SITUATION:
[3-4 plain English sentences. Summarize the ENTIRE thread history — not just the last email. Who are the people involved? What has been said across all emails? What is the actual current situation? Write like you're explaining it to a friend who hasn't read any of it.]

CONTEXT ANALYSIS:
[What is the deeper context here? Is there a power dynamic? Has anything been promised and not delivered? Is the tone of the thread changing — friendly at first, now tense? What should the user be aware of before replying? Be specific and reference actual things said in the thread.]

CONFLICTS:
[Look across the ENTIRE thread. Did someone make a promise they're now contradicting? Is the user's draft ignoring something important that was said earlier? Is someone moving goalposts? If the draft contradicts the thread, call it out specifically. If nothing is wrong say exactly: No conflicts detected.]

SUGGESTED REPLY:
[Write the reply the user should actually send. Rules:
- Read the ENTIRE thread before writing this — the reply must make sense given everything that was said
- Sound like a real human, not an AI — no corporate speak, no filler
- Match the exact tone of the conversation — casual threads get casual replies
- Never start with "I hope this email finds you well" or any opener like that
- If the other person is wrong or contradicting something they said earlier, the reply should address that directly
- If no draft was provided, write the complete reply from scratch
- Be concise — every sentence should earn its place
- Never be a pushover]"""

    return [
        {{
            "role": "system",
            "content": f"""You are ThinkMail — an email situational intelligence assistant.
Today's date is {today}.

Your core job: Read the FULL email thread. Understand the complete history and context — not just the latest message. Catch what's contextually wrong even if it's grammatically fine.

You are on the USER's side. Always.
You think like a smart friend who read every email in the chain.
You catch timeline conflicts, broken promises, power dynamics, and meaning errors.
You write replies that sound human — confident, direct, appropriate.
You never generate generic AI text. You never suggest pushover replies.
You always follow the exact output format — every section, every time."""
        }},
        {{"role": "user", "content": content}}
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
    name = user_info.get("name", "")
    email = user_info["email"]

    # Save user to Supabase
    await upsert_user(email=email, name=name, user_id=user_id)

    jwt_token = create_jwt(user_id, email, name)
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
    display: flex; align-items: center; justify-content: center;
    height: 100vh; background: #111214; color: #e8eaed;
    flex-direction: column; gap: 14px; text-align: center; padding: 24px;
  }}
  .icon {{ width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, #EA4335, #4285F4); display: flex; align-items: center; justify-content: center; font-size: 24px; margin-bottom: 4px; }}
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
  try {{
    localStorage.setItem('mailmind_token', '{token}');
    localStorage.setItem('mailmind_name', '{name}');
    localStorage.setItem('mailmind_email', '{email}');
  }} catch(e) {{}}
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

    try:
        fixes_used, fixes_remaining = await increment_usage(user_id)
    except Exception as e:
        raise HTTPException(status_code=429, detail=str(e))

    today = datetime.now().strftime("%A, %B %d %Y")
    messages = build_prompt(body.thread or "", body.draft or "", today)
    result = await call_groq(messages)

    return FixResponse(result=result, fixes_used=fixes_used, fixes_remaining=fixes_remaining)

@app.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    stats = await get_user_stats(user["sub"])
    return stats

@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {
        "email": user.get("email"),
        "name": user.get("name"),
        "user_id": user.get("sub")
    }

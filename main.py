"""
ThinkMail Backend (FIXED)
"""

import os
import json
import re
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

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too many requests."})

# ── Models ─────────────────────────────────────

class FixRequest(BaseModel):
    thread: Optional[str] = ""
    draft: Optional[str] = ""

class FixResponse(BaseModel):
    result: str
    fixes_used: int
    fixes_remaining: int

# ── JWT ─────────────────────────────────────

def create_jwt(user_id: str, email: str, name: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": datetime.utcnow() + timedelta(days=30)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_jwt(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401)

def get_current_user(request: Request):
    auth = request.headers.get("Authorization", "")
    return verify_jwt(auth.split(" ")[1])

# ── GROQ ─────────────────────────────────────

async def call_groq(messages):
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.5,
                "messages": messages,
                "max_tokens": 1200
            }
        )
    return res.json()["choices"][0]["message"]["content"]

# ── PROMPT (FIXED CORE LOGIC) ─────────────────────────────────────

def build_prompt(thread: str, draft: str, today: str):

    return [
        {
            "role": "system",
            "content": f"""
You are ThinkMail — elite email strategist.

RULES:
- Analyze the thread deeply
- Copy tone EXACTLY
- Match opener + closing pattern
- NEVER be generic
- NEVER say "hope you're doing well"

OUTPUT FORMAT:

TONE:
URGENCY:
VIBE:
INTENT:
RISK:
SITUATION:
CONTEXT ANALYSIS:
CONFLICTS:
SUGGESTED REPLY:
"""
        },
        {
            "role": "user",
            "content": f"""
TODAY: {today}

THREAD:
{thread or "No thread"}

DRAFT:
{draft or "None"}

INSTRUCTIONS:
- Identify relationship
- Match style EXACTLY
- Reference real details
- Write like human
"""
        }
    ]

# ── AUTH ─────────────────────────────────────

@app.get("/auth/google")
async def auth_google():
    params = "&".join([
        "response_type=code",
        f"client_id={GOOGLE_CLIENT_ID}",
        f"redirect_uri={REDIRECT_URI}",
        "scope=openid email profile"
    ])
    return RedirectResponse("https://accounts.google.com/o/oauth2/v2/auth?" + params)

@app.get("/auth/callback")
async def callback(code: str):
    async with httpx.AsyncClient() as client:
        token = await client.post("https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        })

        access = token.json()["access_token"]

        user = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access}"}
        )

    data = user.json()
    user_id = hashlib.sha256(data["email"].encode()).hexdigest()[:16]

    await upsert_user(email=data["email"], name=data["name"], user_id=user_id)

    jwt_token = create_jwt(user_id, data["email"], data["name"])

    return RedirectResponse(
        f"/auth/extension-callback?token={jwt_token}&name={data['name']}&email={data['email']}"
    )

# ── FIXED SIGN-IN PAGE ─────────────────────────────────────

@app.get("/auth/extension-callback")
async def extension_callback(token: str, name: str = "", email: str = ""):
    return HTMLResponse(f"""
<html>
<body style="background:#0d1117;color:white;text-align:center;padding-top:100px">

<h1>✅ Signed in successfully</h1>
<p>Redirecting...</p>

<script>
chrome.runtime.sendMessage(
  "{{YOUR_EXTENSION_ID}}",
  {{
    action: "authComplete",
    token: "{token}",
    name: "{name}",
    email: "{email}"
  }}
);
window.close();
</script>

</body>
</html>
""")


# ── FIX ROUTE ─────────────────────────────────────

@app.post("/fix", response_model=FixResponse)
async def fix(body: FixRequest, user=Depends(get_current_user)):

    today = datetime.now().strftime("%B %d %Y")

    messages = build_prompt(body.thread, body.draft, today)

    result = ""

    for _ in range(3):
        result = await call_groq(messages)

        required = ["TONE:", "URGENCY:", "VIBE:", "INTENT:", "RISK:", "SUGGESTED REPLY:"]

        if all(x in result for x in required):
            break

    return FixResponse(result=result, fixes_used=1, fixes_remaining=49)

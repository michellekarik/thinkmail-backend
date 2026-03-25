"""
ThinkMail Backend — Production
- Google OAuth sign-in
- Groq AI analysis
- Supabase read receipt tracking
- JWT auth
"""

import os
import hashlib
import httpx
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from jose import jwt, JWTError

from database import upsert_user, increment_usage, get_user_stats

load_dotenv()

app = FastAPI(title="ThinkMail API", version="1.1.0")

# ── Env ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET           = os.getenv("JWT_SECRET", "thinkmail-secret-2026")
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY         = os.getenv("SUPABASE_SERVICE_KEY", "")
# Set BACKEND_URL in Vercel env vars to your deployment URL
# e.g. https://thinkmail-backend.vercel.app
BACKEND_URL          = os.getenv("BACKEND_URL", "https://thinkmail-backend.vercel.app")

# ── CORS ─────────────────────────────────────────────────────────────────────
# IMPORTANT: allow_credentials=True is incompatible with allow_origins=["*"]
# and will crash on Vercel. Keep credentials=False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase headers ──────────────────────────────────────────────────────────
def supa_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_jwt(user_id: str, email: str, name: str) -> str:
    return jwt.encode(
        {"sub": user_id, "email": email, "name": name,
         "exp": datetime.utcnow() + timedelta(days=30)},
        JWT_SECRET, algorithm="HS256"
    )

def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return jwt.decode(auth.split(" ", 1)[1], JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── Models ────────────────────────────────────────────────────────────────────
class FixRequest(BaseModel):
    thread: Optional[str] = ""
    draft:  Optional[str] = ""

class FixResponse(BaseModel):
    result:          str
    fixes_used:      int
    fixes_remaining: int

# ── AI prompt ─────────────────────────────────────────────────────────────────
def build_prompt(thread: str, draft: str, today: str) -> list[dict]:
    system = (
        "You are ThinkMail — situational email intelligence.\n"
        "Today: " + today + "\n\n"
        "RULES:\n"
        "1. Read the FULL thread before responding.\n"
        "2. NO filler phrases ('I hope you are well' etc) unless the history uses them.\n"
        "3. ENERGY MATCHING: mirror their length, punctuation and tone exactly.\n"
        "4. IDENTITY: open with the OTHER person's name, sign off with the USER's name.\n"
        "5. SUGGESTED REPLY is ALWAYS required — never skip it.\n\n"
        "Return EXACTLY this format:\n"
        "TONE: [one word]\n"
        "URGENCY: [Low/Medium/High]\n"
        "VIBE: [one word]\n"
        "INTENT: [one short phrase]\n"
        "RISK: [Low/Medium/High]\n\n"
        "SITUATION:\n[3-4 sentences]\n\n"
        "CONTEXT ANALYSIS:\n[power dynamic, what user must know]\n\n"
        "CONFLICTS:\n[issues with draft, or: No conflicts detected]\n\n"
        "SUGGESTED REPLY:\n[full reply mirroring thread format]"
    )
    user = (
        f"THREAD:\n{thread or 'No thread.'}\n\n"
        f"USER DRAFT:\n{draft or 'No draft — write from scratch.'}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ThinkMail Active", "version": "1.1.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ── Google OAuth ──────────────────────────────────────────────────────────────

@app.get("/auth/google")
async def auth_google():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    redirect_uri = BACKEND_URL + "/auth/callback"
    params = "&".join([
        "response_type=code",
        f"client_id={GOOGLE_CLIENT_ID}",
        f"redirect_uri={redirect_uri}",
        "scope=openid%20email%20profile",
        "access_type=offline",
        "prompt=consent",
    ])
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@app.get("/auth/callback")
async def auth_callback(code: str, request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    redirect_uri = BACKEND_URL + "/auth/callback"

    async with httpx.AsyncClient(timeout=15.0) as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            },
        )
        if token_res.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange OAuth code")

        access_token = token_res.json().get("access_token")
        user_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if user_res.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info")
        user_info = user_res.json()

    user_id   = hashlib.sha256(user_info["email"].encode()).hexdigest()[:16]
    name      = user_info.get("name", "")
    email     = user_info["email"]

    await upsert_user(email=email, name=name, user_id=user_id)
    token = create_jwt(user_id, email, name)

    # Redirect to Gmail — background.js watches for this hash and saves
    # the token into chrome.storage.local automatically
    import urllib.parse as _up
    params = _up.urlencode({"tm_token": token, "tm_name": name, "tm_email": email})
    return RedirectResponse(f"https://mail.google.com/#thinkmail-auth?{params}")


# ── Fix email ─────────────────────────────────────────────────────────────────

@app.post("/fix", response_model=FixResponse)
async def fix_email(request: Request, body: FixRequest, user: dict = Depends(get_current_user)):
    try:
        fixes_used, fixes_remaining = await increment_usage(user["sub"])
    except Exception as e:
        raise HTTPException(status_code=429, detail=str(e))

    today    = datetime.now().strftime("%A, %B %d %Y")
    messages = build_prompt(body.thread or "", body.draft or "", today)

    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages,
                  "temperature": 0.2, "max_tokens": 1200},
        )
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail="AI service error")
        result_text = res.json()["choices"][0]["message"]["content"]

    return FixResponse(result=result_text, fixes_used=fixes_used, fixes_remaining=fixes_remaining)


# ── Usage / Me ────────────────────────────────────────────────────────────────

@app.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    return await get_user_stats(user["sub"])

@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"email": user.get("email"), "name": user.get("name"), "user_id": user.get("sub")}


# ── Read Receipts ─────────────────────────────────────────────────────────────
# Supabase table needed:
#   CREATE TABLE track_receipts (
#     id         text PRIMARY KEY,
#     opened_at  timestamptz,
#     created_at timestamptz DEFAULT now()
#   );

_PIXEL = (
    b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00'
    b'\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9'
    b'\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00'
    b'\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
)

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}


@app.get("/track/{track_id}.png")
async def track_pixel(track_id: str):
    """Serve 1x1 transparent pixel and record first open in Supabase."""
    import re
    if SUPABASE_URL and SUPABASE_KEY and re.match(r'^[0-9a-f]{24}$', track_id):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                # Only write if not already recorded — preserve first-open time
                check = await client.get(
                    f"{SUPABASE_URL}/rest/v1/track_receipts?id=eq.{track_id}&select=opened_at",
                    headers=supa_headers(),
                )
                already = (
                    check.status_code == 200
                    and check.json()
                    and check.json()[0].get("opened_at") is not None
                )
                if not already:
                    await client.post(
                        f"{SUPABASE_URL}/rest/v1/track_receipts",
                        headers={**supa_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
                        json={"id": track_id, "opened_at": datetime.utcnow().isoformat(),
                              "created_at": datetime.utcnow().isoformat()},
                    )
        except Exception:
            pass  # Never block pixel delivery
    return Response(content=_PIXEL, media_type="image/gif", headers=_NO_CACHE)


@app.get("/track/{track_id}/status")
async def track_status(track_id: str, user: dict = Depends(get_current_user)):
    """Returns whether the tracked email has been opened. Polled by background.js."""
    import re
    if not re.match(r'^[0-9a-f]{24}$', track_id):
        raise HTTPException(status_code=400, detail="Invalid track ID")
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"opened": False, "openedAt": None}

    async with httpx.AsyncClient(timeout=5.0) as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/track_receipts?id=eq.{track_id}&select=opened_at",
            headers=supa_headers(),
        )
    if res.status_code != 200 or not res.json():
        return {"opened": False, "openedAt": None}

    opened_at = res.json()[0].get("opened_at")
    return {"opened": bool(opened_at), "openedAt": opened_at}

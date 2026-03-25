"""
ThinkMail Backend — api/index.py
Single self-contained file for Vercel serverless deployment.
All database logic is inlined here so Vercel doesn't need to resolve sibling imports.
"""

import os
import re
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

load_dotenv()

app = FastAPI(title="ThinkMail API", version="1.1.0")

# ── Env ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET           = os.getenv("JWT_SECRET", "thinkmail-secret-2026")
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY         = os.getenv("SUPABASE_SERVICE_KEY", "")
BACKEND_URL          = os.getenv("BACKEND_URL", "https://thinkmail-backend.vercel.app")
FREE_TIER_LIMIT      = int(os.getenv("FREE_TIER_LIMIT", "20"))

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase helpers ──────────────────────────────────────────────────────────
def supa_headers(extra: dict = {}):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    h.update(extra)
    return h

# ── Database helpers (inlined from database.py) ───────────────────────────────

async def upsert_user(email: str, name: str, user_id: str):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"{SUPABASE_URL}/rest/v1/users",
            headers=supa_headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
            json={
                "id": user_id, "email": email, "name": name,
                "last_seen": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),
            },
        )

async def increment_usage(user_id: str) -> tuple[int, int]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return 1, FREE_TIER_LIMIT - 1
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}"
            f"&select=fixes_today,fixes_today_reset_at,total_fixes",
            headers=supa_headers(),
        )
        if res.status_code != 200 or not res.json():
            raise Exception("User not found")
        user        = res.json()[0]
        fixes_today = user.get("fixes_today", 0) or 0
        total       = user.get("total_fixes", 0) or 0
        reset_str   = user.get("fixes_today_reset_at")
        now         = datetime.utcnow()

        if reset_str:
            reset_at = datetime.fromisoformat(reset_str.replace("Z", ""))
            if now >= reset_at:
                fixes_today = 0
        if fixes_today >= FREE_TIER_LIMIT:
            raise Exception(f"Daily limit of {FREE_TIER_LIMIT} fixes reached. Resets tomorrow.")

        new_count = fixes_today + 1
        midnight  = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}",
            headers=supa_headers({"Prefer": "return=minimal"}),
            json={
                "fixes_today": new_count,
                "fixes_today_reset_at": midnight.isoformat(),
                "total_fixes": total + 1,
                "last_seen": now.isoformat(),
            },
        )
    return new_count, FREE_TIER_LIMIT - new_count

async def get_user_stats(user_id: str) -> dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"fixes_used": 0, "fixes_remaining": FREE_TIER_LIMIT, "total_fixes": 0}
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}"
            f"&select=fixes_today,total_fixes,created_at,last_seen",
            headers=supa_headers(),
        )
    if res.status_code != 200 or not res.json():
        return {"fixes_used": 0, "fixes_remaining": FREE_TIER_LIMIT, "total_fixes": 0}
    user        = res.json()[0]
    fixes_today = user.get("fixes_today", 0) or 0
    return {
        "fixes_used":      fixes_today,
        "fixes_remaining": max(0, FREE_TIER_LIMIT - fixes_today),
        "total_fixes":     user.get("total_fixes", 0),
        "member_since":    user.get("created_at", ""),
        "last_seen":       user.get("last_seen", ""),
    }

# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_jwt(user_id: str, email: str, name: str) -> str:
    return jwt.encode(
        {"sub": user_id, "email": email, "name": name,
         "exp": datetime.utcnow() + timedelta(days=30)},
        JWT_SECRET, algorithm="HS256",
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
        f"Today: {today}\n\n"
        "RULES:\n"
        "1. Read the FULL thread before responding.\n"
        "2. NO filler phrases unless the thread uses them.\n"
        "3. Mirror their length, punctuation and tone exactly.\n"
        "4. Open with the OTHER person's name, sign off with the USER's name.\n"
        "5. SUGGESTED REPLY is always required — never skip it.\n\n"
        "Return EXACTLY this format:\n"
        "TONE: [one word]\nURGENCY: [Low/Medium/High]\nVIBE: [one word]\n"
        "INTENT: [one phrase]\nRISK: [Low/Medium/High]\n\n"
        "SITUATION:\n[3-4 sentences]\n\n"
        "CONTEXT ANALYSIS:\n[power dynamic, what user must know]\n\n"
        "CONFLICTS:\n[issues with draft, or: No conflicts detected]\n\n"
        "SUGGESTED REPLY:\n[full reply mirroring thread format exactly]"
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
                "code": code, "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri, "grant_type": "authorization_code",
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

    user_id = hashlib.sha256(user_info["email"].encode()).hexdigest()[:16]
    name    = user_info.get("name", "")
    email   = user_info["email"]

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
# Supabase table required:
#   CREATE TABLE track_receipts (
#     id         text PRIMARY KEY,
#     sent_at    timestamptz,
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

# Gmail proxy and email scanner User-Agents — these pre-fetch images on delivery,
# NOT when a human opens the email. We ignore pixel hits from these.
_BOT_UA = [
    "googleimageproxy", "google image proxy", "googleproxy",
    "mimecast", "barracuda", "proofpoint", "outlook-link-preview",
    "ms-office", "wget", "curl", "python-requests",
    "java/", "go-http-client", "apache-httpclient",
    "yahoo pipes", "preview",
]

def _is_bot(ua: str) -> bool:
    ua = (ua or "").lower()
    return any(frag in ua for frag in _BOT_UA)


@app.post("/track/{track_id}/register")
async def track_register(track_id: str, user: dict = Depends(get_current_user)):
    """
    Called by background.js immediately after email is sent.
    Records send time so the pixel endpoint can enforce the 45s grace period
    (ignoring Gmail's proxy which fires within seconds of delivery).
    """
    if not re.match(r'^[0-9a-f]{24}$', track_id):
        raise HTTPException(status_code=400, detail="Invalid track ID")
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/track_receipts",
                    headers=supa_headers({"Prefer": "resolution=ignore-duplicates,return=minimal"}),
                    json={
                        "id": track_id,
                        "sent_at":    datetime.utcnow().isoformat(),
                        "created_at": datetime.utcnow().isoformat(),
                        "opened_at":  None,
                    },
                )
        except Exception:
            pass
    return {"ok": True}


@app.get("/track/{track_id}.png")
async def track_pixel(track_id: str, request: Request):
    """
    Serve 1x1 pixel. Only record a real open if:
    1. User-Agent is NOT a known bot/proxy
    2. Hit arrives MORE than 45s after the email was sent
    """
    ua = request.headers.get("user-agent", "")

    if SUPABASE_URL and SUPABASE_KEY and re.match(r'^[0-9a-f]{24}$', track_id) and not _is_bot(ua):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                check = await client.get(
                    f"{SUPABASE_URL}/rest/v1/track_receipts"
                    f"?id=eq.{track_id}&select=sent_at,opened_at",
                    headers=supa_headers(),
                )
                rows = check.json() if check.status_code == 200 else []
                row  = rows[0] if rows else {}

                if not row.get("opened_at"):
                    # Enforce grace period
                    in_grace = False
                    sent_str = row.get("sent_at")
                    if sent_str:
                        try:
                            sent_dt  = datetime.fromisoformat(sent_str.replace("Z", ""))
                            in_grace = (datetime.utcnow() - sent_dt).total_seconds() < 45
                        except Exception:
                            pass

                    if not in_grace:
                        await client.patch(
                            f"{SUPABASE_URL}/rest/v1/track_receipts?id=eq.{track_id}",
                            headers=supa_headers({"Prefer": "return=minimal"}),
                            json={"opened_at": datetime.utcnow().isoformat()},
                        )
        except Exception:
            pass

    return Response(content=_PIXEL, media_type="image/gif", headers=_NO_CACHE)


@app.get("/track/{track_id}/status")
async def track_status(track_id: str, user: dict = Depends(get_current_user)):
    """Polled by background.js to check if email was opened."""
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

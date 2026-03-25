"""
ThinkMail Backend — Final Production Version
- Explicit 'app' for Vercel
- Mirroring & Intent Intelligence
- Fixed datetime and Auth dependency bugs
"""

import os
import json
import hashlib
import httpx
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from jose import jwt, JWTError

# Internal imports (Make sure database.py is in your repo)
from database import upsert_user, increment_usage, get_user_stats

load_dotenv()

# --- VERCEL CRITICAL: Explicit 'app' instance ---
app = FastAPI(title="ThinkMail API", version="1.1.0")

# Env Variables
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET           = os.getenv("JWT_SECRET", "thinkmail-secret-2026")
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY         = os.getenv("SUPABASE_SERVICE_KEY", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth Helper ---
def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        token = auth.split(" ", 1)[1]
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- Models ---
class FixRequest(BaseModel):
    thread: Optional[str] = ""
    draft:  Optional[str] = ""

class FixResponse(BaseModel):
    result:          str
    fixes_used:      int
    fixes_remaining: int

# --- AI Logic: Mirroring & Intent Extraction ---
def build_mirror_prompt(thread: str, draft: str, today: str) -> list[dict]:
    system_content = (
        "You are ThinkMail — situational intelligence.\n"
        "GOAL: Ensure the user wins the interaction by matching the room exactly.\n\n"
        "RULES:\n"
        "1. NO AI-FILLER: Absolutely no 'I hope you are well' or corporate fluff unless the history uses it.\n"
        "2. ENERGY MATCHING: If they write short, you write short. Match their energy and punctuation.\n"
        "3. INTENT: Is the contact stalling or annoyed? Protect the user's leverage.\n"
        "4. IDENTITY: Sign off with the USER'S name. Address the OTHER person by name."
    )
    user_content = f"TODAY: {today}\nTHREAD: {thread}\nUSER DRAFT: {draft}\n\nReturn analysis sections (TONE, URGENCY, VIBE, INTENT, RISK, SITUATION, CONTEXT, CONFLICTS) and the SUGGESTED REPLY."
    return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

# --- Routes ---
@app.get("/")
async def root():
    return {"status": "ThinkMail Active", "mode": "Situational Intelligence"}

@app.post("/fix", response_model=FixResponse)
async def fix_email(request: Request, body: FixRequest, user: dict = Depends(get_current_user)):
    user_id = user["sub"]
    try:
        # DB usage tracking
        fixes_used, fixes_remaining = await increment_usage(user_id)
    except Exception as e:
        raise HTTPException(status_code=429, detail=str(e))

    today = datetime.now().strftime("%A, %B %d %Y")
    messages = build_mirror_prompt(body.thread or "", body.draft or "", today)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.2
            }
        )
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail="AI Service Error")
            
        data = res.json()
        result_text = data["choices"][0]["message"]["content"]

    return FixResponse(result=result_text, fixes_used=fixes_used, fixes_remaining=fixes_remaining)

# Tracking Pixel (1x1 Transparent GIF)
# ── Read Receipts ─────────────────────────────────────────────────────────────
# Supabase table: track_receipts
#   id          text PRIMARY KEY
#   sent_at     timestamptz   -- when the email was sent (set by /track/register)
#   opened_at   timestamptz   -- null until a REAL human open is detected
#   created_at  timestamptz

_PIXEL = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'

_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

@app.post("/track/{track_id}/register")
async def track_register(track_id: str, user: dict = Depends(get_current_user)):
    """Called by background.js after send. Creates the row in Supabase."""
    import re as _re
    if not _re.match(r'^[0-9a-f]{24}$', track_id):
        raise HTTPException(status_code=400, detail="Invalid track ID")
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/track_receipts",
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "resolution=ignore-duplicates,return=minimal",
                    },
                    json={
                        "id":         track_id,
                        "opened_at":  None,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )
        except Exception:
            pass
    return {"ok": True}


@app.get("/track/{track_id}.png")
async def track_pixel(track_id: str, request: Request):
    """
    Serve the 1x1 pixel.

    Uses a Supabase RPC to do a safe conditional upsert in one round-trip:
      INSERT ... ON CONFLICT (id) DO UPDATE SET opened_at = now()
      WHERE track_receipts.opened_at IS NULL
    This means: create the row if missing (pixel arrived before /register),
    set opened_at if not yet set, and leave it alone if already recorded.
    Atomic, no race condition, preserves the FIRST open timestamp always.
    """
    import re as _re
    if SUPABASE_URL and SUPABASE_KEY and _re.match(r'^[0-9a-f]{24}$', track_id):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/rpc/record_open",
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"p_id": track_id},
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
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
        )
    if res.status_code != 200 or not res.json():
        return {"opened": False, "openedAt": None}
    opened_at = res.json()[0].get("opened_at")
    return {"opened": bool(opened_at), "openedAt": opened_at}

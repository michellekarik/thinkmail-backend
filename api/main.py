import os
import re
import httpx
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, RedirectResponse
from jose import jwt

# ========================
# APP INIT (CRITICAL)
# ========================
app = FastAPI()

# ========================
# ENV
# ========================
JWT_SECRET = os.getenv("JWT_SECRET", "secret")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# ⚠️ CHANGE THIS TO YOUR DOMAIN
BASE_URL = "https://thinkmail-backend.vercel.app"

# ========================
# CORS
# ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# AUTH
# ========================
def get_user(req: Request):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401)
    try:
        token = auth.split(" ")[1]
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except:
        raise HTTPException(401)

# ========================
# ROOT (FIXES 404)
# ========================
@app.get("/")
def root():
    return {"status": "ThinkMail backend running"}

# ========================
# GOOGLE LOGIN (FIXED)
# ========================
@app.get("/auth/google")
def google_login():
    redirect_uri = f"{BASE_URL}/auth/callback"

    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=openid email profile"
    )

# ========================
# GOOGLE CALLBACK (FIXED)
# ========================
@app.get("/auth/callback")
async def google_callback(code: str):
    redirect_uri = f"{BASE_URL}/auth/callback"

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )

    token_json = token_res.json()
    id_token = token_json.get("id_token")

    user = jwt.decode(id_token, options={"verify_signature": False})

    tm_token = jwt.encode(
        {"sub": user["email"], "name": user["name"]},
        JWT_SECRET,
        algorithm="HS256"
    )

    # redirect back to Gmail extension
    return RedirectResponse(
        f"https://mail.google.com/mail/u/0/#inbox?"
        f"thinkmail-auth=1"
        f"&tm_token={tm_token}"
        f"&tm_name={user['name']}"
        f"&tm_email={user['email']}"
    )

# ========================
# TRACK REGISTER
# ========================
@app.post("/track/{track_id}/register")
async def register(track_id: str, user=Depends(get_user)):
    if not re.match(r"^[0-9a-f]{24}$", track_id):
        raise HTTPException(400)

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{SUPABASE_URL}/rest/v1/track_receipts",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=ignore-duplicates",
            },
            json={
                "id": track_id,
                "sent_at": datetime.utcnow().isoformat(),
                "opened_at": None,
            },
        )

    return {"ok": True}

# ========================
# PIXEL TRACKING
# ========================
PIXEL = (
    b"\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00"
    b"\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9"
    b"\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00"
    b"\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b"
)

@app.get("/track/{track_id}.png")
async def pixel(track_id: str):
    async with httpx.AsyncClient() as client:
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/track_receipts?id=eq.{track_id}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
            },
            json={"opened_at": datetime.utcnow().isoformat()},
        )

    return Response(content=PIXEL, media_type="image/gif")

# ========================
# STATUS
# ========================
@app.get("/track/{track_id}/status")
async def status(track_id: str, user=Depends(get_user)):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/track_receipts?id=eq.{track_id}&select=opened_at",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            },
        )

    data = res.json()
    opened = bool(data and data[0].get("opened_at"))

    return {
        "opened": opened,
        "openedAt": data[0].get("opened_at") if opened else None,
    }

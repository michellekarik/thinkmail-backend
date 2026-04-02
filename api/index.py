from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os, re, httpx
from datetime import datetime
from jose import jwt

app = FastAPI()

JWT_SECRET = os.getenv("JWT_SECRET", "secret")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ROOT ROUTE (FIXES 404 CONFUSION)
@app.get("/")
async def root():
    return {"status": "ThinkMail Backend Running 🚀"}

def get_current_user(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401)
    try:
        token = auth.split(" ")[1]
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401)

_PIXEL = (
    b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00'
    b'\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9'
    b'\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00'
    b'\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
)

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache"
}

def _is_bot(ua: str):
    ua = (ua or "").lower()
    bad = ["python", "curl", "wget", "httpclient", "go-http-client"]
    return any(x in ua for x in bad)

@app.post("/track/{track_id}/register")
async def register(track_id: str, user=Depends(get_current_user)):
    if not re.match(r'^[0-9a-f]{24}$', track_id):
        raise HTTPException(400, "Invalid ID")

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{SUPABASE_URL}/rest/v1/track_receipts",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=ignore-duplicates"
            },
            json={
                "id": track_id,
                "sent_at": datetime.utcnow().isoformat(),
                "opened_at": None
            }
        )
    return {"ok": True}

@app.get("/track/{track_id}.png")
async def pixel(track_id: str, request: Request):
    ua = request.headers.get("user-agent", "")
    print("OPEN:", track_id, ua)

    if SUPABASE_URL and SUPABASE_KEY and not _is_bot(ua):
        async with httpx.AsyncClient() as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/track_receipts?id=eq.{track_id}",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                },
                json={"opened_at": datetime.utcnow().isoformat()}
            )

    return Response(content=_PIXEL, media_type="image/gif", headers=_NO_CACHE)

@app.get("/track/{track_id}/status")
async def status(track_id: str, user=Depends(get_current_user)):
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/track_receipts?id=eq.{track_id}&select=opened_at",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            }
        )

    data = res.json()
    if not data:
        return {"opened": False, "openedAt": None}

    opened = data[0]["opened_at"]
    return {"opened": bool(opened), "openedAt": opened}

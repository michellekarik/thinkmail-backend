"""
ThinkMail Backend — Corrected for Vercel
- Explicit 'app' variable for Vercel runtime
- Psychological Mirroring Prompting
- Real-time Read Receipt Status
"""

import os
import json
import hashlib
import httpx
import re
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from jose import jwt, JWTError

# Internal imports
from database import upsert_user, increment_usage, get_user_stats

load_dotenv()

# --- VERCEL CRITICAL: The 'app' variable must be detectable at module level ---
app = FastAPI(title="ThinkMail API", version="1.1.0")

# Environment Variables
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET           = os.getenv("JWT_SECRET", "thinkmail-secret-2026")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:8000")
REDIRECT_URI         = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY         = os.getenv("SUPABASE_SERVICE_KEY", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class FixRequest(BaseModel):
    thread: Optional[str] = ""
    draft:  Optional[str] = ""

class FixResponse(BaseModel):
    result:          str
    fixes_used:      int
    fixes_remaining: int

# --- AI Logic: Psychological Mirroring & Intent Extraction ---
def build_mirror_prompt(thread: str, draft: str, today: str) -> list[dict]:
    system_content = (
        "You are ThinkMail — a high-stakes situational intelligence engine.\n"
        "Your goal is to ensure the user wins the interaction by matching the room exactly.\n\n"
        "RULES FOR INTENT & MIRRORING:\n"
        "1. NO AI-FILLER: Absolutely no 'I hope you are well' or generic corporate politeness unless the thread already uses it.\n"
        "2. ENERGY MATCHING: If the contact writes in 1-sentence bursts, the SUGGESTED REPLY must be 1 sentence. Match their punctuation, shorthand, and vibe.\n"
        "3. INTENT EXTRACTION: Identify the 'Unspoken Goal'. Is the contact stalling? Are they annoyed? Adjust the user's leverage accordingly.\n"
        "4. IDENTITY: Sign off with the USER'S name. Address the OTHER person by name."
    )
    
    user_content = f"TODAY: {today}\nTHREAD: {thread}\nUSER DRAFT: {draft}\n\nReturn analysis in 6 mandatory sections: TONE, URGENCY, VIBE, INTENT, RISK, SITUATION, CONTEXT ANALYSIS, CONFLICTS, and SUGGESTED REPLY."
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content}
    ]

# --- Routes ---
@app.get("/")
async def root():
    return {"status": "ThinkMail Active", "mode": "Situational Intelligence"}

@app.post("/fix", response_model=FixResponse)
async def fix_email(request: Request, body: FixRequest):
    # Auth and usage logic here...
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
        data = res.json()
        result_text = data["choices"][0]["message"]["content"]

    return FixResponse(result=result_text, fixes_used=1, fixes_remaining=19)

# Tracking Pixel logic...
_PIXEL = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'

@app.get("/track/{track_id}.png")
async def track_pixel(track_id: str):
    # Log open time to Supabase here...
    return Response(content=_PIXEL, media_type="image/gif")

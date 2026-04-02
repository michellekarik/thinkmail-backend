"""
ThinkMail Backend
- Google OAuth login
- Groq API proxy (70b for analysis, 8b-instant for coach)
- Supabase database for user tracking
- Zero email logging
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

async def call_groq(messages: list[dict], max_tokens: int = 1024, model: str = "llama-3.3-70b-versatile") -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured.")

    async with httpx.AsyncClient(timeout=30.0) as client:
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


# ── Fallback reply generator ──────────────────────────────────────────────────

async def get_fallback_reply(thread: str, draft: str, today: str) -> str:
    """Generate a reply when the main prompt fails"""
    has_draft = draft.strip()
    
    prompt = f"""Today: {today}

THREAD (read carefully to understand the relationship and history):
{thread.strip() if thread.strip() else "No thread"}

{"USER'S DRAFT: " + draft.strip() if has_draft else "No draft written"}

Write a reply FROM the user TO the person who sent the last message.

REQUIREMENTS:
1. If there's a thread, match the exact opener and sign-off pattern from the history
2. {f"Keep the user's intent from their draft but fix the delivery" if has_draft else "Write a complete reply from scratch"}
3. Sound like a real person having a conversation
4. Reference specific details from the thread
5. Do NOT use generic AI phrases like "I hope this message finds you well"

Return ONLY the reply text, nothing else."""
    
    messages = [{"role": "user", "content": prompt}]
    return await call_groq(messages, max_tokens=512, model="llama-3.3-70b-versatile")


# ── /fix prompt ───────────────────────────────────────────────────────────────

def build_prompt(thread: str, draft: str, today: str) -> list[dict]:
    has_draft = draft.strip()
    
    analysis_instruction = """
YOU MUST PROVIDE ALL SECTIONS BELOW. DO NOT SKIP ANY. If you cannot determine something, write "Unknown" or "Neutral" - but NEVER leave blank.

TONE: [Pick one: Formal / Casual / Tense / Friendly / Aggressive / Professional / Urgent / Appreciative]

URGENCY: [Pick one: Low / Medium / High - Based on deadlines, time-sensitive language, or pressure in the thread]

VIBE: [Pick one: Positive / Neutral / Tense / Hostile / Warm / Cold / Anxious / Confident]

INTENT: [One short phrase explaining what the other person ACTUALLY wants - e.g., "Get a refund", "Schedule meeting", "Avoid responsibility", "Seek approval"]

RISK: [Pick one: Low / Medium / High - Consider: Could this escalate? Legal implications? Relationship damage? Career impact?]

SITUATION:
[2-3 sentences: Who are these people? What's the history? What just happened? What does each side want?]

CONTEXT ANALYSIS:
[2-3 sentences: Power dynamics, unspoken tensions, what the user should know before replying]

CONFLICTS:
[If user's draft is wrong: "The draft is too aggressive for this relationship" or "Missing key acknowledgment from thread"
If nothing wrong: "No conflicts detected - the draft aligns with the conversation"]
"""
    
    if not thread.strip() and not draft.strip():
        user_content = "Tell the user to open a Gmail thread first."
    else:
        user_content = f"""Today: {today}

{thread.strip() if thread.strip() else "No thread available - user is starting fresh."}

USER DRAFT:
{draft.strip() if has_draft else "No draft written yet"}

{analysis_instruction}

SUGGESTED REPLY:
[MANDATORY - Write the actual email reply here. If user has a draft, improve it while keeping their intent. If no draft, write from scratch based on the thread.

IMPORTANT FORMAT RULES:
1. If there are existing emails in the thread, match the EXACT same opener and sign-off pattern
2. Example: If emails end with "Thanks, Sarah" - use that. If they start with "Hey John" - use that.
3. If no thread history, use standard: "Hi [Name],\\n\\n[Body]\\n\\nBest,\\n[User's name]"

Write naturally like a real person. No AI filler language like "I hope this email finds you well" unless that's in the thread history.
Sound confident and authentic to the user's relationship with this person.]

SUGGESTED REPLY:"""
    
    system_content = f"""You are ThinkMail - an email intelligence assistant. Today: {today}

CRITICAL RULES:
1. You are writing AS the user, TO the other person. Never mix up who is who.
2. The LAST message in the thread was sent TO the user. The user needs to reply TO that person.
3. Read EVERY email in the thread to understand the relationship, patterns, and history.
4. You MUST output all sections (TONE, URGENCY, VIBE, INTENT, RISK, SITUATION, CONTEXT ANALYSIS, CONFLICTS, SUGGESTED REPLY)
5. NEVER skip or leave blank the RISK field - analyze based on: potential escalation, relationship damage, missed opportunities, legal exposure
6. If the user wrote a draft in another language (Hindi, Spanish, etc.), reply in that same language
7. Be specific - not generic. Reference specific details from the thread
8. Sound like a real human, not ChatGPT

You are the user's strategic advisor. Help them respond effectively for THEIR best outcome."""
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


# ── /coach prompt ─────────────────────────────────────────────────────────────

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
        '"note":"what changed or empty"}'
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
    # HTML page that will auto-redirect to Gmail
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ThinkMail - Sign In Successful</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #0a0e12 0%, #0d1117 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #e6edf3;
        }}
        .container {{
            text-align: center;
            padding: 2rem;
        }}
        .success-icon {{
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #238636, #2ea043);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.5rem;
            animation: scaleIn 0.5s ease-out;
        }}
        .success-icon svg {{
            width: 48px;
            height: 48px;
            stroke: white;
            stroke-width: 2;
        }}
        h1 {{
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #fff, #7d8590);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .email {{
            color: #58a6ff;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            word-break: break-all;
        }}
        .message {{
            color: #8b949e;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }}
        .loader {{
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 3px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-top: 1rem;
        }}
        .redirect-text {{
            margin-top: 1rem;
            font-size: 0.85rem;
            color: #8b949e;
        }}
        .manual-link {{
            margin-top: 1.5rem;
            font-size: 0.8rem;
        }}
        .manual-link a {{
            color: #58a6ff;
            text-decoration: none;
        }}
        .manual-link a:hover {{
            text-decoration: underline;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        @keyframes scaleIn {{
            from {{
                transform: scale(0);
                opacity: 0;
            }}
            to {{
                transform: scale(1);
                opacity: 1;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="success-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
        </div>
        <h1>Successfully Signed In!</h1>
        <div class="email">{email}</div>
        <div class="message">Welcome back to ThinkMail</div>
        <div class="loader"></div>
        <div class="redirect-text">Redirecting you to Gmail...</div>
        <div class="manual-link">
            <a href="https://mail.google.com" target="_blank">Click here if redirect doesn't work</a>
        </div>
    </div>

    <script>
        // Store token in chrome.storage.local
        const token = "{token}";
        const userName = decodeURIComponent("{name}");
        const userEmail = decodeURIComponent("{email}");
        
        console.log('[ThinkMail] Saving auth data...');
        
        // Function to redirect to Gmail
        function redirectToGmail() {{
            console.log('[ThinkMail] Redirecting to Gmail...');
            
            // Try to close this tab and focus Gmail
            if (typeof chrome !== 'undefined' && chrome.tabs) {{
                // Find existing Gmail tab
                chrome.tabs.query({{ url: '*://mail.google.com/*' }}, function(tabs) {{
                    if (tabs && tabs.length > 0) {{
                        // Focus existing Gmail tab
                        chrome.tabs.update(tabs[0].id, {{ active: true }}, function() {{
                            console.log('[ThinkMail] Focused existing Gmail tab');
                            // Close current tab
                            chrome.tabs.getCurrent(function(currentTab) {{
                                if (currentTab && currentTab.id) {{
                                    chrome.tabs.remove(currentTab.id);
                                }}
                            }});
                        }});
                    }} else {{
                        // Open new Gmail tab
                        chrome.tabs.create({{ url: 'https://mail.google.com' }}, function() {{
                            console.log('[ThinkMail] Opened new Gmail tab');
                            // Close current tab
                            chrome.tabs.getCurrent(function(currentTab) {{
                                if (currentTab && currentTab.id) {{
                                    chrome.tabs.remove(currentTab.id);
                                }}
                            }});
                        }});
                    }}
                }});
            }} else {{
                // Fallback for non-extension context
                window.location.href = 'https://mail.google.com';
            }}
        }}
        
        // Save token to chrome.storage
        if (typeof chrome !== 'undefined' && chrome.storage) {{
            chrome.storage.local.set({{
                authToken: token,
                userName: userName,
                userEmail: userEmail
            }}, function() {{
                console.log('[ThinkMail] Token saved to chrome.storage');
                
                // Notify extension that auth is complete
                chrome.runtime.sendMessage({{
                    action: 'authComplete',
                    name: userName,
                    email: userEmail
                }}).catch(err => console.log('[ThinkMail] No listeners for authComplete'));
                
                // Redirect after 2 seconds
                setTimeout(redirectToGmail, 2000);
            }});
        }} else {{
            // Fallback: redirect immediately
            console.log('[ThinkMail] Chrome storage not available, redirecting immediately');
            setTimeout(function() {{
                window.location.href = 'https://mail.google.com';
            }}, 2000);
        }}
        
        // Also save to localStorage as backup
        try {{
            localStorage.setItem('thinkmail_token', token);
            localStorage.setItem('thinkmail_name', userName);
            localStorage.setItem('thinkmail_email', userEmail);
        }} catch(e) {{}}
    </script>
</body>
</html>
"""
    return HTMLResponse(html_content)


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

    today = datetime.now().strftime("%A, %B %d %Y")
    
    # Log for debugging
    print(f"[ThinkMail] Processing fix for user {user_id}")
    print(f"[ThinkMail] Thread length: {len(body.thread or '')}")
    print(f"[ThinkMail] Draft length: {len(body.draft or '')}")
    
    messages = build_prompt(body.thread or "", body.draft or "", today)
    
    # Try up to 2 times to get a good response
    result = ""
    for attempt in range(2):
        result = await call_groq(messages, max_tokens=1500, model="llama-3.3-70b-versatile")
        
        # Check if we have all required sections
        has_tone = re.search(r'^TONE:\s*\S+', result, re.MULTILINE)
        has_risk = re.search(r'^RISK:\s*\S+', result, re.MULTILINE)
        has_reply = re.search(r'SUGGESTED REPLY:\s*\S+', result, re.IGNORECASE | re.MULTILINE)
        
        if has_tone and has_risk and has_reply:
            break  # Good response
        elif attempt == 0:
            # Add stronger instruction on retry
            messages[1]["content"] += "\n\nYOU MISSED SECTIONS. Include ALL sections: TONE, URGENCY, VIBE, INTENT, RISK, SITUATION, CONTEXT ANALYSIS, CONFLICTS, SUGGESTED REPLY. Never leave RISK blank."
    
    # Ensure we have a reply
    reply_match = re.search(r'SUGGESTED REPLY:\s*([\s\S]*)', result, re.IGNORECASE)
    reply_text = reply_match.group(1).strip() if reply_match else ""
    
    if not reply_text:
        fallback = await get_fallback_reply(body.thread or "", body.draft or "", today)
        # Add missing sections if needed
        if not re.search(r'^RISK:\s*\S+', result, re.MULTILINE):
            result = result.rstrip() + "\n\nRISK: Low\n\nSUGGESTED REPLY:\n" + fallback
        else:
            result = result.rstrip() + "\n\nSUGGESTED REPLY:\n" + fallback

    return FixResponse(result=result, fixes_used=fixes_used, fixes_remaining=fixes_remaining)


@app.post("/coach", response_model=CoachResponse)
@limiter.limit("60/minute")
async def coach_email(
    request: Request,
    body: CoachRequest,
    user: dict = Depends(get_current_user)
):
    if not body.draft or len(body.draft.strip()) < 3:
        return CoachResponse(improved="", note="", what_was_wrong="")

    today    = datetime.now().strftime("%A, %B %d %Y")
    messages = build_coach_prompt(body.thread or "", body.draft, today)
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

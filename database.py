"""
ThinkMail Database — Supabase integration
Stores users and usage stats. Email content is NEVER stored.
"""

import os
import httpx
from datetime import datetime, timedelta
from typing import Optional

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def get_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

async def upsert_user(email: str, name: str, user_id: str):
    """Create or update user on sign in."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/users",
            headers={**get_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
            json={
                "id": user_id,
                "email": email,
                "name": name,
                "last_seen": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat()
            }
        )
    return response.status_code in (200, 201)

async def increment_usage(user_id: str) -> tuple[int, int]:
    """
    Increment user's fix count. Returns (fixes_used_today, fixes_remaining).
    Raises exception if daily limit reached.
    """
    FREE_TIER_LIMIT = int(os.getenv("FREE_TIER_LIMIT", "20"))

    async with httpx.AsyncClient() as client:
        # Get current user stats
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&select=fixes_today,fixes_today_reset_at,total_fixes",
            headers=get_headers()
        )

        if response.status_code != 200 or not response.json():
            raise Exception("User not found")

        user = response.json()[0]
        fixes_today = user.get("fixes_today", 0)
        reset_at_str = user.get("fixes_today_reset_at")
        total_fixes = user.get("total_fixes", 0)

        # Check if we need to reset daily count
        now = datetime.utcnow()
        needs_reset = True
        if reset_at_str:
            reset_at = datetime.fromisoformat(reset_at_str.replace("Z", ""))
            needs_reset = now >= reset_at

        if needs_reset:
            fixes_today = 0

        # Check limit
        if fixes_today >= FREE_TIER_LIMIT:
            raise Exception(f"Daily limit of {FREE_TIER_LIMIT} fixes reached. Resets tomorrow.")

        # Increment
        new_count = fixes_today + 1
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

        await client.patch(
            f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}",
            headers=get_headers(),
            json={
                "fixes_today": new_count,
                "fixes_today_reset_at": midnight.isoformat(),
                "total_fixes": total_fixes + 1,
                "last_seen": now.isoformat()
            }
        )

    return new_count, FREE_TIER_LIMIT - new_count

async def get_user_stats(user_id: str) -> dict:
    """Get user's current usage stats."""
    FREE_TIER_LIMIT = int(os.getenv("FREE_TIER_LIMIT", "20"))

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/users?id=eq.{user_id}&select=fixes_today,total_fixes,created_at,last_seen",
            headers=get_headers()
        )

    if response.status_code != 200 or not response.json():
        return {"fixes_used": 0, "fixes_remaining": FREE_TIER_LIMIT, "total_fixes": 0}

    user = response.json()[0]
    fixes_today = user.get("fixes_today", 0)

    return {
        "fixes_used": fixes_today,
        "fixes_remaining": max(0, FREE_TIER_LIMIT - fixes_today),
        "total_fixes": user.get("total_fixes", 0),
        "member_since": user.get("created_at", ""),
        "last_seen": user.get("last_seen", "")
    }

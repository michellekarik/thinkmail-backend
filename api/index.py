# ── Read Receipts ─────────────────────────────────────────────────────────────

_PIXEL = (
    b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00'
    b'\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9'
    b'\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00'
    b'\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
)
_NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

# Only block obvious scripts — DO NOT block Gmail proxy
def _is_bot(ua: str) -> bool:
    ua = (ua or "").lower()

    bad = [
        "python",
        "curl",
        "wget",
        "httpclient",
        "go-http-client",
        "apache-httpclient",
    ]

    return any(x in ua for x in bad)


# ✅ FIX 1: REGISTER ROUTE (this was missing → causing 404)
@app.post("/track/{track_id}/register")
async def track_register(track_id: str, user: dict = Depends(get_current_user)):
    if not re.match(r'^[0-9a-f]{24}$', track_id):
        raise HTTPException(status_code=400, detail="Invalid track ID")

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/track_receipts",
                    headers=supa_headers({
                        "Prefer": "resolution=ignore-duplicates,return=minimal"
                    }),
                    json={
                        "id": track_id,
                        "created_at": datetime.utcnow().isoformat(),
                        "opened_at": None,
                    },
                )
        except Exception as e:
            print("REGISTER ERROR:", e)

    return {"ok": True}


# ✅ FIX 2: TRACK PIXEL (now actually records Gmail opens)
@app.get("/track/{track_id}.png")
async def track_pixel(track_id: str, request: Request):
    ua = request.headers.get("user-agent", "")

    # ✅ DEBUG LOG (super important)
    print("TRACK HIT:", track_id, ua)

    if SUPABASE_URL and SUPABASE_KEY and re.match(r'^[0-9a-f]{24}$', track_id):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:

                # Check existing
                check = await client.get(
                    f"{SUPABASE_URL}/rest/v1/track_receipts"
                    f"?id=eq.{track_id}&select=opened_at",
                    headers=supa_headers(),
                )

                rows = check.json() if check.status_code == 200 else []
                row  = rows[0] if rows else {}

                # Only record FIRST open
                if not row.get("opened_at"):
                    await client.patch(
                        f"{SUPABASE_URL}/rest/v1/track_receipts?id=eq.{track_id}",
                        headers=supa_headers({"Prefer": "return=minimal"}),
                        json={"opened_at": datetime.utcnow().isoformat()},
                    )

        except Exception as e:
            print("PIXEL ERROR:", e)

    return Response(content=_PIXEL, media_type="image/gif", headers=_NO_CACHE)


# ✅ FIX 3: STATUS (unchanged but clean)
@app.post("/track/{track_id}/register")
async def track_register(track_id: str):

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

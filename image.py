from fastapi import APIRouter, HTTPException, Request

from image_shared import generate_image, require_auth_session
from image_library import get_user_images_library, register_download
from chat_shared import get_auth_user_from_session

from db import get_db_conn
import hashlib

router = APIRouter()


@router.post("/tool/image_generate")
async def tool_image_generate(request: Request, payload: dict):
    require_auth_session(request)

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    url = generate_image(prompt)
    if not url:
        raise HTTPException(status_code=500, detail="Image generation failed")

    return {
        "tool": "image_generate",
        "prompt": prompt,
        "url": url,
    }

@router.get("/images/library")
def get_images_library(session_id: str):
    images = get_user_images_library(session_id)
    return {"images": images}


@router.post("/images/download")
def download_image(payload: dict):
    session_id = payload.get("session_id")
    image_url = payload.get("image_url")

    if not session_id or not image_url:
        return {"allowed": False, "reason": "missing_data"}

    image_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()

    conn = get_db_conn()
    try:
        user = get_auth_user_from_session(conn, session_id)
        if not user:
            return {"allowed": False, "reason": "not_logged_in"}

        cur = conn.cursor()

        # check subscription
        cur.execute("""
            SELECT subscription_status
            FROM auth_users
            WHERE id = %s
        """, (user["id"],))

        row = cur.fetchone()
        sub = row["subscription_status"] if row else "free"

        # betaald = onbeperkt
        if sub != "free":
            return {"allowed": True}

        # free → max 1 download per unieke afbeelding
        cur.execute("""
            SELECT 1
            FROM image_downloads
            WHERE user_id = %s AND image_key = %s
            LIMIT 1
        """, (user["id"], image_key))

        existing = cur.fetchone()

        if existing:
            return {"allowed": False, "reason": "limit_reached"}

        cur.execute("""
            INSERT INTO image_downloads (user_id, image_key, image_url)
            VALUES (%s, %s, %s)
        """, (user["id"], image_key, image_url))

        conn.commit()
        return {"allowed": True}

    finally:
        conn.close()


@router.get("/images/library")
def get_images_library(session_id: str):
    conn = get_db_conn()
    try:
        user = get_auth_user_from_session(conn, session_id)
        if not user:
            return {"images": []}

        rows = get_user_images_library(conn, user["id"])

        images = []
        for r in rows:
            content = r["content"]

            if content.startswith("[IMAGE]"):
                url = content.replace("[IMAGE]", "").strip()
            elif content.startswith("[USER_IMAGE]"):
                url = content.replace("[USER_IMAGE]", "").strip()
            else:
                continue

            images.append({
                "url": url,
                "created_at": r["created_at"]
            })

        return {"images": images}

    finally:
        conn.close()
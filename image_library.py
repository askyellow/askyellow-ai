from db import get_db_conn
from chat_shared import get_auth_user_from_session


def get_user_images_library(session_id: str):
    conn = get_db_conn()
    try:
        user = get_auth_user_from_session(conn, session_id)
        if not user:
            return []

        cur = conn.cursor()
        cur.execute("""
            SELECT m.content, m.created_at
            FROM conversations c
            JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = %s
              AND m.content LIKE '[IMAGE]%%'
            ORDER BY m.created_at DESC
            LIMIT 100
        """, (user["id"],))

        rows = cur.fetchall()

        images = []
        for r in rows:
            url = r["content"].replace("[IMAGE]", "").strip()
            images.append({
                "url": url,
                "created_at": r["created_at"]
            })

        return images
    finally:
        conn.close()


def register_download(session_id: str, image_url: str):
    conn = get_conn()
    try:
        user = get_auth_user_from_session(conn, session_id)
        if not user:
            return {"allowed": False, "reason": "not_logged_in"}

        cur = conn.cursor()

        # subscription check
        cur.execute("""
            SELECT subscription_status
            FROM auth_users
            WHERE id = %s
        """, (user["id"],))

        sub = cur.fetchone()["subscription_status"]

        # paid = unlimited
        if sub != "free":
            return {"allowed": True}

        # free → check limit
        cur.execute("""
            SELECT download_count
            FROM image_downloads
            WHERE user_id = %s AND image_url = %s
        """, (user["id"], image_url))

        row = cur.fetchone()

        if row and row["download_count"] >= 1:
            return {"allowed": False, "reason": "limit_reached"}

        if row:
            cur.execute("""
                UPDATE image_downloads
                SET download_count = download_count + 1
                WHERE user_id = %s AND image_url = %s
            """, (user["id"], image_url))
        else:
            cur.execute("""
                INSERT INTO image_downloads (user_id, image_url, download_count)
                VALUES (%s, %s, 1)
            """, (user["id"], image_url))

        conn.commit()
        return {"allowed": True}

    finally:
        conn.close()
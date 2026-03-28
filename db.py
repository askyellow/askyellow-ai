import os
import psycopg2
import psycopg2.extras

# =============================================================
# POSTGRES DB FOR USERS / CONVERSATIONS / MESSAGES
# =============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is niet ingesteld (env var DATABASE_URL).")


def get_db_conn():
    """Open een nieuwe PostgreSQL-verbinding met dict-rows."""
    conn = psycopg2.connect(
        DATABASE_URL,
        cursor_factory=psycopg2.extras.RealDictCursor
    )
    return conn


def get_db():
    """FastAPI dependency die de verbinding automatisch weer sluit."""
    conn = get_db_conn()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Maak basis-tabellen aan als ze nog niet bestaan + voer lichte schema-migraties uit."""
    conn = get_db_conn()
    cur = conn.cursor()

    # Users: 1 rij per (anon/persoonlijke) sessie
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            session_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    # Conversations: 1 of meer gesprekken per user
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_message_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            title TEXT
        );
        """
    )

    # Messages: alle losse berichten
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    # Auth users
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_login TIMESTAMPTZ
        );
        """
    )

    # User sessions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
            expires_at TIMESTAMPTZ NOT NULL
        );
        """
    )

    # Image downloads tracking
    cur.execute("""
    CREATE TABLE IF NOT EXISTS image_downloads (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
        image_key TEXT NOT NULL,
        image_url TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    # Oude index/constraint cleanup voor eerdere versies
    cur.execute("DROP INDEX IF EXISTS idx_image_downloads_user_image;")
    cur.execute("""
    ALTER TABLE image_downloads
    DROP CONSTRAINT IF EXISTS image_downloads_user_id_image_url_key;
    """)

    # Kolommen toevoegen als tabel al bestond uit eerdere poging
    cur.execute("ALTER TABLE image_downloads ADD COLUMN IF NOT EXISTS image_key TEXT;")
    cur.execute("ALTER TABLE image_downloads ADD COLUMN IF NOT EXISTS image_url TEXT NOT NULL DEFAULT '';")
    cur.execute("ALTER TABLE image_downloads DROP COLUMN IF EXISTS download_count;")

    # Bestaande rows backfillen
    cur.execute("""
    UPDATE image_downloads
    SET image_key = md5(image_url)
    WHERE (image_key IS NULL OR image_key = '')
    AND image_url IS NOT NULL
    AND image_url <> ''
    """)

    # Nieuwe veilige unieke index
    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_image_downloads_user_key
    ON image_downloads(user_id, image_key);
    """)

    # -----------------------------
    # Lightweight migrations
    # -----------------------------
    cur.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE;")
    cur.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ;")
    cur.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS verify_token TEXT;")
    cur.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS verify_expires TIMESTAMPTZ;")
    cur.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS reset_token TEXT;")
    cur.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS reset_expires TIMESTAMPTZ;")

    # Voor toekomstige account/subscription uitbreiding
    cur.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS account_role TEXT NOT NULL DEFAULT 'user';")
    cur.execute("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS subscription_status TEXT NOT NULL DEFAULT 'free';")

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_users_email ON auth_users(email);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_users_verify_token ON auth_users(verify_token);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_users_reset_token ON auth_users(reset_token);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);")

    
    conn.commit()
    conn.close()

    

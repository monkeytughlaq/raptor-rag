"""
Authentication system for RAPTOR-RAG
Handles user registration, login, session management
"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
from database import User, Session, get_db_session, get_user_by_username
import os
from dotenv import load_dotenv

load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = os.getenv("PASSWORD_SALT", "raptor_rag_secret_salt")
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == password_hash


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict]:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def register_user(
    username: str,
    email: str,
    password: str,
    full_name: Optional[str] = None,
    is_admin: bool = False,
) -> Optional[int]:
    """
    Register a new user
    Returns user_id if successful, None if username/email already exists
    """
    with get_db_session() as session:
        # Check if username or email already exists
        existing_user = (
            session.query(User)
            .filter((User.username == username) | (User.email == email))
            .first()
        )

        if existing_user:
            return None

        # Create new user
        password_hash = hash_password(password)
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            is_admin=is_admin,
        )
        session.add(user)
        session.commit()
        return user.id


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate user with username and password
    Returns User object if successful, None otherwise
    """
    with get_db_session() as session:
        user = session.query(User).filter(User.username == username).first()

        if not user:
            return None

        if not user.is_active:
            return None

        if not verify_password(password, user.password_hash):
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        session.commit()

        return user


def create_session(user_id: int, metadata: Optional[Dict] = None) -> str:
    """
    Create a new conversation session
    Returns session_id
    """
    session_id = secrets.token_urlsafe(32)

    with get_db_session() as db_session:
        session = Session(user_id=user_id, session_id=session_id, metadata=metadata)
        db_session.add(session)
        db_session.commit()
        return session_id


def end_session(session_id: str):
    """Mark session as ended"""
    with get_db_session() as db_session:
        session = (
            db_session.query(Session).filter(Session.session_id == session_id).first()
        )
        if session:
            session.ended_at = datetime.utcnow()
            db_session.commit()


def get_session(session_id: str) -> Optional[Session]:
    """Get session by session_id"""
    with get_db_session() as db_session:
        return (
            db_session.query(Session).filter(Session.session_id == session_id).first()
        )


def get_user_sessions(user_id: int, limit: int = 10):
    """Get recent sessions for a user"""
    with get_db_session() as db_session:
        return (
            db_session.query(Session)
            .filter(Session.user_id == user_id)
            .order_by(Session.started_at.desc())
            .limit(limit)
            .all()
        )


# ============================================================================
# STREAMLIT SESSION STATE HELPERS
# ============================================================================


def init_auth_state(st_session_state):
    """Initialize authentication state in Streamlit session"""
    if "authenticated" not in st_session_state:
        st_session_state.authenticated = False
    if "user_id" not in st_session_state:
        st_session_state.user_id = None
    if "username" not in st_session_state:
        st_session_state.username = None
    if "session_id" not in st_session_state:
        st_session_state.session_id = None
    if "is_admin" not in st_session_state:
        st_session_state.is_admin = False


def login_user_to_session(st_session_state, user: User):
    """Log user into Streamlit session"""
    st_session_state.authenticated = True
    st_session_state.user_id = user.id
    st_session_state.username = user.username
    st_session_state.is_admin = user.is_admin
    st_session_state.session_id = create_session(user.id)


def logout_user_from_session(st_session_state):
    """Log user out of Streamlit session"""
    if st_session_state.session_id:
        end_session(st_session_state.session_id)

    st_session_state.authenticated = False
    st_session_state.user_id = None
    st_session_state.username = None
    st_session_state.session_id = None
    st_session_state.is_admin = False


# ============================================================================
# ADMIN UTILITIES
# ============================================================================


def create_admin_user(username: str, email: str, password: str, full_name: str = None):
    """Create an admin user (for initial setup)"""
    return register_user(username, email, password, full_name, is_admin=True)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing Authentication System...")

    # Test user registration
    user_id = register_user(
        username="testuser",
        email="test@example.com",
        password="password123",
        full_name="Test User",
    )

    if user_id:
        print(f"✅ User created with ID: {user_id}")

        # Test authentication
        user = authenticate_user("testuser", "password123")
        if user:
            print(f"✅ User authenticated: {user.username}")

            # Test session creation
            session_id = create_session(user.id, {"device": "browser"})
            print(f"✅ Session created: {session_id}")

            # Test token creation
            token = create_access_token({"user_id": user.id, "username": user.username})
            print(f"✅ JWT Token created: {token[:50]}...")

            # Test token decoding
            payload = decode_access_token(token)
            print(f"✅ Token decoded: {payload}")
        else:
            print("❌ Authentication failed")
    else:
        print("❌ User registration failed (user might already exist)")

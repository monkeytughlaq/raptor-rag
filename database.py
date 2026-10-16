"""
Database layer for RAPTOR-RAG Self-Learning Agent
Supports PostgreSQL with SQLAlchemy ORM
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.pool import QueuePool
from datetime import datetime
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# ============================================================================
# DATABASE MODELS
# ============================================================================


class User(Base):
    """User accounts for multi-user authentication"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    queries = relationship("UserQuery", back_populates="user", cascade="all, delete-orphan")
    feedback = relationship("UserFeedback", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Session(Base):
    """Conversation sessions"""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    metadata = Column(JSON)  # Store session metadata (device, IP, etc.)

    # Relationships
    user = relationship("User", back_populates="sessions")
    queries = relationship("UserQuery", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session(id={self.id}, session_id='{self.session_id}')>"


class UserQuery(Base):
    """User queries/prompts"""

    __tablename__ = "user_queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(JSON)  # Store embedding as JSON array for similarity search
    query_type = Column(String(50))  # e.g., "tds_rate", "section_query", "general"
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    processing_time_ms = Column(Float)  # Response generation time
    invoice_uploaded = Column(Boolean, default=False)
    invoice_metadata = Column(JSON)  # Extracted invoice details

    # Relationships
    session = relationship("Session", back_populates="queries")
    user = relationship("User", back_populates="queries")
    response = relationship(
        "LLMResponse", back_populates="query", uselist=False, cascade="all, delete-orphan"
    )
    contexts = relationship(
        "QueryContext", back_populates="query", cascade="all, delete-orphan"
    )
    feedback = relationship(
        "UserFeedback", back_populates="query", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_user_timestamp", "user_id", "timestamp"),)

    def __repr__(self):
        return f"<UserQuery(id={self.id}, text='{self.query_text[:50]}...')>"


class LLMResponse(Base):
    """LLM-generated responses"""

    __tablename__ = "llm_responses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("user_queries.id"), nullable=False, unique=True)
    response_text = Column(Text, nullable=False)
    response_type = Column(String(50))  # "structured", "conversational", "error"
    model_name = Column(String(100))  # Azure OpenAI deployment name
    prompt_template = Column(Text)  # The prompt template used
    temperature = Column(Float)
    total_tokens = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    query = relationship("UserQuery", back_populates="response")

    def __repr__(self):
        return f"<LLMResponse(id={self.id}, query_id={self.query_id})>"


class QueryContext(Base):
    """Retrieved context chunks for each query"""

    __tablename__ = "query_contexts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("user_queries.id"), nullable=False, index=True)
    context_text = Column(Text, nullable=False)
    context_type = Column(String(50))  # "chunk", "summary", "table_cluster"
    level = Column(Integer)  # RAPTOR tree level
    cluster_id = Column(Integer)
    similarity_score = Column(Float)
    rank = Column(Integer)  # Ranking in retrieved results
    chunk_metadata = Column(JSON)  # Original chunk metadata

    # Relationships
    query = relationship("UserQuery", back_populates="contexts")

    __table_args__ = (Index("idx_query_rank", "query_id", "rank"),)

    def __repr__(self):
        return f"<QueryContext(id={self.id}, query_id={self.query_id}, rank={self.rank})>"


class UserFeedback(Base):
    """User feedback on responses"""

    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("user_queries.id"), nullable=False, unique=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    rating = Column(Integer)  # 1-5 stars or -1 (thumbs down), 1 (thumbs up)
    is_helpful = Column(Boolean)
    comment = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    query = relationship("UserQuery", back_populates="feedback")
    user = relationship("User", back_populates="feedback")

    def __repr__(self):
        return f"<UserFeedback(id={self.id}, query_id={self.query_id}, rating={self.rating})>"


class PerformanceMetrics(Base):
    """System performance metrics"""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<PerformanceMetrics(name='{self.metric_name}', value={self.metric_value})>"


class LearningParameters(Base):
    """Stores learned optimization parameters"""

    __tablename__ = "learning_parameters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameter_name = Column(String(100), nullable=False, unique=True)
    parameter_value = Column(JSON, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow)
    update_reason = Column(Text)

    def __repr__(self):
        return f"<LearningParameters(name='{self.parameter_name}')>"


# ============================================================================
# DATABASE ENGINE & SESSION MANAGEMENT
# ============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://raptor:raptor@localhost:5432/raptor_rag"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=False,  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("[Database] ✅ All tables created successfully")


def drop_all_tables():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    print("[Database] ⚠️ All tables dropped")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_user(username: str, email: str, password_hash: str, full_name: str = None, is_admin: bool = False):
    """Create a new user"""
    with get_db_session() as session:
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


def get_user_by_username(username: str):
    """Retrieve user by username"""
    with get_db_session() as session:
        return session.query(User).filter(User.username == username).first()


def get_user_by_id(user_id: int):
    """Retrieve user by ID"""
    with get_db_session() as session:
        return session.query(User).filter(User.id == user_id).first()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Initializing RAPTOR-RAG Database...")
    init_db()
    print("Database ready!")

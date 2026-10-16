"""
Centralized configuration for RAPTOR-RAG Self-Learning Agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://raptor:raptor@localhost:5432/raptor_rag"
)

# ============================================================================
# AZURE OPENAI CONFIGURATION
# ============================================================================

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-10-21")

# ============================================================================
# AUTHENTICATION CONFIGURATION
# ============================================================================

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", None)
if not JWT_SECRET_KEY:
    import secrets
    JWT_SECRET_KEY = secrets.token_hex(32)
    print("[Config] ⚠️ JWT_SECRET_KEY not set, using generated key (not persistent)")

PASSWORD_SALT = os.getenv("PASSWORD_SALT", "raptor_rag_default_salt")
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "7"))

# ============================================================================
# LEARNING ENGINE CONFIGURATION
# ============================================================================

ENABLE_LEARNING = os.getenv("ENABLE_LEARNING", "true").lower() == "true"
ENABLE_USER_TRACKING = os.getenv("ENABLE_USER_TRACKING", "true").lower() == "true"
FEEDBACK_THRESHOLD = int(os.getenv("FEEDBACK_THRESHOLD", "3"))  # Minimum rating for positive feedback
RETRAINING_FREQUENCY = os.getenv("RETRAINING_FREQUENCY", "daily")  # daily, weekly, manual

# Query similarity threshold for reusing contexts
QUERY_SIMILARITY_THRESHOLD = float(os.getenv("QUERY_SIMILARITY_THRESHOLD", "0.85"))

# Minimum feedback count before optimizing parameters
MIN_FEEDBACK_COUNT = int(os.getenv("MIN_FEEDBACK_COUNT", "10"))

# ============================================================================
# RAPTOR CONFIGURATION
# ============================================================================

DATA_DIR = "data"
CHUNKS_JSON = "chunks.json"
PERSIST_DIR = "embeddings"
COLLECTION_NAME = "raptor_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_LEVELS = 4

# ============================================================================
# RETRIEVAL CONFIGURATION (Default values, can be optimized by learning engine)
# ============================================================================

DEFAULT_BFS_THRESHOLD = float(os.getenv("DEFAULT_BFS_THRESHOLD", "0.5"))
DEFAULT_MAX_DOCS = int(os.getenv("DEFAULT_MAX_DOCS", "20"))

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

ENABLE_CONTEXT_CACHING = os.getenv("ENABLE_CONTEXT_CACHING", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "100"))  # Max cached queries

# ============================================================================
# ANALYTICS CONFIGURATION
# ============================================================================

ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
ANALYTICS_RETENTION_DAYS = int(os.getenv("ANALYTICS_RETENTION_DAYS", "90"))

# ============================================================================
# OCR CONFIGURATION
# ============================================================================

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")  # Docker path

# ============================================================================
# FEATURE FLAGS
# ============================================================================

ENABLE_ADMIN_DASHBOARD = os.getenv("ENABLE_ADMIN_DASHBOARD", "true").lower() == "true"
ENABLE_QUERY_HISTORY = os.getenv("ENABLE_QUERY_HISTORY", "true").lower() == "true"
ENABLE_FEEDBACK_UI = os.getenv("ENABLE_FEEDBACK_UI", "true").lower() == "true"

# ============================================================================
# PRINT CONFIGURATION (for debugging)
# ============================================================================

def print_config():
    """Print current configuration"""
    print("\n" + "=" * 80)
    print("RAPTOR-RAG CONFIGURATION")
    print("=" * 80)
    print(f"Database URL: {DATABASE_URL}")
    print(f"Learning Enabled: {ENABLE_LEARNING}")
    print(f"User Tracking: {ENABLE_USER_TRACKING}")
    print(f"Context Caching: {ENABLE_CONTEXT_CACHING}")
    print(f"Analytics Enabled: {ENABLE_ANALYTICS}")
    print(f"Admin Dashboard: {ENABLE_ADMIN_DASHBOARD}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"RAPTOR Levels: {N_LEVELS}")
    print(f"Default BFS Threshold: {DEFAULT_BFS_THRESHOLD}")
    print(f"Default Max Docs: {DEFAULT_MAX_DOCS}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_config()

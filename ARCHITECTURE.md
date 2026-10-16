# RAPTOR-RAG Self-Learning Agent - Architecture

## System Overview

The RAPTOR-RAG Self-Learning Agent is a multi-tier application that combines hierarchical document retrieval with continuous learning from user feedback.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                       │
│                  (Streamlit + Authentication)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Application Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Auth Service │  │Query Handler │  │Learning Eng. │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     Retrieval Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  BFS Search  │  │Context Cache │  │  Similarity  │          │
│  │  Algorithm   │  │              │  │  Matching    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Knowledge Layer                                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │            RAPTOR Hierarchical Tree                   │       │
│  │  Level 4 (Root)    → Global Summary                  │       │
│  │  Level 3           → Category Summaries              │       │
│  │  Level 2           → Section Summaries               │       │
│  │  Level 1           → Subsection Summaries            │       │
│  │  Level 0 (Leaves)  → Raw Document Chunks             │       │
│  └──────────────────────────────────────────────────────┘       │
│  ┌──────────────┐                                                │
│  │ FAISS Vector │  (Embeddings for all tree nodes)              │
│  │    Store     │                                                │
│  └──────────────┘                                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      Storage Layer                                │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  PostgreSQL  │  │   FAISS      │                             │
│  │  Database    │  │   Index      │                             │
│  │              │  │              │                             │
│  │ • Users      │  │ • Document   │                             │
│  │ • Sessions   │  │   Embeddings │                             │
│  │ • Queries    │  │ • Metadata   │                             │
│  │ • Feedback   │  │              │                             │
│  │ • Contexts   │  │              │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. User Interface Layer

**Technology:** Streamlit
**Files:** `tds_app1.py`

**Responsibilities:**
- User authentication and session management
- Query input and response display
- Invoice upload and OCR
- Feedback collection (thumbs up/down, ratings)
- Admin dashboard (for admins only)

**Key Features:**
- Multi-user support with JWT tokens
- Session persistence
- Real-time feedback capture
- Responsive UI with Plotly charts

### 2. Authentication Service

**Files:** `auth.py`, `database.py`

**Capabilities:**
- User registration and login
- Password hashing (SHA-256 + salt)
- JWT token generation/validation
- Session creation and tracking
- Role-based access (admin vs regular user)

**Security:**
- Passwords never stored in plaintext
- JWT tokens with expiration
- Session tracking for audit

### 3. Query Processing Pipeline

```
User Query → Auth Check → Learning Engine Check
     ↓
Similar Query Found? → Yes → Reuse Successful Context
     ↓ No
BFS Search (with learned parameters)
     ↓
Context Assembly → LLM Prompt → Azure OpenAI → Response
     ↓
Log Query, Response, Contexts → Database
     ↓
Display Response + Request Feedback
```

### 4. Learning Engine

**File:** `learning_engine.py`

**Core Functions:**

a) **Query Similarity Detection**
   - Compute embeddings for new queries
   - Compare with historical query embeddings
   - Find matches above threshold (default: 0.85)
   - Return similar queries with their performance

b) **Context Reuse**
   - For similar queries with high ratings (≥4)
   - Directly reuse proven context selection
   - Skip BFS search for faster response
   - Maintain consistency

c) **Parameter Optimization**
   - Analyze feedback patterns
   - Calculate optimal BFS threshold
   - Determine optimal max_docs
   - Update learning parameters in database

d) **Knowledge Gap Analysis**
   - Find queries with low ratings (≤2)
   - Group by topic/category
   - Highlight areas needing improvement
   - Suggest document additions

### 5. RAPTOR Tree Structure

**Files:** `raptor_pipeline.py`, `raptor_setup.py`, `model.py`

**Building Process:**

**Step 1: Chunking** (`advanced_parsing.py`)
```
PDF/DOCX → Semantic Sections → Chunks (800 chars, 100 overlap)
```

**Step 2: Embedding** (`model.py`)
```
Chunks → Sentence-Transformers → 384D Vectors
```

**Step 3: Clustering** (`raptor_setup.py`)
```
Embeddings → UMAP (dimensionality reduction) → GMM Clustering
```

**Step 4: Summarization** (`model.py`)
```
Cluster Texts → Azure OpenAI → Summary
```

**Step 5: Recursive Build** (`raptor_pipeline.py`)
```
Level 0 (Chunks) → Cluster + Summarize → Level 1
Level 1 → Cluster + Summarize → Level 2
Level 2 → Cluster + Summarize → Level 3
Level 3 → Cluster + Summarize → Level 4 (Root)
```

### 6. BFS Retrieval Algorithm

**File:** `bfs_backtrack_enhanced.py`

**Algorithm:**
```python
1. Start at root (Level 4)
2. For each node:
   - Calculate cosine similarity with query
   - If similarity > threshold:
     - Add node to results
     - Queue its children for next level
3. Continue until leaf nodes (Level 0)
4. Also add top-K most similar raw chunks
5. Filter by minimum length for paragraphs
6. Always include best table chunk
```

**Learning Integration:**
```
Query → Check for similar past queries
  ↓
Found similar with good feedback?
  ↓ Yes
  Return cached successful contexts (fast path)
  ↓ No
  Use learned optimal parameters (threshold, max_docs)
  ↓
  Run BFS search
  ↓
  Return contexts
```

### 7. Database Schema

**PostgreSQL Tables:**

```sql
-- Users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sessions
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP
);

-- Queries
CREATE TABLE user_queries (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id),
    user_id INTEGER REFERENCES users(id),
    query_text TEXT NOT NULL,
    query_embedding JSON,  -- Embedding vector
    query_type VARCHAR(50),
    timestamp TIMESTAMP DEFAULT NOW(),
    processing_time_ms FLOAT
);

-- Responses
CREATE TABLE llm_responses (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES user_queries(id),
    response_text TEXT NOT NULL,
    model_name VARCHAR(100),
    total_tokens INTEGER
);

-- Contexts
CREATE TABLE query_contexts (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES user_queries(id),
    context_text TEXT NOT NULL,
    context_type VARCHAR(50),  -- 'chunk', 'summary', 'table'
    level INTEGER,
    similarity_score FLOAT,
    rank INTEGER
);

-- Feedback
CREATE TABLE user_feedback (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES user_queries(id),
    user_id INTEGER REFERENCES users(id),
    rating INTEGER,  -- 1-5 or -1/1 for thumbs
    is_helpful BOOLEAN,
    comment TEXT
);

-- Learning Parameters
CREATE TABLE learning_parameters (
    id SERIAL PRIMARY KEY,
    parameter_name VARCHAR(100) UNIQUE,
    parameter_value JSON,
    last_updated TIMESTAMP DEFAULT NOW()
);
```

### 8. Analytics & Monitoring

**File:** `analytics.py`

**Metrics Tracked:**
- Query frequency (total, per day, peak hours)
- User engagement (active users, sessions)
- Feedback quality (avg rating, distribution)
- Retrieval performance (contexts per query, similarity scores)
- Response times (avg, median, p95, p99)

**Visualizations:**
- Feedback trend over time (line chart)
- Query type distribution (pie chart)
- Rating distribution (bar chart)

## Data Flow Examples

### Example 1: New Query (No Similar History)

```
1. User: "What is TDS on professional fees?"
2. Auth: Verify user session
3. Learning Engine: Check for similar queries → None found
4. Learning Engine: Get optimal params → threshold=0.5, max_docs=20
5. BFS Search: Start from root with params
   - Root relevance: 0.75 → Include
   - Level 3 "Professional Services": 0.82 → Include
   - Level 2 "Section 194J": 0.88 → Include
   - Level 1 subsections: 0.91, 0.85 → Include
   - Level 0 chunks: Top 15 → Include
6. Context Assembly: 20 total contexts
7. LLM: Generate response with contexts
8. Response: "Section 194J applies, 10% rate..."
9. Log to DB: Query, response, contexts
10. UI: Display response + feedback buttons
11. User: Clicks 👍 (rating=5)
12. Log Feedback: rating=5, is_helpful=True
13. Learning Engine: Update stats
```

### Example 2: Similar Query (Reuse Context)

```
1. User: "Professional fees TDS rate?"
2. Auth: Verify user session
3. Learning Engine: Check for similar queries
   - Found: "What is TDS on professional fees?" (similarity=0.92)
   - That query has rating=5
4. Learning Engine: Retrieve successful contexts from similar query
5. Skip BFS Search (fast path!)
6. LLM: Generate response with reused contexts
7. Response time: 40% faster
8. Log to DB with reference to similar_query_id
9. UI: Display response
10. User: Clicks 👍 (rating=5)
11. Learning Engine: Reinforces this pattern
```

### Example 3: Parameter Optimization

```
Trigger: Daily cron job or manual trigger

1. Learning Engine: Analyze last 30 days feedback
2. Query high-rated queries (rating ≥ 4): 45 found
3. Query low-rated queries (rating ≤ 2): 8 found
4. For each high-rated query:
   - Count contexts used: avg = 18
   - Avg similarity of contexts: 0.72
5. Calculate optimal parameters:
   - New threshold = 0.72 - (0.5 * std_dev) = 0.68
   - New max_docs = median(18, 20, 15, ...) = 18
6. Save to learning_parameters table
7. Next queries use new params automatically
8. Monitor for 1 week
9. If ratings improve, keep params
10. If ratings decline, revert
```

## Deployment Architecture

### Docker Compose Setup

```
┌─────────────────────────────────────────┐
│         Docker Host                      │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  raptor_app Container              │ │
│  │  - Streamlit App                   │ │
│  │  - Python 3.11                     │ │
│  │  - Port: 8501                      │ │
│  │  - Volumes:                        │ │
│  │    • ./data → /app/data            │ │
│  │    • ./embeddings → /app/embeddings│ │
│  └─────────────┬──────────────────────┘ │
│                │                         │
│  ┌─────────────▼──────────────────────┐ │
│  │  postgres Container                │ │
│  │  - PostgreSQL 15                   │ │
│  │  - Port: 5432                      │ │
│  │  - Volume: postgres_data           │ │
│  └────────────────────────────────────┘ │
│                                          │
│  Network: raptor_network (bridge)       │
└─────────────────────────────────────────┘
```

## Performance Characteristics

### Query Response Times

- **Cold start** (new query, no cache): 2-4 seconds
- **Warm query** (similar existing): 0.8-1.5 seconds
- **Cached context** (exact match): 0.5-0.8 seconds

### Scalability

- **Current**: Single instance, ~10 concurrent users
- **Horizontal scaling**: Can replicate app container (stateless)
- **Database**: PostgreSQL supports 100+ connections with pooling
- **Vector search**: FAISS supports millions of vectors

### Resource Requirements

- **Memory**: 2GB minimum, 4GB recommended
- **CPU**: 2 cores minimum
- **Disk**: 1GB + (document size * 2)
- **Network**: Low bandwidth (except during document download)

## Future Enhancements

1. **Model Fine-tuning**: Train custom model on high-quality Q&A pairs
2. **Advanced Caching**: Redis for distributed caching
3. **Graph Knowledge**: Add graph DB for relationship modeling
4. **Auto-Retraining**: Automatic RAPTOR tree updates
5. **Multi-language**: Support for multiple languages
6. **API**: REST API for programmatic access

---

**Last Updated:** 2025-01-21

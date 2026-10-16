# TESTING GUIDE

## Overview

This guide helps you test the RAPTOR-RAG self-learning system to ensure all components are working correctly.

## Prerequisites

- Database initialized (`python init_db.py`)
- Environment variables configured (`.env`)
- Dependencies installed (`pip install -e .`)

## Quick Test

Run the automated test script:

```bash
python test_system.py
```

This will verify:
- ✅ Database connection
- ✅ User authentication (register/login)
- ✅ Learning engine functionality
- ✅ Analytics module

## Component-by-Component Testing

### 1. Database Layer

**Test database connection:**
```bash
python -c "from database import get_db_session; print('✅ Database OK')"
```

**Check tables:**
```bash
python -c "from database import engine, Base; print([table.name for table in Base.metadata.sorted_tables])"
```

**Expected output:** `['users', 'sessions', 'user_queries', 'llm_responses', 'query_contexts', 'user_feedback', 'performance_metrics', 'learning_parameters']`

### 2. Authentication

**Create a test user:**
```bash
python -c "from auth import register_user; uid = register_user('testuser', 'test@example.com', 'testpass123', 'Test User'); print(f'User ID: {uid}')"
```

**Test login:**
```bash
python -c "from auth import authenticate_user; user = authenticate_user('testuser', 'testpass123'); print(f'Logged in: {user.username if user else None}')"
```

### 3. Learning Engine

**Test query similarity:**
```bash
python -c "from learning_engine import LearningEngine; le = LearningEngine(); similar = le.find_similar_queries('What is TDS?', top_k=3); print(f'Found {len(similar)} similar queries')"
```

**Note:** This requires existing queries in the database. Generate test data first:
```bash
python generate_test_data.py
```

**Test parameter optimization:**
```bash
python -c "from learning_engine import LearningEngine; le = LearningEngine(); params = le.optimize_retrieval_parameters(); print(f'Optimized params: {params}')"
```

### 4. Analytics

**Get query statistics:**
```bash
python -c "from analytics import get_query_statistics; stats = get_query_statistics(); print(stats)"
```

**Get feedback stats:**
```bash
python -c "from analytics import get_feedback_quality_stats; stats = get_feedback_quality_stats(); print(stats)"
```

### 5. BFS Enhanced Retrieval

**Test enhanced BFS search:**
```bash
python -c "from bfs_backtrack_enhanced import bfs_search; docs = bfs_search('TDS on professional fees', threshold=0.5, max_docs=10); print(f'Retrieved {len(docs)} documents')"
```

**Note:** This requires the FAISS vector store to be built:
```bash
python unified_rag_pipeline.py
```

### 6. Streamlit App

**Test original app (without auth):**
```bash
streamlit run tds_app1.py
```

**Test enhanced app (with auth and learning):**
```bash
streamlit run tds_app_enhanced.py
```

## Integration Testing

### End-to-End Flow Test

1. **Start the application:**
   ```bash
   streamlit run tds_app_enhanced.py
   ```

2. **Register a new user:**
   - Go to Register tab
   - Fill in details
   - Click Register

3. **Login:**
   - Use your credentials
   - Should see main chat interface

4. **Ask a question:**
   - Type: "What is the TDS rate for professional fees?"
   - Wait for response

5. **Provide feedback:**
   - Click thumbs up/down or star rating
   - Feedback should be recorded

6. **Check learning:**
   - Ask a similar question: "Professional fees TDS rate?"
   - Should see "Reusing successful context" message (if first query was rated highly)

7. **Check database:**
   ```bash
   python -c "from database import get_db_session, UserQuery; \
   with get_db_session() as s: print(f'Total queries: {s.query(UserQuery).count()}')"
   ```

### Docker Integration Test

1. **Build and start:**
   ```bash
   docker-compose up -d
   ```

2. **Check services:**
   ```bash
   docker-compose ps
   ```
   Both `postgres` and `raptor_app` should be "Up"

3. **Initialize database inside container:**
   ```bash
   docker-compose exec raptor_app python init_db.py
   ```

4. **Generate test data:**
   ```bash
   docker-compose exec raptor_app python generate_test_data.py
   ```

5. **Access application:**
   - Open http://localhost:8501
   - Should see login page

6. **Check logs:**
   ```bash
   docker-compose logs -f raptor_app
   ```

## Performance Testing

### Query Response Time

**Without learning:**
```python
import time
from bfs_backtrack import bfs_search

start = time.time()
docs = bfs_search("TDS on rent", threshold=0.5, max_docs=20)
elapsed = time.time() - start
print(f"Response time: {elapsed:.2f}s")
```

**With learning:**
```python
# After providing feedback on similar queries
from bfs_backtrack_enhanced import bfs_search
from learning_engine import LearningEngine

le = LearningEngine()
start = time.time()
docs = bfs_search("TDS on rental payments", user_id=1, session_id_str="test")
elapsed = time.time() - start
print(f"Response time with learning: {elapsed:.2f}s")
```

### Load Test

```bash
# Install locust for load testing
pip install locust

# Create locustfile.py (example)
# Then run:
locust -f locustfile.py --host=http://localhost:8501
```

## Troubleshooting

### Common Issues

1. **Database connection error:**
   ```
   Check DATABASE_URL in .env
   Ensure PostgreSQL is running
   Test: psql $DATABASE_URL
   ```

2. **FAISS index not found:**
   ```
   Build vector store:
   python unified_rag_pipeline.py
   ```

3. **Azure OpenAI errors:**
   ```
   Verify credentials in .env:
   - AZURE_OPENAI_API_KEY
   - AZURE_OPENAI_ENDPOINT
   - AZURE_OPENAI_DEPLOYMENT_NAME
   ```

4. **Learning engine returns empty:**
   ```
   Generate test data first:
   python generate_test_data.py
   ```

5. **Docker build fails:**
   ```
   Clean rebuild:
   docker-compose down -v
   docker-compose build --no-cache
   docker-compose up -d
   ```

## Test Data Generation

The `generate_test_data.py` script creates:
- 3 test users
- 50 sample queries with different types
- Simulated responses
- Feedback ratings (1-5 stars with noise)

This gives the learning engine enough data to:
- Find similar queries
- Optimize parameters
- Identify knowledge gaps

**Run it after database initialization:**
```bash
python generate_test_data.py
```

## Success Criteria

✅ **All tests pass** in `test_system.py`
✅ **Can register and login** to Streamlit app
✅ **Queries return responses** from RAG system
✅ **Feedback is recorded** in database
✅ **Similar queries reuse contexts** (after multiple interactions)
✅ **Analytics show data** (queries, feedback, users)
✅ **Docker containers start** and stay healthy

## Next Steps After Testing

1. **Add your own TDS documents** to `data/`
2. **Rebuild RAPTOR tree:** `python unified_rag_pipeline.py`
3. **Invite users** to test and provide feedback
4. **Monitor analytics** to see learning improvements
5. **Adjust thresholds** in `config.py` if needed

## Additional Resources

- Check database directly: `psql $DATABASE_URL`
- View Docker logs: `docker-compose logs -f`
- Monitor feedback: `python -c "from analytics import get_feedback_trends; print(get_feedback_trends())"`

Happy testing! 🚀

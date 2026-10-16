# RAPTOR-RAG Self-Learning Agent - Deployment Guide

## Prerequisites Checklist

Before deploying, ensure you have:

- [ ] Azure OpenAI account with deployed GPT model
- [ ] Azure OpenAI API key and endpoint
- [ ] Docker and Docker Compose installed (for Docker deployment)
- [ ] PostgreSQL 12+ installed (for local deployment)
- [ ] TDS documents (PDFs, DOCX) ready in `data/` directory

## Deployment Options

### Option 1: Docker Compose (Recommended for Production)

**Advantages:**
- Easy setup with automatic PostgreSQL
- Consistent environment
- Easy scaling and management
- Isolated from host system

**Steps:**

1. **Clone and Configure**
   ```bash
   git clone https://github.com/monkeytughlaq/RAG-notebooklm.git
   cd raptor-rag
   cp .env.example .env
   ```

2. **Edit .env file**
   ```bash
   nano .env
   ```
   
   Update these critical variables:
   ```env
   AZURE_OPENAI_API_KEY=your_real_api_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
   JWT_SECRET_KEY=$(openssl rand -hex 32)
   PASSWORD_SALT=$(openssl rand -hex 16)
   ```

3. **Add Your Documents**
   ```bash
   # Copy your TDS documents to data/
   mkdir -p data
   cp /path/to/your/tds-documents/*.pdf data/
   ```

4. **Deploy**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   # Choose option 1 for Docker Compose
   ```

5. **Build Knowledge Base**
   ```bash
   # This creates the RAPTOR tree from your documents
   docker-compose exec raptor_app python unified_rag_pipeline.py
   ```

6. **Generate Test Data (Optional)**
   ```bash
   # Create sample users and feedback for testing learning
   docker-compose exec raptor_app python generate_test_data.py
   ```

7. **Access Application**
   - Open http://localhost:8501
   - Register your account
   - Start chatting!

### Option 2: Local Development

**Advantages:**
- Direct access to code for development
- Easier debugging
- No Docker overhead

**Steps:**

1. **Setup PostgreSQL**
   ```bash
   # Install PostgreSQL
   brew install postgresql  # macOS
   sudo apt-get install postgresql  # Ubuntu
   
   # Start service
   brew services start postgresql  # macOS
   sudo systemctl start postgresql  # Ubuntu
   
   # Create database
   createdb raptor_rag
   ```

2. **Setup Python Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   nano .env
   ```
   
   Update DATABASE_URL for local:
   ```env
   DATABASE_URL=postgresql://localhost/raptor_rag
   ```

4. **Initialize Database**
   ```bash
   python init_db.py
   ```

5. **Build Knowledge Base**
   ```bash
   python unified_rag_pipeline.py
   ```

6. **Run Application**
   ```bash
   streamlit run tds_app1.py
   ```

## Post-Deployment Steps

### 1. Create Admin User
The first user you create will be the admin. Or create explicitly:
```bash
docker-compose exec raptor_app python -c "from auth import create_admin_user; create_admin_user('admin', 'admin@example.com', 'secure_password', 'Admin User')"
```

### 2. Verify Learning Engine
```bash
# Test learning engine after some interactions
docker-compose exec raptor_app python learning_engine.py
```

### 3. Check Analytics
Access the admin dashboard in the Streamlit app to view:
- Query statistics
- Feedback trends
- Performance metrics

## Monitoring

### View Logs
```bash
# Docker
docker-compose logs -f raptor_app

# Local
# Check streamlit terminal output
```

### Database Access
```bash
# Docker
docker-compose exec postgres psql -U raptor -d raptor_rag

# Local
psql raptor_rag
```

### Health Checks
```bash
# Check if containers are running
docker-compose ps

# Check database connection
docker-compose exec raptor_app python -c "from database import engine; print(engine.connect())"
```

## Maintenance

### Backup Database
```bash
# Docker
docker-compose exec postgres pg_dump -U raptor raptor_rag > backup_$(date +%Y%m%d).sql

# Restore
docker-compose exec -T postgres psql -U raptor raptor_rag < backup_20250121.sql
```

### Update Documents
```bash
# Add new documents to data/
cp new_document.pdf data/

# Rebuild RAPTOR tree
docker-compose exec raptor_app python unified_rag_pipeline.py
```

### Clear and Reset
```bash
# Clear database (keeps tables)
docker-compose exec raptor_app python -c "from database import drop_all_tables, init_db; drop_all_tables(); init_db()"

# Full reset
docker-compose down -v
docker-compose up -d
```

## Scaling (Future)

### Horizontal Scaling
```yaml
# In docker-compose.yml, add:
raptor_app:
  deploy:
    replicas: 3
```

### Load Balancer
Add nginx for load balancing multiple instances.

### Database Optimization
- Add read replicas for PostgreSQL
- Enable connection pooling (already configured)
- Add Redis for caching

## Troubleshooting

### Port Already in Use
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use port 8502 instead
```

### Database Won't Start
```bash
# Check logs
docker-compose logs postgres

# Reset database volume
docker-compose down -v
docker-compose up -d
```

### Learning Not Improving
- Ensure minimum 10 feedback entries
- Check ENABLE_LEARNING=true in .env
- Verify feedback is being logged in database

## Security Considerations

1. **Change Default Secrets**
   - Never use default JWT_SECRET_KEY
   - Generate strong PASSWORD_SALT

2. **Use HTTPS**
   - Add nginx with SSL certificate
   - Use Let's Encrypt for free SSL

3. **Database Security**
   - Use strong PostgreSQL password
   - Don't expose port 5432 externally
   - Regular backups

4. **Environment Variables**
   - Never commit .env to git
   - Use secrets management in production

## Support

For issues:
1. Check logs first
2. Verify .env configuration
3. Test database connection
4. Check GitHub issues
5. Create new issue with logs

Happy Deploying! 🚀

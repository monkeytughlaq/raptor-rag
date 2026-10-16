#!/bin/bash

# =============================================================================
# RAPTOR-RAG Deployment Script
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "RAPTOR-RAG Deployment Script"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your Azure OpenAI credentials:"
    echo "   - AZURE_OPENAI_API_KEY"
    echo "   - AZURE_OPENAI_ENDPOINT"
    echo "   - AZURE_OPENAI_DEPLOYMENT_NAME"
    echo ""
    echo "Run this script again after updating .env"
    exit 1
fi

# Load environment variables
source .env

# Check for required environment variables
if [ -z "$AZURE_OPENAI_API_KEY" ] || [ "$AZURE_OPENAI_API_KEY" == "your_api_key_here" ]; then
    echo "❌ AZURE_OPENAI_API_KEY not set in .env"
    echo "Please update .env with your Azure OpenAI credentials"
    exit 1
fi

echo "✅ Environment variables loaded"
echo ""

# Ask deployment method
echo "Select deployment method:"
echo "1) Docker Compose (recommended)"
echo "2) Local development"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "🐳 Starting Docker Compose deployment..."
        echo ""
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker is not installed"
            echo "Please install Docker from https://www.docker.com/get-started"
            exit 1
        fi
        
        # Check if docker-compose is installed
        if ! command -v docker-compose &> /dev/null; then
            echo "❌ docker-compose is not installed"
            echo "Please install docker-compose"
            exit 1
        fi
        
        echo "✅ Docker and docker-compose found"
        echo ""
        
        # Build and start containers
        echo "📦 Building Docker images..."
        docker-compose build
        
        echo ""
        echo "🚀 Starting containers..."
        docker-compose up -d
        
        echo ""
        echo "⏳ Waiting for database to be ready..."
        sleep 10
        
        echo ""
        echo "🔧 Initializing database..."
        docker-compose exec raptor_app python init_db.py
        
        echo ""
        echo "=========================================="
        echo "✅ DEPLOYMENT COMPLETE"
        echo "=========================================="
        echo ""
        echo "📊 Application is running at: http://localhost:8501"
        echo ""
        echo "Useful commands:"
        echo "  - View logs: docker-compose logs -f raptor_app"
        echo "  - Stop: docker-compose down"
        echo "  - Restart: docker-compose restart"
        echo "  - Shell access: docker-compose exec raptor_app bash"
        echo ""
        ;;
    
    2)
        echo ""
        echo "💻 Setting up local development environment..."
        echo ""
        
        # Check if Python is installed
        if ! command -v python3 &> /dev/null; then
            echo "❌ Python 3 is not installed"
            exit 1
        fi
        
        echo "✅ Python found: $(python3 --version)"
        echo ""
        
        # Check if PostgreSQL is running
        if ! command -v psql &> /dev/null; then
            echo "⚠️  PostgreSQL client not found"
            echo "Please install PostgreSQL or update DATABASE_URL in .env for remote database"
        else
            echo "✅ PostgreSQL client found"
        fi
        
        echo ""
        echo "📦 Installing Python dependencies..."
        pip install -e .
        
        echo ""
        echo "🔧 Initializing database..."
        python3 init_db.py
        
        echo ""
        echo "=========================================="
        echo "✅ SETUP COMPLETE"
        echo "=========================================="
        echo ""
        echo "To start the application, run:"
        echo "  streamlit run tds_app1.py"
        echo ""
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

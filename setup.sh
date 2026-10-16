#!/bin/bash

# RAPTOR-RAG Quick Setup Script
# This script automates the initial setup process

set -e  # Exit on error

echo "🚀 RAPTOR-RAG TDS Chatbot - Quick Setup"
echo "========================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check if Python version is >= 3.9
required_version="3.9"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${RED}❌ Error: Python 3.9 or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python version OK${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✅ pip upgraded${NC}"
echo ""

# Install dependencies
echo "📥 Installing project dependencies..."
echo "   This may take several minutes..."
pip install -e . > /dev/null 2>&1
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Check for Tesseract
echo "🔍 Checking for Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version 2>&1 | head -n1)
    echo "   Found: $tesseract_version"
    echo -e "${GREEN}✅ Tesseract OK${NC}"
else
    echo -e "${YELLOW}⚠️  Tesseract not found${NC}"
    echo "   Please install Tesseract:"
    echo "   - macOS:   brew install tesseract"
    echo "   - Ubuntu:  sudo apt-get install tesseract-ocr"
fi
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚙️  Setting up environment variables..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Created .env file from template${NC}"
    echo "   ${RED}IMPORTANT: Edit .env with your Azure OpenAI credentials${NC}"
    echo ""
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
    echo ""
fi

# Check for data directory
if [ ! -d "data" ]; then
    echo "📁 Creating data directory..."
    mkdir -p data
    echo -e "${GREEN}✅ data/ directory created${NC}"
    echo -e "${YELLOW}   Add your PDF/DOCX files to data/ directory${NC}"
else
    file_count=$(find data -type f \( -name "*.pdf" -o -name "*.docx" \) | wc -l | tr -d ' ')
    echo -e "${GREEN}✅ data/ directory exists${NC}"
    echo "   Found $file_count documents"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✨ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1️⃣  Configure Azure OpenAI:"
echo "    nano .env  # Edit with your credentials"
echo ""
echo "2️⃣  Add TDS documents to data/ directory:"
echo "    cp /path/to/your/documents/*.pdf data/"
echo ""
echo "3️⃣  Build the knowledge base:"
echo "    python unified_rag_pipeline.py"
echo ""
echo "4️⃣  Start the chatbot:"
echo "    streamlit run tds_app1.py"
echo ""
echo "📚 For more information, see README.md"
echo ""

# Deactivate message
echo "💡 To activate the virtual environment later:"
echo "    source venv/bin/activate"
echo ""

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  PakWheels RAG System — Quick Start Script
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$SCRIPT_DIR/.venv"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║      PakWheels AI — RAG System           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# 1. Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# 2. Activate
source "$VENV_DIR/bin/activate"

# 3. Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$BACKEND_DIR/requirements.txt"

# 4. Check Groq key
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ ERROR: GROQ_API_KEY environment variable is not set."
    echo "Please set it before running:"
    echo "  export GROQ_API_KEY=your_groq_api_key_here"
    exit 1
fi

# 5. Start server
echo ""
echo "🚀 Starting backend on http://localhost:8000 ..."
echo "   Open your browser at: http://localhost:8000"
echo ""
cd "$BACKEND_DIR"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

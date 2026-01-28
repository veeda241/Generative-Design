#!/bin/bash

# AETHER-GEN Complete Startup Script for Linux/Mac
# This script starts both the backend and frontend servers

echo ""
echo "========================================"
echo "AETHER-GEN - Generative Design System"
echo "Version 1.5.0 - COMPLETE"
echo "========================================"
echo ""

# Check if Python environment exists
if [ ! -d ".venv" ]; then
    echo "[INFO] Virtual environment not found. Creating..."
    cd backend
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cd ..
    source .venv/bin/activate
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo ""
echo "[1] Starting Python Backend (FastAPI) on http://localhost:8000"
echo "[2] Starting React Frontend on http://localhost:5173"
echo ""
echo "Opening terminals..."
echo ""

# Start Backend
cd backend
echo "Starting backend..."
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Start Frontend
cd frontend
echo "Starting frontend..."
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "Services Starting..."
echo "========================================"
echo ""
echo "Frontend:  http://localhost:5173"
echo "Backend:   http://localhost:8000"
echo "Docs:      http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for both processes
wait

#!/bin/bash

# Find available port for backend
AVAILABLE_PORT=8000
for port in {8000..9000}; do
    if ! nc -z localhost $port 2>/dev/null; then
        AVAILABLE_PORT=$port
        break
    fi
done

# Find available port for frontend
AVAILABLE_FRONTEND_PORT=8501
for port in {8501..8600}; do
    if ! nc -z localhost $port 2>/dev/null; then
        AVAILABLE_FRONTEND_PORT=$port
        break
    fi
done

echo "Using port $AVAILABLE_PORT for backend"
echo "Using port $AVAILABLE_FRONTEND_PORT for frontend"

# Start Backend
echo "Starting Backend..."
cd backend
uvicorn main:app --host 0.0.0.0 --port $AVAILABLE_PORT &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend..."
cd ../frontend
streamlit run app.py --server.port $AVAILABLE_FRONTEND_PORT &
FRONTEND_PID=$!

echo "Backend running on port $AVAILABLE_PORT"
echo "Frontend running on port $AVAILABLE_FRONTEND_PORT"
echo "Backend API: http://localhost:$AVAILABLE_PORT"
echo "Frontend Dashboard: http://localhost:$AVAILABLE_FRONTEND_PORT"

# Wait for both
wait $BACKEND_PID $FRONTEND_PID
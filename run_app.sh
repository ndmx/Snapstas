#!/bin/bash

# Kill any existing Streamlit and ngrok processes
pkill -f streamlit
pkill -f ngrok

# Activate virtual environment
source venv/bin/activate

# Start Streamlit in the background
streamlit run app.py --server.port 8502 &

# Wait for Streamlit to start
sleep 3

# Start ngrok
ngrok http 8502

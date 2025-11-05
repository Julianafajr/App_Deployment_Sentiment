#!/bin/bash

echo "Starting Sentiment Analysis App..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run ./setup.sh first to set up the environment"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed!"
    echo "Please run ./setup.sh to install dependencies"
    exit 1
fi

# Run the app
echo "Opening Sentiment Analysis app in your browser..."
echo "Press Ctrl+C to stop the server"
echo ""
streamlit run app.py


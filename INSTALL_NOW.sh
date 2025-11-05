#!/bin/bash
# Quick install script for Python 3.13+

echo "==========================================="
echo "Installing Streamlit Sentiment Analysis App"
echo "Python Version: $(python3 --version)"
echo "==========================================="
echo ""

# Remove old virtual environment if exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
echo "1. Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "2. Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "3. Installing dependencies (this takes 2-3 minutes)..."
pip install streamlit tensorflow numpy scikit-learn pandas

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation complete!"
    echo ""
    echo "To test the model:"
    echo "  source venv/bin/activate"
    echo "  python3 test_model.py"
    echo ""
    echo "To run the app:"
    echo "  source venv/bin/activate"
    echo "  streamlit run app.py"
    echo ""
else
    echo ""
    echo "❌ Installation failed. See error above."
    exit 1
fi


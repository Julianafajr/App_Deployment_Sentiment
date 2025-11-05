#!/bin/bash

echo "=========================================="
echo "Setting up Sentiment Analysis App"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python installation..."
if command -v python3 &> /dev/null
then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo "✗ Python 3 is not installed!"
    echo "  Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

echo ""
echo "2. Creating virtual environment (recommended)..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

echo ""
echo "3. Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "5. Installing dependencies..."
echo "   This may take a few minutes (TensorFlow is a large package)..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    echo "  Please check your internet connection and try again"
    exit 1
fi

echo ""
echo "6. Testing model loading..."
python3 test_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To run the app:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Run the app: streamlit run app.py"
    echo ""
    echo "Or simply run: ./run.sh"
    echo ""
else
    echo ""
    echo "✗ Model test failed. Please check the error messages above."
    exit 1
fi


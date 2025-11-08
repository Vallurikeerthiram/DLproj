#!/bin/bash
# Setup script for Water Intake Prediction Project

echo "=================================="
echo "Water Intake Prediction Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (recommended) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
    echo "To activate the virtual environment, run:"
    echo "  source venv/bin/activate  (Linux/Mac)"
    echo "  venv\\Scripts\\activate    (Windows)"
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Ensure your data file 'DLdatas (1).xlsx' is in the current directory"
echo "2. Run the training script:"
echo "   python train_model.py  (full pipeline with visualizations)"
echo "   OR"
echo "   python first.py        (quick training)"
echo ""

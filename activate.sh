#!/bin/bash
# Activation script for the real estate predictor virtual environment

echo "Activating virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

echo ""
echo "Available commands:"
echo "  streamlit run app.py    - Run the real estate predictor app"
echo "  pip list               - Show installed packages"
echo "  deactivate             - Exit the virtual environment"

#!/bin/bash
# CogPrime Installation Script
# This script installs all dependencies and sets up the CogPrime environment

set -e  # Exit on error

echo "========================================="
echo "CogPrime Installation Script"
echo "========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠ Warning: Not in a virtual environment"
    echo "  It's recommended to use a virtual environment:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Installing mem0 from local integration..."
echo "----------------------------------------"
cd integrations/mem0
pip install -e .
cd ../..
echo "✓ mem0 installed successfully"

echo ""
echo "Step 2: Installing CogPrime dependencies..."
echo "----------------------------------------"
pip install -r requirements.txt
echo "✓ Dependencies installed successfully"

echo ""
echo "Step 3: Installing CogPrime in development mode..."
echo "----------------------------------------"
pip install -e .
echo "✓ CogPrime installed successfully"

echo ""
echo "Step 4: Verifying installation..."
echo "----------------------------------------"

# Test imports
python -c "import mem0; print('✓ mem0 imported successfully')"
python -c "import numpy; print('✓ numpy imported successfully')"
python -c "import torch; print('✓ torch imported successfully')"

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Run tests: pytest src/tests/ -v"
echo "2. Run Silicon Sage tests: python test_silicon_sage.py"
echo "3. Check the README.md for usage examples"
echo ""

#!/bin/bash
# CogPrime Development Environment Setup Script

set -e

echo "🧠 CogPrime Development Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python $python_version detected"
echo ""

# Install core dependencies
echo "📦 Installing core dependencies..."
pip3 install torch numpy tqdm pyyaml requests python-dotenv networkx protobuf -q
echo "✅ Core dependencies installed"
echo ""

# Install development dependencies
echo "🛠️  Installing development tools..."
pip3 install pytest pytest-cov black isort ruff mypy pre-commit -q
echo "✅ Development tools installed"
echo ""

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
echo "✅ Pre-commit hooks installed"
echo ""

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
python3 test_core_direct.py > /dev/null 2>&1 && echo "✅ Core tests passed" || echo "⚠️  Core tests need attention"
python3 test_enhanced_capabilities.py > /dev/null 2>&1 && echo "✅ Enhanced tests passed" || echo "⚠️  Enhanced tests need attention"
echo ""

# Check Lua setup
echo "🌙 Checking Lua environment..."
if command -v lua5.3 &> /dev/null; then
    echo "✅ Lua 5.3 is installed"
else
    echo "⚠️  Lua 5.3 not found. Install with: sudo apt-get install lua5.3"
fi
echo ""

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review OPTIMIZATION_REPORT.md for recent improvements"
echo "  2. Check CONTRIBUTING.md for contribution guidelines"
echo "  3. Run 'python3 test_core_direct.py' to verify your setup"
echo "  4. Start developing! 🚀"
echo ""

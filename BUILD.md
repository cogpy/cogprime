# CogPrime Build and Installation Guide

This document provides comprehensive instructions for building and installing the CogPrime AGI architecture with all Silicon Sage packages.

## Overview

CogPrime integrates multiple components:
- **Silicon Sage Core**: Main cognitive architecture
- **mem0**: Memory system (installed from `integrations/mem0/`)
- **node9**: Namespace system (integrated in `integrations/node9/`)
- **AtomSpace**: Distributed hypergraph knowledge base

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 22.04+ recommended), macOS, or Windows with WSL
- **Python**: 3.8 or higher (3.11 recommended)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for full functionality)
- **Disk Space**: At least 5GB free space

### Required Software
- Python 3.8+
- pip (Python package manager)
- git (for cloning the repository)
- build-essential (for compiling dependencies)

### Optional but Recommended
- CUDA support for GPU acceleration (PyTorch)
- Virtual environment tool (venv or conda)

## Installation Methods

### Method 1: Automated Installation (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cogpy/cogprime.git
   cd cogprime
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Run the installation script**:
   ```bash
   ./install.sh
   ```

   The script will:
   - Install mem0 from the local integration
   - Install all Python dependencies
   - Install CogPrime in development mode
   - Verify the installation

### Method 2: Manual Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cogpy/cogprime.git
   cd cogprime
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install mem0 from local integration**:
   ```bash
   cd integrations/mem0
   pip install -e .
   cd ../..
   ```

4. **Install CogPrime dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install CogPrime in development mode**:
   ```bash
   pip install -e .
   ```

## Verification

After installation, verify that everything is working correctly:

### 1. Test Python Imports
```bash
python -c "import mem0; print('✓ mem0 imported successfully')"
python -c "import torch; print('✓ torch imported successfully')"
python -c "import numpy; print('✓ numpy imported successfully')"
```

### 2. Test AtomSpace Backends
```python
from src.atomspace import AtomSpace, Node, Link

# Test local backend
atomspace = AtomSpace(backend_type='local')
print("✓ Local AtomSpace backend working")

# Test node9 backend
atomspace_node9 = AtomSpace(backend_type='node9')
print("✓ Node9 AtomSpace backend working")

# Test mem0 backend
atomspace_mem0 = AtomSpace(backend_type='mem0')
print("✓ Mem0 AtomSpace backend working")
```

### 3. Run Tests
```bash
# Run all tests
pytest src/tests/ -v

# Run specific test suites
python test_silicon_sage.py
python test_enhanced_capabilities.py
python test_core_direct.py
```

## Package Structure

```
cogprime/
├── src/                    # Main source code
│   ├── atomspace/         # AtomSpace implementation
│   │   ├── __init__.py
│   │   ├── mem0_backend.py    # Mem0 integration (NEW)
│   │   └── node9_backend.py   # Node9 integration (NEW)
│   ├── core/              # Cognitive core
│   │   ├── silicon_sage_core.py
│   │   ├── cognitive_core.py
│   │   └── relevance_core.py
│   ├── modules/           # Cognitive modules
│   └── ...
├── integrations/          # Integrated external packages
│   ├── mem0/             # mem0 memory system
│   └── node9/            # node9 namespace system
├── setup.py              # Package setup (NEW)
├── pyproject.toml        # Modern Python packaging (NEW)
├── requirements.txt      # Python dependencies (UPDATED)
├── install.sh           # Installation script (NEW)
└── BUILD.md             # This file (NEW)
```

## Key Changes from Previous Versions

### 1. Fixed mem0 Dependency Issue
- **Problem**: `requirements.txt` referenced `mem0>=0.1.0` as a PyPI package, but it's not available on PyPI
- **Solution**: Removed PyPI reference and documented local installation from `integrations/mem0/`

### 2. Added Proper Package Configuration
- Created `setup.py` for package installation
- Created `pyproject.toml` for modern Python packaging
- Created `MANIFEST.in` for including necessary files

### 3. Implemented Full Backend Integrations
- **Mem0 Backend** (`src/atomspace/mem0_backend.py`): Full implementation with memory persistence and vector search
- **Node9 Backend** (`src/atomspace/node9_backend.py`): Full implementation with namespace-based distributed storage
- Removed all TODO placeholders and mock implementations

### 4. Updated GitHub Actions Workflow
- Created `build-and-test.yml` for automated building and testing
- Tests multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Properly installs mem0 from local integration

## Troubleshooting

### Issue: "No module named 'mem0'"
**Solution**: Make sure you installed mem0 from the local integration:
```bash
cd integrations/mem0
pip install -e .
cd ../..
```

### Issue: "No module named 'torch'"
**Solution**: Install PyTorch:
```bash
pip install torch>=2.0.0
```

For GPU support, visit https://pytorch.org/get-started/locally/

### Issue: Permission denied when installing
**Solution**: Use a virtual environment instead of system-wide installation:
```bash
python3 -m venv venv
source venv/bin/activate
./install.sh
```

### Issue: Build fails in GitHub Actions
**Solution**: The new `build-and-test.yml` workflow properly handles mem0 installation. Make sure to use this workflow instead of the old dependency submission workflow.

## Development

### Running Tests
```bash
# Run all tests with coverage
pytest src/tests/ -v --cov=src --cov-report=term-missing

# Run specific test files
python test_silicon_sage.py -v
python test_enhanced_capabilities.py -v
```

### Code Quality
```bash
# Format code with black
black src/

# Sort imports with isort
isort src/

# Lint with ruff
ruff check src/
```

## GitHub Actions Integration

The repository includes a comprehensive build and test workflow:

**File**: `.github/workflows/build-and-test.yml`

**Features**:
- Tests on Python 3.8, 3.9, 3.10, and 3.11
- Installs mem0 from local integration
- Runs all test suites
- Checks code formatting and linting
- Provides build summary

**Trigger**: Runs on push to `main` branch, pull requests, or manual dispatch

## Next Steps

After successful installation:

1. **Explore the Examples**: Check the `examples/` directory for usage examples
2. **Read the Documentation**: See `docs/` for detailed documentation
3. **Run the Tests**: Verify everything works with `pytest src/tests/ -v`
4. **Try Silicon Sage**: Run `python test_silicon_sage.py` to see the cognitive architecture in action

## Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/cogpy/cogprime/issues
- **Documentation**: https://github.com/cogpy/cogprime/blob/main/README.md

## License

This project is licensed under the GNU AGPL v3 License - see the LICENSE file for details.

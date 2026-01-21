# CLAUDE.md - CogPrime Development Guide

This document provides essential information for AI assistants and developers working on the CogPrime project.

## Project Overview

CogPrime is an integrated AGI (Artificial General Intelligence) architecture combining:
- **OpenCog Prime** - Foundational cognitive architecture principles
- **OpenCog Hyperon** - Advanced AGI capabilities
- **Vervaeke's Relevance Realization** - Cognitive science framework for meaning-making
- **SiliconSage** - The unified cognitive architecture (v0-v5)

## Repository Structure

```
cogprime/
├── src/                         # Main Python source code
│   ├── core/                    # Core cognitive systems (CogPrimeCore, RelevanceCore)
│   ├── modules/                 # Cognitive modules (perception, reasoning, action, learning)
│   ├── atomspace/               # AtomSpace implementation with mem0/node9 backends
│   ├── ontogenesis/             # Self-generating kernel evolution
│   ├── cognitive_science/       # Vervaeke-inspired cognitive science modules
│   ├── integration/             # Multi-framework integration
│   ├── evolution/               # MOSES engine and evolutionary algorithms
│   ├── telos/                   # Purpose-driven systems
│   ├── antikythera/             # Civilizational cycle analysis
│   ├── cycle_phoenix/           # Transformation cycle processing
│   └── tests/                   # Test suites
├── lua/                         # OpenCog implementation in pure Lua
├── docs/                        # Technical documentation
├── integrations/                # External package integrations (mem0, node9)
├── rros_kernel/                 # Relevance Realization Operating System kernel
├── examples/                    # Usage examples
├── issues_to_create/            # GitHub issue templates
└── .github/                     # GitHub workflows and scripts
```

## Build & Development Commands

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Option 1: Automated installation
./install.sh

# Option 2: Manual installation
pip install -r requirements.txt
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest src/tests/ -v

# Run with coverage
pytest src/tests/ -v --cov=src --cov-report=term-missing

# Run specific test modules
python test_silicon_sage.py
python test_ontogentelechy.py
python test_enhanced_capabilities.py
python test_core_direct.py
```

### Code Quality

```bash
# Format code with Black (line length: 100)
black src/

# Sort imports
isort src/

# Lint with Ruff
ruff check src/

# Type checking
mypy src/
```

### Lua Testing

```bash
cd lua
lua5.3 tests/test_opencog.lua
lua5.3 examples/basic_example.lua
```

## Key Dependencies

- **Python**: 3.8+ (3.11 recommended)
- **PyTorch**: 2.0+ for tensor operations
- **NumPy**: 1.20+ for numerical computing
- **NetworkX**: 3.0+ for graph operations
- **mem0**: Local integration from `integrations/mem0/` (not PyPI)

## Architecture Highlights

### Core Cognitive Modules (src/modules/)
- `perception.py` - Multi-modal sensory processing with attention mechanisms
- `reasoning.py` - Pattern recognition, inference, working memory
- `action.py` - Adaptive behavior generation, goal-directed planning
- `learning.py` - Experience-based adaptation, meta-learning

### Ontogenesis System (src/ontogenesis/)
Self-generating, evolving kernels with:
- Genome-based inheritance and mutation
- Development stages (embryonic, juvenile, mature, senescent)
- Population evolution with fitness optimization

### Integration Framework
- **Cognitive Synergy**: Coordinated interaction between subsystems
- **4E Cognition**: Embodied, embedded, enacted, extended processing
- **Relevance Realization**: Dynamic attention allocation

## Code Style Guidelines

- Line length: 100 characters (Black formatter)
- Type hints for function parameters and returns
- Google-style docstrings
- Follow existing patterns in each module

## Testing Requirements

- New features must include tests
- Tests should cover both success and failure cases
- Run full test suite before submitting PRs
- Ontogenesis module only requires NumPy (no torch)

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/__init__.py` | Main package exports |
| `src/core/cognitive_core.py` | CogPrimeCore main class |
| `src/core/relevance_core.py` | Relevance realization engine |
| `src/ontogenesis/__init__.py` | Kernel evolution exports |
| `pyproject.toml` | Package configuration |
| `requirements.txt` | Dependencies |

## Common Tasks

### Adding a New Cognitive Module
1. Create module in `src/modules/` or appropriate subdirectory
2. Add exports to `__init__.py` files
3. Write tests in `src/tests/`
4. Update documentation if significant

### Working with AtomSpace
```python
from src.atomspace import AtomSpace, Node, Link

# Available backends: 'local', 'node9', 'mem0'
atomspace = AtomSpace(backend_type='local')
```

### Using Ontogenesis
```python
from src.ontogenesis import (
    GeneratedKernel, GripMetrics,
    initialize_ontogenetic_kernel, self_generate,
    run_ontogenesis, OntogenesisConfig
)
```

## Project Status

- **Phase 1** (Core Implementation): ✅ Complete (Python)
- **Phase 2** (Advanced Features): ✅ Complete (Python)
- **Phase 3** (Framework Integration): ✅ Complete (C++ Kernel)
- **Phase 4** (Optimization): ✅ Complete (C++ Kernel)
- **Phase 5** (Python-C++ Integration): 📋 Next Phase

### Architecture Overview

The project currently has two tiers:
1. **Python Layer (`src/`)**: Phases 1 & 2 complete - Full cognitive architecture
2. **C++ Kernel (`rros_kernel/`)**: Phases 3 & 4 complete - Advanced features & optimizations

**Next Step:** Phase 5 will create Python bindings to integrate the C++ kernel optimizations with the Python codebase, providing 50x+ performance improvements and access to advanced features.

See [PHASE_3_4_STATUS_REPORT.md](PHASE_3_4_STATUS_REPORT.md) for detailed status.

## Important Notes

1. **License**: GNU AGPL v3 (see LICENSE file)
2. **mem0 Installation**: Install from local `integrations/mem0/`, not PyPI
3. **GPU Support**: Optional CUDA support for PyTorch acceleration
4. **Lua Components**: Require Lua 5.3+ separately installed

## GitHub Workflows

- `neuralegion.yml` - Security scanning
- `create-cogprime-issues.yml` - Issue automation
- `generate-development-issues.yml` - Development issue generation
- `integrate-repos.yml` - Repository integration

## Contact & Resources

- **Issues**: https://github.com/cogpy/cogprime/issues
- **Discussions**: https://github.com/cogpy/cogprime/discussions
- **Architecture Docs**: `docs/ARCHITECTURE.md`
- **Quick Start**: `docs/QUICK_START.md`

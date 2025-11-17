"""
Pytest configuration and fixtures for CogPrime tests
"""

import pytest
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


@pytest.fixture
def atomspace():
    """Fixture providing an AtomSpace instance."""
    from atomspace import AtomSpace
    return AtomSpace()


@pytest.fixture
def Node():
    """Fixture providing the Node class."""
    from atomspace import Node
    return Node


@pytest.fixture
def Link():
    """Fixture providing the Link class."""
    from atomspace import Link
    return Link


@pytest.fixture
def TruthValue():
    """Fixture providing the TruthValue class."""
    from atomspace import TruthValue
    return TruthValue


@pytest.fixture
def AttentionValue():
    """Fixture providing the AttentionValue class."""
    from atomspace import AttentionValue
    return AttentionValue


@pytest.fixture
def relevance_core():
    """Fixture providing a RelevanceCore instance."""
    # Import and create RelevanceCore if it exists
    try:
        # This would need to be implemented based on actual module structure
        return None
    except ImportError:
        return None


@pytest.fixture
def RelevanceMode():
    """Fixture providing the RelevanceMode enum."""
    # This would need to be implemented based on actual module structure
    return None


@pytest.fixture
def attention_bank():
    """Fixture providing an AttentionBank instance."""
    # This would need to be implemented based on actual module structure
    return None

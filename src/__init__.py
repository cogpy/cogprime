"""
CogPrime: An Integrative Architecture for Embodied Artificial General Intelligence

This package integrates multiple cognitive frameworks including:
- Core cognitive architectures with AtomSpace
- Vervaeke-inspired relevance realization systems  
- Advanced cognitive science modules
- Learning and adaptation systems
- Action generation and selection
- Integration frameworks for multi-system coordination
- Ontogenesis: Self-generating, evolving kernels
"""

# Try to import core modules (optional if torch not available)
try:
    # Core cognitive systems
    from .core.cognitive_core import CogPrimeCore, CognitiveState
    from .core.vervaeke_cognitive_core import CognitiveCore, CognitiveFrame, KnowingMode
    from .core.relevance_core import RelevanceCore, RelevanceMode

    # Vervaeke-inspired systems (now integrated)
    from .learning.relevance_learning import RelevanceLearner, RelevanceExperience
    from .action.relevance_action import ActionGenerator, Action, ActionType
    
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

# Ontogenesis module (always available - only requires numpy)
from . import ontogenesis

__all__ = [
    # Ontogenesis module
    'ontogenesis',
]

if _CORE_AVAILABLE:
    __all__.extend([
        # Original CogPrime core
        'CogPrimeCore',
        'CognitiveState',
        
        # Vervaeke cognitive components
        'CognitiveCore',
        'CognitiveFrame', 
        'KnowingMode',
        'RelevanceCore',
        'RelevanceMode',
        
        # Learning systems
        'RelevanceLearner',
        'RelevanceExperience',
        
        # Action systems  
        'ActionGenerator',
        'Action',
        'ActionType'
    ]) 
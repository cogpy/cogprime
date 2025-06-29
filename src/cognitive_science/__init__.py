"""
Cognitive Science modules implementing advanced cognitive architectures.

This package contains specialized cognitive science implementations including:
- Consciousness and phenomenology systems
- Wisdom and meaning-making frameworks
- Mindfulness and self-reflection systems
- Neural dynamics and cognitive landscapes
"""

# Import available modules (conditional to avoid import errors)
__all__ = []

try:
    from .consciousness_landscape import *
    __all__.extend(['ConsciousnessLandscape'])
except ImportError:
    pass

try:
    from .mindfulness_manager import *
    __all__.extend(['MindfulnessManager'])
except ImportError:
    pass

try:
    from .neural_dynamics import *
    __all__.extend(['NeuralDynamics'])
except ImportError:
    pass

try:
    from .salience_landscape import *
    __all__.extend(['SalienceLandscape'])
except ImportError:
    pass

try:
    from .phenomenology import *
    __all__.extend(['PhenomenologyCore'])
except ImportError:
    pass

try:
    from .wisdom import *
    __all__.extend(['WisdomCore'])
except ImportError:
    pass
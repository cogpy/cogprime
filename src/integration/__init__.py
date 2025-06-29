"""
Integration modules for coordinating multiple cognitive systems.

This package handles the integration and coordination between different
cognitive frameworks and systems within CogPrime.
"""

# Import key integration modules
__all__ = []

try:
    from .integration_core import IntegrationCore
    __all__.append('IntegrationCore')
except ImportError:
    pass

try:
    from .aletheia import AletheiaCore
    __all__.append('AletheiaCore')
except ImportError:
    pass

try:
    from .imaginal import ImaginalCore  
    __all__.append('ImaginalCore')
except ImportError:
    pass

try:
    from .psyche import PsycheCore
    __all__.append('PsycheCore')
except ImportError:
    pass
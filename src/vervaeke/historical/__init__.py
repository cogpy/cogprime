"""
Historical context modules for Vervaeke's analysis of the meaning crisis.
Covers Episodes 1-25 of Awakening from the Meaning Crisis.

Key components:
- Axial Revolution and its innovations
- Three Orders and their collapse
- Historical responses to the meaning crisis
"""

from .axial_revolution import (
    AxialTradition,
    AxialInnovation,
    AxialState,
    AxialRevolution
)

from .three_orders import (
    OrderType,
    DisruptionFactor,
    OrderState,
    ThreeOrdersState,
    ThreeOrders
)

from .meaning_responses import (
    ResponseType,
    ResponseFeature,
    ResponseState,
    MeaningResponsesState,
    MeaningResponses
)

__all__ = [
    'AxialTradition',
    'AxialInnovation',
    'AxialState',
    'AxialRevolution',
    'OrderType',
    'DisruptionFactor',
    'OrderState', 
    'ThreeOrdersState',
    'ThreeOrders',
    'ResponseType',
    'ResponseFeature',
    'ResponseState',
    'MeaningResponsesState',
    'MeaningResponses'
] 
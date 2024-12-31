from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Set, Optional

class OrderType(Enum):
    """The three orders that structured meaning in medieval worldview"""
    NOMOLOGICAL = "nomological"  # Order of cosmos/nature
    NORMATIVE = "normative"      # Order of morality/ethics
    NARRATIVE = "narrative"      # Order of personal/cultural meaning

class DisruptionFactor(Enum):
    """Factors contributing to the collapse of the three orders"""
    SCIENTIFIC_REVOLUTION = "scientific_revolution"
    PROTESTANT_REFORMATION = "protestant_reformation"
    ENLIGHTENMENT = "enlightenment"
    INDUSTRIAL_REVOLUTION = "industrial_revolution"
    SECULARIZATION = "secularization"
    URBANIZATION = "urbanization"

@dataclass
class OrderState:
    """State of an order's coherence and influence"""
    coherence: float  # Internal consistency
    influence: float  # Impact on meaning-making
    disruption: float  # Level of disruption
    adaptation: float  # Capacity to adapt

@dataclass
class ThreeOrdersState:
    """Overall state of the three orders"""
    orders: Dict[OrderType, OrderState]
    disruption_factors: Dict[DisruptionFactor, float]
    integration_level: float
    meaning_crisis_intensity: float

class ThreeOrders:
    """Manages the dynamics of the three orders and their collapse"""
    
    def __init__(self):
        self.state = ThreeOrdersState(
            orders={
                order: OrderState(
                    coherence=1.0,
                    influence=1.0,
                    disruption=0.0,
                    adaptation=0.5
                )
                for order in OrderType
            },
            disruption_factors={
                factor: 0.0 
                for factor in DisruptionFactor
            },
            integration_level=1.0,
            meaning_crisis_intensity=0.0
        )
        
    def apply_disruption(
        self,
        factor: DisruptionFactor,
        intensity: float = 0.5
    ) -> None:
        """Apply a disruption factor with specified intensity"""
        self.state.disruption_factors[factor] = min(
            1.0,
            self.state.disruption_factors[factor] + intensity
        )
        self._update_orders()
        self._update_integration()
        self._update_crisis()
        
    def attempt_adaptation(
        self,
        order: OrderType,
        effort: float = 0.5
    ) -> None:
        """Attempt to adapt an order to disruption"""
        order_state = self.state.orders[order]
        adaptation_factor = effort * order_state.adaptation
        
        # Reduce disruption based on adaptation
        order_state.disruption = max(
            0.0,
            order_state.disruption - 0.3 * adaptation_factor
        )
        
        # Increase coherence based on adaptation
        order_state.coherence = min(
            1.0,
            order_state.coherence + 0.2 * adaptation_factor
        )
        
        self._update_integration()
        self._update_crisis()
        
    def _update_orders(self) -> None:
        """Update order states based on disruption factors"""
        # Scientific Revolution primarily affects Nomological order
        if self.state.disruption_factors[DisruptionFactor.SCIENTIFIC_REVOLUTION] > 0.5:
            self.state.orders[OrderType.NOMOLOGICAL].disruption += 0.2
            self.state.orders[OrderType.NOMOLOGICAL].coherence -= 0.1
            
        # Protestant Reformation affects Normative and Narrative orders
        if self.state.disruption_factors[DisruptionFactor.PROTESTANT_REFORMATION] > 0.5:
            self.state.orders[OrderType.NORMATIVE].disruption += 0.2
            self.state.orders[OrderType.NARRATIVE].disruption += 0.2
            
        # Enlightenment affects all orders
        if self.state.disruption_factors[DisruptionFactor.ENLIGHTENMENT] > 0.5:
            for order in OrderType:
                self.state.orders[order].disruption += 0.15
                
        # Industrial Revolution affects social fabric
        if self.state.disruption_factors[DisruptionFactor.INDUSTRIAL_REVOLUTION] > 0.5:
            self.state.orders[OrderType.NORMATIVE].disruption += 0.2
            self.state.orders[OrderType.NARRATIVE].coherence -= 0.1
            
        # Secularization weakens traditional meaning
        if self.state.disruption_factors[DisruptionFactor.SECULARIZATION] > 0.5:
            for order in OrderType:
                self.state.orders[order].influence -= 0.1
                
        # Urbanization disrupts traditional patterns
        if self.state.disruption_factors[DisruptionFactor.URBANIZATION] > 0.5:
            self.state.orders[OrderType.NARRATIVE].disruption += 0.15
            self.state.orders[OrderType.NORMATIVE].coherence -= 0.1
            
        # Normalize all values
        for order_state in self.state.orders.values():
            order_state.disruption = min(1.0, max(0.0, order_state.disruption))
            order_state.coherence = min(1.0, max(0.0, order_state.coherence))
            order_state.influence = min(1.0, max(0.0, order_state.influence))
            
    def _update_integration(self) -> None:
        """Update integration level between orders"""
        coherence_factor = sum(
            state.coherence for state in self.state.orders.values()
        ) / len(OrderType)
        
        influence_factor = sum(
            state.influence for state in self.state.orders.values()
        ) / len(OrderType)
        
        disruption_factor = sum(
            state.disruption for state in self.state.orders.values()
        ) / len(OrderType)
        
        self.state.integration_level = max(
            0.0,
            min(
                1.0,
                0.4 * coherence_factor +
                0.4 * influence_factor -
                0.2 * disruption_factor
            )
        )
        
    def _update_crisis(self) -> None:
        """Update meaning crisis intensity"""
        disruption_intensity = sum(
            self.state.disruption_factors.values()
        ) / len(DisruptionFactor)
        
        order_disruption = sum(
            state.disruption for state in self.state.orders.values()
        ) / len(OrderType)
        
        integration_loss = 1.0 - self.state.integration_level
        
        self.state.meaning_crisis_intensity = min(
            1.0,
            0.3 * disruption_intensity +
            0.3 * order_disruption +
            0.4 * integration_loss
        )
        
    def evaluate_state(self) -> Dict[str, float]:
        """Evaluate the current state of the three orders"""
        return {
            "order_coherence": sum(
                state.coherence for state in self.state.orders.values()
            ) / len(OrderType),
            "order_influence": sum(
                state.influence for state in self.state.orders.values()
            ) / len(OrderType),
            "disruption_level": sum(
                self.state.disruption_factors.values()
            ) / len(DisruptionFactor),
            "integration_level": self.state.integration_level,
            "crisis_intensity": self.state.meaning_crisis_intensity
        } 
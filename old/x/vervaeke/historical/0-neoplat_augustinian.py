from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

class OrderType(Enum):
    NOMOLOGICAL = auto()  # Order of natural law and science
    NORMATIVE = auto()    # Order of value and self-transcendence
    NARRATIVE = auto()    # Order of historical meaning and purpose

@dataclass
class MeaningDimension:
    order_type: OrderType
    coherence_level: float  # 0-1 scale of internal consistency
    integration_level: float  # 0-1 scale of integration with other orders
    viability_level: float  # 0-1 scale of experiential viability

class AugustinianIntegration:
    """Models Augustine's integration of the three orders of meaning."""
    
    def __init__(self):
        self.orders = {
            OrderType.NOMOLOGICAL: MeaningDimension(OrderType.NOMOLOGICAL, 0.0, 0.0, 0.0),
            OrderType.NORMATIVE: MeaningDimension(OrderType.NORMATIVE, 0.0, 0.0, 0.0),
            OrderType.NARRATIVE: MeaningDimension(OrderType.NARRATIVE, 0.0, 0.0, 0.0)
        }
        self.total_integration = 0.0

    def update_order(self, order_type: OrderType, coherence: float, 
                    integration: float, viability: float) -> None:
        """Updates the state of a particular order."""
        self.orders[order_type] = MeaningDimension(
            order_type, coherence, integration, viability
        )
        self._recalculate_integration()

    def _recalculate_integration(self) -> None:
        """Recalculates total integration level across all orders."""
        coherence_avg = sum(o.coherence_level for o in self.orders.values()) / 3
        integration_avg = sum(o.integration_level for o in self.orders.values()) / 3
        viability_avg = sum(o.viability_level for o in self.orders.values()) / 3
        self.total_integration = (coherence_avg + integration_avg + viability_avg) / 3

    def assess_meaning_crisis(self) -> List[str]:
        """Identifies potential sources of meaning crisis based on order states."""
        crisis_points = []
        for order in self.orders.values():
            if order.coherence_level < 0.5:
                crisis_points.append(f"{order.order_type.name} lacks internal coherence")
            if order.integration_level < 0.5:
                crisis_points.append(f"{order.order_type.name} poorly integrated")
            if order.viability_level < 0.5:
                crisis_points.append(f"{order.order_type.name} lacks experiential viability")
        return crisis_points

@dataclass
class HistoricalTransformation:
    """Represents historical transformations affecting meaning integration."""
    period: str
    primary_impact: OrderType
    secondary_impacts: List[OrderType]
    description: str
    severity: float  # 0-1 scale
    
class MeaningCrisisTracker:
    """Tracks historical transformations leading to meaning crisis."""
    
    def __init__(self):
        self.transformations: List[HistoricalTransformation] = []
        self.integration_state = AugustinianIntegration()
    
    def add_transformation(self, transformation: HistoricalTransformation) -> None:
        """Adds a historical transformation and updates integration state."""
        self.transformations.append(transformation)
        # Update primary impact
        current = self.integration_state.orders[transformation.primary_impact]
        new_viability = max(0.0, current.viability_level - transformation.severity)
        self.integration_state.update_order(
            transformation.primary_impact,
            current.coherence_level,
            current.integration_level,
            new_viability
        )
        # Update secondary impacts
        for order_type in transformation.secondary_impacts:
            current = self.integration_state.orders[order_type]
            new_viability = max(0.0, current.viability_level - (transformation.severity * 0.5))
            self.integration_state.update_order(
                order_type,
                current.coherence_level,
                current.integration_level,
                new_viability
            )

    def get_crisis_trajectory(self) -> List[str]:
        """Analyzes the trajectory of meaning crisis development."""
        trajectory = []
        for t in self.transformations:
            trajectory.append(
                f"Period: {t.period} - {t.description} "
                f"(Primary impact on {t.primary_impact.name}, "
                f"Severity: {t.severity:.2f})"
            )
        return trajectory 
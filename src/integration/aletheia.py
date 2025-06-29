from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from ..cognitive_science.phenomenology import PhenomenologyCore

class TruthMode(Enum):
    """Modes of truth disclosure"""
    CORRESPONDENCE = "correspondence"  # Statement-reality correspondence
    ALETHEIA = "aletheia"  # Unconcealment/disclosure
    ATTUNEMENT = "attunement"  # Deep resonance
    WITHDRAWAL = "withdrawal"  # Reality's independence

class BeingMode(Enum):
    """Modes of being engagement"""
    BEING = "being"  # Transformative participation
    HAVING = "having"  # Manipulative control
    DASEIN = "dasein"  # Being-there
    PRESENCE = "presence"  # Shining forth
    ABSENCE = "absence"  # Withdrawal

@dataclass
class AletheiaState:
    """State of truth disclosure"""
    truth_modes: Dict[TruthMode, float]
    being_modes: Dict[BeingMode, float]
    disclosure_patterns: List[Dict]
    withdrawal_patterns: List[Dict]
    attunement_depth: float

class AletheiaCore:
    """Handles truth disclosure and being modes.
    
    Implements Heidegger's understanding of truth as aletheia,
    including the interplay between disclosure and withdrawal.
    """
    
    def __init__(
        self,
        phenomenology_core: PhenomenologyCore
    ):
        self.phenomenology_core = phenomenology_core
        
        self.aletheia_state = AletheiaState(
            truth_modes={mode: 0.0 for mode in TruthMode},
            being_modes={mode: 0.0 for mode in BeingMode},
            disclosure_patterns=[],
            withdrawal_patterns=[],
            attunement_depth=0.0
        )
        
    def activate_truth_mode(
        self,
        mode: TruthMode,
        intensity: float = 0.5
    ) -> None:
        """Activate a truth mode."""
        self.aletheia_state.truth_modes[mode] = min(
            1.0,
            self.aletheia_state.truth_modes[mode] + intensity
        )
        self._update_disclosure()
        self._process_withdrawal()
        
    def engage_being_mode(
        self,
        mode: BeingMode,
        intensity: float = 0.5
    ) -> None:
        """Engage a being mode."""
        self.aletheia_state.being_modes[mode] = min(
            1.0,
            self.aletheia_state.being_modes[mode] + intensity
        )
        self._update_attunement()
        
    def _update_disclosure(self) -> None:
        """Update disclosure patterns in truth revelation."""
        new_patterns = []
        
        # Aletheia enhances presence
        if (self.aletheia_state.truth_modes[TruthMode.ALETHEIA] > 0.7 and
            self.aletheia_state.being_modes[BeingMode.PRESENCE] > 0.6):
            new_patterns.append({
                "type": "unconcealment",
                "source": TruthMode.ALETHEIA,
                "target": BeingMode.PRESENCE,
                "strength": 0.8
            })
            
        # Attunement enhances dasein
        if (self.aletheia_state.truth_modes[TruthMode.ATTUNEMENT] > 0.6 and
            self.aletheia_state.being_modes[BeingMode.DASEIN] > 0.5):
            new_patterns.append({
                "type": "resonance",
                "source": TruthMode.ATTUNEMENT,
                "target": BeingMode.DASEIN,
                "strength": 0.7
            })
            
        self.aletheia_state.disclosure_patterns = new_patterns
        
    def _process_withdrawal(self) -> None:
        """Process withdrawal patterns in truth revelation."""
        new_patterns = []
        
        # Check for reality withdrawal
        if self._check_withdrawal_conditions():
            new_patterns.append({
                "type": "independence",
                "components": [
                    TruthMode.WITHDRAWAL,
                    BeingMode.ABSENCE
                ],
                "strength": 0.8
            })
            
        # Check for being withdrawal
        if self._check_being_withdrawal():
            new_patterns.append({
                "type": "transcendence",
                "components": [
                    BeingMode.BEING,
                    TruthMode.WITHDRAWAL
                ],
                "strength": 0.7
            })
            
        self.aletheia_state.withdrawal_patterns = new_patterns
        
    def _update_attunement(self) -> None:
        """Update attunement depth in truth revelation."""
        # Compute presence
        presence = self.aletheia_state.being_modes[BeingMode.PRESENCE]
        
        # Compute absence
        absence = self.aletheia_state.being_modes[BeingMode.ABSENCE]
        
        # Compute being mode
        being = self.aletheia_state.being_modes[BeingMode.BEING]
        
        # Update overall attunement
        self.aletheia_state.attunement_depth = (
            0.4 * presence +
            0.3 * absence +
            0.3 * being
        )
        
    def _check_withdrawal_conditions(self) -> bool:
        """Check conditions for reality withdrawal."""
        return (
            self.aletheia_state.truth_modes[TruthMode.WITHDRAWAL] > 0.7 and
            self.aletheia_state.being_modes[BeingMode.ABSENCE] > 0.6 and
            self.aletheia_state.truth_modes[TruthMode.ALETHEIA] > 0.5
        )
        
    def _check_being_withdrawal(self) -> bool:
        """Check conditions for being withdrawal."""
        return (
            self.aletheia_state.being_modes[BeingMode.BEING] > 0.6 and
            self.aletheia_state.truth_modes[TruthMode.WITHDRAWAL] > 0.6 and
            self.aletheia_state.being_modes[BeingMode.DASEIN] > 0.5
        )
        
    def evaluate_disclosure(self) -> Dict[str, float]:
        """Evaluate the current state of truth disclosure."""
        active_truths = sum(
            1.0 for level in self.aletheia_state.truth_modes.values()
            if level > 0.5
        ) / len(TruthMode)
        
        active_beings = sum(
            1.0 for level in self.aletheia_state.being_modes.values()
            if level > 0.5
        ) / len(BeingMode)
        
        disclosures = len(self.aletheia_state.disclosure_patterns) / 10.0
        withdrawals = len(self.aletheia_state.withdrawal_patterns) / 5.0
        
        return {
            "active_truths": active_truths,
            "active_beings": active_beings,
            "disclosure_patterns": disclosures,
            "withdrawal_patterns": withdrawals,
            "attunement_depth": self.aletheia_state.attunement_depth,
            "overall_disclosure": (
                0.2 * active_truths +
                0.2 * active_beings +
                0.2 * disclosures +
                0.2 * withdrawals +
                0.2 * self.aletheia_state.attunement_depth
            )
        }
        
    def get_disclosure_recommendations(self) -> List[str]:
        """Get recommendations for deepening truth disclosure."""
        recommendations = []
        
        # Check for weak truth modes
        weak_truths = [
            mode for mode, level in self.aletheia_state.truth_modes.items()
            if level < 0.3
        ]
        if weak_truths:
            recommendations.append(
                f"Develop truth modes: {', '.join(t.value for t in weak_truths)}"
            )
            
        # Check being modes
        weak_beings = [
            mode for mode, level in self.aletheia_state.being_modes.items()
            if level < 0.3
        ]
        if weak_beings:
            recommendations.append(
                f"Engage being modes: {', '.join(b.value for b in weak_beings)}"
            )
            
        # Check attunement depth
        if self.aletheia_state.attunement_depth < 0.5:
            recommendations.append(
                "Deepen attunement through balanced presence and withdrawal"
            )
            
        return recommendations 
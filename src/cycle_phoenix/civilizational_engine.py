"""
Civilizational Engine for Cycle of Phoenix

This module implements the civilizational transformation engine that models
the dynamics of rise, peak, and fall phases across historical and future
civilizational cycles.
"""

import time
import math
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading

# Configure logging
logger = logging.getLogger(__name__)


class TransformationPhase(Enum):
    """Phases of civilizational transformation."""
    EMERGENCE = "emergence"          # New patterns emerging
    INTEGRATION = "integration"      # Patterns being integrated
    CRYSTALLIZATION = "crystallization"  # Stable pattern formation
    STAGNATION = "stagnation"       # Patterns becoming rigid
    BREAKDOWN = "breakdown"         # Pattern breakdown beginning
    COLLAPSE = "collapse"           # Complete pattern collapse
    RENEWAL = "renewal"             # Seeds of new patterns


@dataclass
class TransformationMetrics:
    """Metrics for tracking civilizational transformation."""
    complexity: float = 0.5         # System complexity (0.0 to 1.0)
    coherence: float = 0.5          # Internal coherence (0.0 to 1.0)
    adaptability: float = 0.5       # Adaptive capacity (0.0 to 1.0)
    resilience: float = 0.5         # Resilience to disruption (0.0 to 1.0)
    innovation_rate: float = 0.5    # Rate of innovation (0.0 to 1.0)
    integration_depth: float = 0.5  # Depth of integration (0.0 to 1.0)
    energy_flow: float = 0.5        # Energy flow through system (0.0 to 1.0)
    
    def get_overall_vitality(self) -> float:
        """Calculate overall civilizational vitality."""
        return (self.complexity + self.coherence + self.adaptability + 
                self.resilience + self.innovation_rate + self.integration_depth + 
                self.energy_flow) / 7.0


@dataclass
class PsychoTechnology:
    """Represents a psycho-technology within a civilization."""
    name: str
    category: str  # "cognitive", "social", "symbolic", "material"
    emergence_phase: TransformationPhase
    peak_effectiveness: float = 1.0
    current_effectiveness: float = 0.0
    transformation_impact: float = 0.5
    integration_requirements: List[str] = field(default_factory=list)
    
    def update_effectiveness(self, current_phase: TransformationPhase, 
                           phase_progress: float) -> None:
        """Update effectiveness based on current transformation phase."""
        # Calculate effectiveness based on phase alignment
        if current_phase == self.emergence_phase:
            # Peak effectiveness during emergence phase
            self.current_effectiveness = self.peak_effectiveness * phase_progress
        elif self._is_compatible_phase(current_phase):
            # Moderate effectiveness in compatible phases
            self.current_effectiveness = self.peak_effectiveness * 0.7
        else:
            # Reduced effectiveness in incompatible phases
            decay_factor = 1.0 - (phase_progress * 0.3)
            self.current_effectiveness = self.peak_effectiveness * 0.3 * decay_factor
    
    def _is_compatible_phase(self, phase: TransformationPhase) -> bool:
        """Check if phase is compatible with this psycho-technology."""
        compatibility_map = {
            TransformationPhase.EMERGENCE: [TransformationPhase.INTEGRATION],
            TransformationPhase.INTEGRATION: [TransformationPhase.CRYSTALLIZATION, TransformationPhase.EMERGENCE],
            TransformationPhase.CRYSTALLIZATION: [TransformationPhase.INTEGRATION, TransformationPhase.STAGNATION],
            TransformationPhase.STAGNATION: [TransformationPhase.BREAKDOWN],
            TransformationPhase.BREAKDOWN: [TransformationPhase.COLLAPSE, TransformationPhase.RENEWAL],
            TransformationPhase.COLLAPSE: [TransformationPhase.RENEWAL],
            TransformationPhase.RENEWAL: [TransformationPhase.EMERGENCE]
        }
        
        return phase in compatibility_map.get(self.emergence_phase, [])


class CivilizationalEngine:
    """
    Engine for modeling civilizational transformation dynamics.
    
    This engine implements the core mechanisms of civilizational rise, peak,
    and fall, including the role of psycho-technologies, meaning-making
    frameworks, and transformation dynamics.
    """
    
    def __init__(self):
        """Initialize the civilizational engine."""
        # Transformation state
        self.current_phase = TransformationPhase.EMERGENCE
        self.phase_progress = 0.0
        self.transformation_metrics = TransformationMetrics()
        
        # Psycho-technologies active in current civilization
        self.active_psycho_technologies: List[PsychoTechnology] = []
        
        # Transformation dynamics
        self.is_active = False
        self.start_time: Optional[float] = None
        self.phase_duration = 100.0  # Base phase duration in simulation time
        self.last_update_time = 0.0
        
        # Pattern tracking
        self.meaning_making_patterns: Dict[str, float] = {}
        self.institutional_patterns: Dict[str, float] = {}
        self.innovation_patterns: Dict[str, float] = {}
        
        # Callbacks
        self.phase_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.transformation_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Historical tracking
        self.transformation_history: List[Dict[str, Any]] = []
        
        # Threading
        self.engine_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        
        # Initialize default psycho-technologies
        self._initialize_default_psycho_technologies()
    
    def _initialize_default_psycho_technologies(self) -> None:
        """Initialize default set of psycho-technologies."""
        default_psycho_techs = [
            PsychoTechnology(
                name="Language",
                category="symbolic",
                emergence_phase=TransformationPhase.EMERGENCE,
                peak_effectiveness=0.9,
                transformation_impact=0.9
            ),
            PsychoTechnology(
                name="Writing",
                category="symbolic", 
                emergence_phase=TransformationPhase.INTEGRATION,
                peak_effectiveness=0.8,
                transformation_impact=0.8
            ),
            PsychoTechnology(
                name="Literacy", 
                category="cognitive",
                emergence_phase=TransformationPhase.CRYSTALLIZATION,
                peak_effectiveness=0.7,
                transformation_impact=0.7
            ),
            PsychoTechnology(
                name="Scientific Method",
                category="cognitive",
                emergence_phase=TransformationPhase.RENEWAL,
                peak_effectiveness=0.9,
                transformation_impact=0.9
            ),
            PsychoTechnology(
                name="Democratic Institutions",
                category="social",
                emergence_phase=TransformationPhase.INTEGRATION,
                peak_effectiveness=0.6,
                transformation_impact=0.7
            ),
            PsychoTechnology(
                name="Market Economy",
                category="material",
                emergence_phase=TransformationPhase.CRYSTALLIZATION,
                peak_effectiveness=0.7,
                transformation_impact=0.6
            )
        ]
        
        self.active_psycho_technologies = default_psycho_techs
    
    def start_transformation(self) -> None:
        """Start the civilizational transformation process."""
        if self.is_active:
            logger.warning("Civilizational engine is already active")
            return
        
        self.is_active = True
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.should_stop.clear()
        
        # Start transformation thread
        self.engine_thread = threading.Thread(
            target=self._transformation_loop,
            name="CivilizationalEngine"
        )
        self.engine_thread.start()
        
        logger.info("Civilizational transformation started")
    
    def pause_transformation(self) -> None:
        """Pause the transformation process."""
        self.is_active = False
        self.should_stop.set()
        
        if self.engine_thread and self.engine_thread.is_alive():
            self.engine_thread.join(timeout=2.0)
        
        logger.info("Civilizational transformation paused")
    
    def resume_transformation(self) -> None:
        """Resume the transformation process."""
        if self.is_active:
            return
        
        self.start_transformation()
        logger.info("Civilizational transformation resumed")
    
    def _transformation_loop(self) -> None:
        """Main transformation loop."""
        while self.is_active and not self.should_stop.is_set():
            try:
                current_time = time.time()
                if self.start_time:
                    simulation_time = current_time - self.start_time
                    self._update_transformation(simulation_time)
                
                time.sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                logger.error(f"Error in civilizational transformation loop: {e}")
                break
    
    def _update_transformation(self, simulation_time: float) -> None:
        """Update transformation state based on simulation time."""
        # Update phase progress
        phase_elapsed = simulation_time - self.last_update_time
        progress_increment = phase_elapsed / self.phase_duration
        self.phase_progress += progress_increment
        
        # Check for phase transition
        if self.phase_progress >= 1.0:
            self._transition_to_next_phase()
        
        # Update psycho-technologies
        self._update_psycho_technologies()
        
        # Update transformation metrics
        self._update_transformation_metrics()
        
        # Update patterns
        self._update_civilizational_patterns()
        
        # Record history
        if simulation_time - self.last_update_time > 10.0:  # Record every 10 simulation seconds
            self._record_transformation_state(simulation_time)
            self.last_update_time = simulation_time
        
        # Trigger callbacks
        self._trigger_transformation_callbacks(simulation_time)
    
    def _transition_to_next_phase(self) -> None:
        """Transition to the next transformation phase."""
        old_phase = self.current_phase
        
        # Determine next phase based on current phase
        phase_sequence = [
            TransformationPhase.EMERGENCE,
            TransformationPhase.INTEGRATION,
            TransformationPhase.CRYSTALLIZATION,
            TransformationPhase.STAGNATION,
            TransformationPhase.BREAKDOWN,
            TransformationPhase.COLLAPSE,
            TransformationPhase.RENEWAL
        ]
        
        current_index = phase_sequence.index(self.current_phase)
        next_index = (current_index + 1) % len(phase_sequence)
        self.current_phase = phase_sequence[next_index]
        self.phase_progress = 0.0
        
        # Adjust phase duration based on transformation dynamics
        self._adjust_phase_duration()
        
        # Trigger phase change callbacks
        phase_info = {
            'old_phase': old_phase.value,
            'new_phase': self.current_phase.value,
            'phase_transition_time': time.time(),
            'transformation_metrics': self.transformation_metrics.__dict__
        }
        
        for callback in self.phase_callbacks:
            try:
                callback(phase_info)
            except Exception as e:
                logger.error(f"Error in phase callback: {e}")
        
        logger.info(f"Phase transition: {old_phase.value} -> {self.current_phase.value}")
    
    def _adjust_phase_duration(self) -> None:
        """Adjust phase duration based on current transformation state."""
        base_duration = 100.0
        
        # Factors affecting phase duration
        vitality = self.transformation_metrics.get_overall_vitality()
        
        # High vitality phases last longer
        vitality_factor = 0.5 + 1.0 * vitality
        
        # Some phases are naturally longer/shorter
        phase_duration_modifiers = {
            TransformationPhase.EMERGENCE: 0.8,
            TransformationPhase.INTEGRATION: 1.2,
            TransformationPhase.CRYSTALLIZATION: 1.5,
            TransformationPhase.STAGNATION: 1.0,
            TransformationPhase.BREAKDOWN: 0.6,
            TransformationPhase.COLLAPSE: 0.4,
            TransformationPhase.RENEWAL: 0.7
        }
        
        phase_modifier = phase_duration_modifiers.get(self.current_phase, 1.0)
        
        self.phase_duration = base_duration * vitality_factor * phase_modifier
    
    def _update_psycho_technologies(self) -> None:
        """Update effectiveness of active psycho-technologies."""
        for psycho_tech in self.active_psycho_technologies:
            psycho_tech.update_effectiveness(self.current_phase, self.phase_progress)
    
    def _update_transformation_metrics(self) -> None:
        """Update civilizational transformation metrics."""
        # Get aggregate effectiveness of psycho-technologies
        total_effectiveness = sum(pt.current_effectiveness for pt in self.active_psycho_technologies)
        avg_effectiveness = total_effectiveness / len(self.active_psycho_technologies) if self.active_psycho_technologies else 0.0
        
        # Update metrics based on phase and psycho-technology effectiveness
        phase_impacts = self._get_phase_metric_impacts()
        
        # Apply phase impacts
        for metric_name, impact in phase_impacts.items():
            if hasattr(self.transformation_metrics, metric_name):
                current_value = getattr(self.transformation_metrics, metric_name)
                
                # Combine phase impact with psycho-technology effectiveness
                combined_impact = (impact + avg_effectiveness) / 2.0
                
                # Gradual change toward target
                change_rate = 0.1  # 10% change per update
                target_value = combined_impact
                new_value = current_value + (target_value - current_value) * change_rate
                
                # Keep within bounds
                new_value = max(0.0, min(1.0, new_value))
                setattr(self.transformation_metrics, metric_name, new_value)
    
    def _get_phase_metric_impacts(self) -> Dict[str, float]:
        """Get the impact of current phase on transformation metrics."""
        phase_metric_impacts = {
            TransformationPhase.EMERGENCE: {
                'complexity': 0.3,
                'coherence': 0.4,
                'adaptability': 0.8,
                'resilience': 0.6,
                'innovation_rate': 0.9,
                'integration_depth': 0.2,
                'energy_flow': 0.7
            },
            TransformationPhase.INTEGRATION: {
                'complexity': 0.6,
                'coherence': 0.7,
                'adaptability': 0.7,
                'resilience': 0.7,
                'innovation_rate': 0.6,
                'integration_depth': 0.9,
                'energy_flow': 0.8
            },
            TransformationPhase.CRYSTALLIZATION: {
                'complexity': 0.8,
                'coherence': 0.9,
                'adaptability': 0.4,
                'resilience': 0.8,
                'innovation_rate': 0.3,
                'integration_depth': 0.9,
                'energy_flow': 0.6
            },
            TransformationPhase.STAGNATION: {
                'complexity': 0.9,
                'coherence': 0.8,
                'adaptability': 0.2,
                'resilience': 0.5,
                'innovation_rate': 0.1,
                'integration_depth': 0.7,
                'energy_flow': 0.3
            },
            TransformationPhase.BREAKDOWN: {
                'complexity': 0.7,
                'coherence': 0.3,
                'adaptability': 0.6,
                'resilience': 0.2,
                'innovation_rate': 0.7,
                'integration_depth': 0.3,
                'energy_flow': 0.4
            },
            TransformationPhase.COLLAPSE: {
                'complexity': 0.2,
                'coherence': 0.1,
                'adaptability': 0.3,
                'resilience': 0.1,
                'innovation_rate': 0.4,
                'integration_depth': 0.1,
                'energy_flow': 0.2
            },
            TransformationPhase.RENEWAL: {
                'complexity': 0.4,
                'coherence': 0.5,
                'adaptability': 0.9,
                'resilience': 0.6,
                'innovation_rate': 0.8,
                'integration_depth': 0.4,
                'energy_flow': 0.8
            }
        }
        
        return phase_metric_impacts.get(self.current_phase, {})
    
    def _update_civilizational_patterns(self) -> None:
        """Update civilizational patterns based on current state."""
        # Update meaning-making patterns
        self.meaning_making_patterns = {
            'symbolic_coherence': self.transformation_metrics.coherence,
            'narrative_integration': self.transformation_metrics.integration_depth,
            'worldview_stability': 1.0 - self.transformation_metrics.adaptability,
            'meaning_crisis_intensity': 1.0 - self.transformation_metrics.coherence
        }
        
        # Update institutional patterns
        self.institutional_patterns = {
            'institutional_complexity': self.transformation_metrics.complexity,
            'governance_effectiveness': self.transformation_metrics.coherence * self.transformation_metrics.resilience,
            'social_cohesion': self.transformation_metrics.integration_depth,
            'institutional_rigidity': 1.0 - self.transformation_metrics.adaptability
        }
        
        # Update innovation patterns
        self.innovation_patterns = {
            'innovation_capacity': self.transformation_metrics.innovation_rate,
            'knowledge_integration': self.transformation_metrics.integration_depth,
            'creative_destruction': self.transformation_metrics.adaptability,
            'technological_momentum': self.transformation_metrics.energy_flow
        }
    
    def _record_transformation_state(self, simulation_time: float) -> None:
        """Record current transformation state to history."""
        state_record = {
            'timestamp': time.time(),
            'simulation_time': simulation_time,
            'phase': self.current_phase.value,
            'phase_progress': self.phase_progress,
            'transformation_metrics': self.transformation_metrics.__dict__,
            'psycho_tech_effectiveness': {
                pt.name: pt.current_effectiveness 
                for pt in self.active_psycho_technologies
            },
            'meaning_making_patterns': self.meaning_making_patterns.copy(),
            'institutional_patterns': self.institutional_patterns.copy(),
            'innovation_patterns': self.innovation_patterns.copy()
        }
        
        self.transformation_history.append(state_record)
        
        # Limit history size
        if len(self.transformation_history) > 1000:
            self.transformation_history = self.transformation_history[-500:]
    
    def _trigger_transformation_callbacks(self, simulation_time: float) -> None:
        """Trigger transformation update callbacks."""
        callback_info = {
            'simulation_time': simulation_time,
            'phase': self.current_phase.value,
            'phase_progress': self.phase_progress,
            'vitality': self.transformation_metrics.get_overall_vitality(),
            'metrics': self.transformation_metrics.__dict__
        }
        
        for callback in self.transformation_callbacks:
            try:
                callback(callback_info)
            except Exception as e:
                logger.error(f"Error in transformation callback: {e}")
    
    def register_phase_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for phase transitions.
        
        Args:
            callback: Function to call on phase transitions
        """
        self.phase_callbacks.append(callback)
    
    def register_transformation_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for transformation updates.
        
        Args:
            callback: Function to call on transformation updates
        """
        self.transformation_callbacks.append(callback)
    
    def sync_with_antikythera(self, antikythera_info: Dict[str, Any]) -> None:
        """Synchronize with Antikythera orchestrator information.
        
        Args:
            antikythera_info: Information from Antikythera orchestrator
        """
        # Extract relevant information from Antikythera
        meta_characteristics = antikythera_info.get('meta_characteristics', {})
        
        # Adjust transformation metrics based on Antikythera state
        if meta_characteristics:
            energy = meta_characteristics.get('energy', 0.5)
            stability = meta_characteristics.get('stability', 0.5)
            transformation_intensity = meta_characteristics.get('transformation_intensity', 0.5)
            
            # Influence transformation metrics
            self.transformation_metrics.energy_flow = (
                self.transformation_metrics.energy_flow * 0.8 + energy * 0.2
            )
            
            self.transformation_metrics.coherence = (
                self.transformation_metrics.coherence * 0.8 + stability * 0.2
            )
            
            # Transformation intensity affects adaptability
            self.transformation_metrics.adaptability = (
                self.transformation_metrics.adaptability * 0.8 + transformation_intensity * 0.2
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status.
        
        Returns:
            Dictionary containing current status
        """
        return {
            'is_active': self.is_active,
            'current_phase': self.current_phase.value,
            'phase_progress': self.phase_progress,
            'vitality': self.transformation_metrics.get_overall_vitality(),
            'metrics': self.transformation_metrics.__dict__,
            'active_psycho_technologies': len(self.active_psycho_technologies),
            'history_length': len(self.transformation_history)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete engine state.
        
        Returns:
            Complete state information
        """
        return {
            'current_phase': self.current_phase.value,
            'phase_progress': self.phase_progress,
            'transformation_metrics': self.transformation_metrics.__dict__,
            'psycho_technologies': [
                {
                    'name': pt.name,
                    'category': pt.category,
                    'current_effectiveness': pt.current_effectiveness,
                    'transformation_impact': pt.transformation_impact
                }
                for pt in self.active_psycho_technologies
            ],
            'patterns': {
                'meaning_making': self.meaning_making_patterns,
                'institutional': self.institutional_patterns,
                'innovation': self.innovation_patterns
            }
        }
    
    def get_transformation_insights(self) -> Dict[str, Any]:
        """Get insights from transformation process.
        
        Returns:
            Dictionary containing transformation insights
        """
        if not self.transformation_history:
            return {'error': 'No transformation history available'}
        
        # Analyze trends over history
        vitality_trend = []
        phase_durations = {}
        
        current_phase_start = None
        for record in self.transformation_history:
            vitality = sum(record['transformation_metrics'].values()) / len(record['transformation_metrics'])
            vitality_trend.append(vitality)
            
            # Track phase durations
            phase = record['phase']
            if current_phase_start is None or record['phase'] != current_phase_start[0]:
                if current_phase_start is not None:
                    duration = record['simulation_time'] - current_phase_start[1]
                    phase = current_phase_start[0]
                    if phase not in phase_durations:
                        phase_durations[phase] = []
                    phase_durations[phase].append(duration)
                current_phase_start = (record['phase'], record['simulation_time'])
        
        # Calculate insights
        insights = {
            'average_vitality': sum(vitality_trend) / len(vitality_trend),
            'vitality_trend': 'increasing' if vitality_trend[-1] > vitality_trend[0] else 'decreasing',
            'most_effective_psycho_tech': self._get_most_effective_psycho_tech(),
            'phase_durations': {phase: sum(durations) / len(durations) 
                              for phase, durations in phase_durations.items() if durations},
            'current_transformation_stage': self._assess_transformation_stage()
        }
        
        return insights
    
    def _get_most_effective_psycho_tech(self) -> str:
        """Get the most effective psycho-technology."""
        if not self.active_psycho_technologies:
            return "None"
        
        most_effective = max(self.active_psycho_technologies, 
                           key=lambda pt: pt.current_effectiveness)
        return most_effective.name
    
    def _assess_transformation_stage(self) -> str:
        """Assess the current transformation stage."""
        vitality = self.transformation_metrics.get_overall_vitality()
        
        if vitality > 0.8:
            return "High vitality - flourishing civilization"
        elif vitality > 0.6:
            return "Moderate vitality - stable civilization"
        elif vitality > 0.4:
            return "Low vitality - struggling civilization"
        elif vitality > 0.2:
            return "Crisis - civilization under stress"
        else:
            return "Collapse - civilizational breakdown"
"""
Temporal Coordination for Antikythera Framework

This module provides temporal coordination capabilities for managing multiple
time scales and temporal patterns within the civilizational cycle framework.
"""

import time
import math
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading


class TimeScale(Enum):
    """Different time scales for temporal coordination."""
    MICROSECOND = "microsecond"  # Individual cognitive processes
    SECOND = "second"            # Immediate reactions
    MINUTE = "minute"            # Short-term planning  
    HOUR = "hour"               # Task completion
    DAY = "day"                 # Daily cycles
    MONTH = "month"             # Seasonal patterns
    YEAR = "year"               # Annual cycles
    DECADE = "decade"           # Generational changes
    CENTURY = "century"         # Cultural shifts
    MILLENNIUM = "millennium"   # Civilizational cycles


@dataclass
class TemporalPattern:
    """Represents a temporal pattern across multiple scales."""
    name: str
    time_scale: TimeScale
    frequency: float  # Cycles per unit time
    amplitude: float  # Strength of pattern
    phase_offset: float = 0.0  # Phase offset in radians
    decay_rate: float = 0.0    # Pattern decay over time
    
    def evaluate_at_time(self, t: float) -> float:
        """Evaluate pattern strength at given time.
        
        Args:
            t: Time value
            
        Returns:
            Pattern strength at time t
        """
        # Apply decay
        decayed_amplitude = self.amplitude * math.exp(-self.decay_rate * t)
        
        # Calculate oscillatory component
        oscillation = math.sin(2 * math.pi * self.frequency * t + self.phase_offset)
        
        return decayed_amplitude * oscillation


class TemporalCoordinator:
    """Coordinates temporal patterns across multiple time scales."""
    
    def __init__(self):
        """Initialize temporal coordinator."""
        self.patterns: Dict[str, TemporalPattern] = {}
        self.time_scales: Dict[TimeScale, float] = self._initialize_time_scales()
        
        # Coordination state
        self.start_time: Optional[float] = None
        self.is_active = False
        
        # Pattern synchronization
        self.sync_events: List[Tuple[float, str, Dict[str, Any]]] = []
        self.pattern_interactions: Dict[Tuple[str, str], float] = {}
        
        # Threading for real-time coordination
        self.coordination_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
    
    def _initialize_time_scales(self) -> Dict[TimeScale, float]:
        """Initialize time scale conversion factors (to seconds)."""
        return {
            TimeScale.MICROSECOND: 1e-6,
            TimeScale.SECOND: 1.0,
            TimeScale.MINUTE: 60.0,
            TimeScale.HOUR: 3600.0,
            TimeScale.DAY: 86400.0,
            TimeScale.MONTH: 86400.0 * 30.44,  # Average month
            TimeScale.YEAR: 86400.0 * 365.25,  # Accounting for leap years
            TimeScale.DECADE: 86400.0 * 365.25 * 10,
            TimeScale.CENTURY: 86400.0 * 365.25 * 100,
            TimeScale.MILLENNIUM: 86400.0 * 365.25 * 1000
        }
    
    def add_pattern(self, pattern: TemporalPattern) -> None:
        """Add a temporal pattern to coordinate.
        
        Args:
            pattern: The temporal pattern to add
        """
        self.patterns[pattern.name] = pattern
    
    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove a temporal pattern.
        
        Args:
            pattern_name: Name of pattern to remove
            
        Returns:
            True if pattern was removed, False if not found
        """
        if pattern_name in self.patterns:
            del self.patterns[pattern_name]
            return True
        return False
    
    def set_pattern_interaction(self, pattern1: str, pattern2: str, 
                              interaction_strength: float) -> None:
        """Set interaction strength between two patterns.
        
        Args:
            pattern1: Name of first pattern
            pattern2: Name of second pattern
            interaction_strength: Strength of interaction (-1.0 to 1.0)
        """
        key = (pattern1, pattern2) if pattern1 < pattern2 else (pattern2, pattern1)
        self.pattern_interactions[key] = interaction_strength
    
    def start(self) -> None:
        """Start temporal coordination."""
        self.is_active = True
        self.start_time = time.time()
        self.should_stop.clear()
        
        # Start coordination thread for real-time patterns
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            name="TemporalCoordinator"
        )
        self.coordination_thread.start()
    
    def stop(self) -> None:
        """Stop temporal coordination."""
        self.is_active = False
        self.should_stop.set()
        
        if self.coordination_thread and self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=2.0)
    
    def _coordination_loop(self) -> None:
        """Main coordination loop for real-time patterns."""
        while self.is_active and not self.should_stop.is_set():
            current_time = time.time()
            
            if self.start_time:
                elapsed_time = current_time - self.start_time
                
                # Update patterns and check for synchronization events
                self._check_synchronization_events(elapsed_time)
                
                # Apply pattern interactions
                self._apply_pattern_interactions(elapsed_time)
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
    
    def _check_synchronization_events(self, current_time: float) -> None:
        """Check for pattern synchronization events.
        
        Args:
            current_time: Current elapsed time
        """
        # Look for patterns that are reaching phase alignment
        pattern_phases = {}
        
        for name, pattern in self.patterns.items():
            if pattern.frequency > 0:
                # Calculate current phase
                phase = (2 * math.pi * pattern.frequency * current_time + 
                        pattern.phase_offset) % (2 * math.pi)
                pattern_phases[name] = phase
        
        # Check for phase alignment (within threshold)
        alignment_threshold = 0.1  # radians
        
        for i, (name1, phase1) in enumerate(pattern_phases.items()):
            for name2, phase2 in list(pattern_phases.items())[i+1:]:
                phase_diff = abs(phase1 - phase2)
                phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
                
                if phase_diff < alignment_threshold:
                    # Patterns are aligned - record synchronization event
                    sync_event = (current_time, "phase_alignment", {
                        'patterns': [name1, name2],
                        'phase_difference': phase_diff
                    })
                    self.sync_events.append(sync_event)
    
    def _apply_pattern_interactions(self, current_time: float) -> None:
        """Apply interactions between patterns.
        
        Args:
            current_time: Current elapsed time
        """
        # Calculate interaction effects between patterns
        for (pattern1_name, pattern2_name), strength in self.pattern_interactions.items():
            if pattern1_name in self.patterns and pattern2_name in self.patterns:
                pattern1 = self.patterns[pattern1_name]
                pattern2 = self.patterns[pattern2_name]
                
                # Get current pattern values
                value1 = pattern1.evaluate_at_time(current_time)
                value2 = pattern2.evaluate_at_time(current_time)
                
                # Apply interaction (simple mutual influence model)
                interaction_effect = strength * value1 * value2 * 0.01
                
                # Modify pattern amplitudes slightly
                pattern1.amplitude += interaction_effect
                pattern2.amplitude += interaction_effect
                
                # Keep amplitudes in reasonable bounds
                pattern1.amplitude = max(0.0, min(2.0, pattern1.amplitude))
                pattern2.amplitude = max(0.0, min(2.0, pattern2.amplitude))
    
    def update_patterns(self, pattern_data: List[Dict[str, Any]]) -> None:
        """Update patterns based on external data.
        
        Args:
            pattern_data: List of pattern update data
        """
        for data in pattern_data:
            pattern_name = data.get('name')
            if not pattern_name:
                continue
            
            # Create or update pattern
            if pattern_name not in self.patterns:
                # Create new pattern from data
                time_scale = TimeScale(data.get('time_scale', 'year'))
                frequency = data.get('frequency', 1.0)
                amplitude = data.get('amplitude', 1.0)
                
                pattern = TemporalPattern(
                    name=pattern_name,
                    time_scale=time_scale,
                    frequency=frequency,
                    amplitude=amplitude
                )
                self.patterns[pattern_name] = pattern
            else:
                # Update existing pattern
                pattern = self.patterns[pattern_name]
                
                # Update properties if provided
                if 'amplitude' in data:
                    pattern.amplitude = data['amplitude']
                if 'frequency' in data:
                    pattern.frequency = data['frequency']
                if 'phase_offset' in data:
                    pattern.phase_offset = data['phase_offset']
    
    def get_current_temporal_state(self) -> Dict[str, Any]:
        """Get current state of all temporal patterns.
        
        Returns:
            Dictionary containing current temporal state
        """
        if not self.start_time:
            return {}
        
        current_time = time.time() - self.start_time
        
        state = {
            'current_time': current_time,
            'patterns': {},
            'synchronization_events': len(self.sync_events),
            'active_interactions': len(self.pattern_interactions)
        }
        
        # Evaluate all patterns at current time
        for name, pattern in self.patterns.items():
            state['patterns'][name] = {
                'value': pattern.evaluate_at_time(current_time),
                'amplitude': pattern.amplitude,
                'frequency': pattern.frequency,
                'time_scale': pattern.time_scale.value
            }
        
        return state
    
    def synchronize_with_civilizational_cycles(self, cycle_data: List[Dict[str, Any]]) -> None:
        """Synchronize temporal patterns with civilizational cycle data.
        
        Args:
            cycle_data: Data from active civilizational cycles
        """
        for cycle_info in cycle_data:
            cycle_id = cycle_info.get('cycle_id', 0)
            phase_type = cycle_info.get('phase_type', 'unknown')
            energy = cycle_info.get('energy', 0.5)
            stability = cycle_info.get('stability', 0.5)
            progress = cycle_info.get('progress', 0.0)
            
            # Create temporal pattern for this cycle
            pattern_name = f"cycle_{cycle_id}_{phase_type}"
            
            # Calculate pattern parameters from cycle characteristics
            # Higher energy = higher amplitude
            # Lower stability = higher frequency (more oscillation)
            amplitude = energy
            frequency = (1.0 - stability) * 0.1 + 0.01  # Base frequency
            
            # Phase offset based on progress
            phase_offset = progress * 2 * math.pi
            
            # Update or create pattern
            if pattern_name in self.patterns:
                pattern = self.patterns[pattern_name]
                pattern.amplitude = amplitude
                pattern.frequency = frequency
                pattern.phase_offset = phase_offset
            else:
                pattern = TemporalPattern(
                    name=pattern_name,
                    time_scale=TimeScale.CENTURY,  # Civilizational scale
                    frequency=frequency,
                    amplitude=amplitude,
                    phase_offset=phase_offset
                )
                self.patterns[pattern_name] = pattern
    
    def calculate_temporal_resonance(self) -> float:
        """Calculate overall temporal resonance across all patterns.
        
        Returns:
            Resonance value (0.0 to 1.0)
        """
        if not self.start_time or not self.patterns:
            return 0.0
        
        current_time = time.time() - self.start_time
        
        # Calculate cross-correlation of all patterns
        pattern_values = []
        for pattern in self.patterns.values():
            value = pattern.evaluate_at_time(current_time)
            pattern_values.append(value)
        
        if len(pattern_values) < 2:
            return 1.0  # Perfect resonance with only one pattern
        
        # Calculate variance of pattern values
        mean_value = sum(pattern_values) / len(pattern_values)
        variance = sum((v - mean_value) ** 2 for v in pattern_values) / len(pattern_values)
        
        # Resonance is inversely related to variance
        # High resonance when patterns are synchronized (low variance)
        max_variance = 2.0  # Maximum expected variance
        resonance = 1.0 - min(variance / max_variance, 1.0)
        
        return resonance
    
    def predict_next_synchronization(self, pattern1_name: str, pattern2_name: str) -> Optional[float]:
        """Predict when two patterns will next synchronize.
        
        Args:
            pattern1_name: Name of first pattern
            pattern2_name: Name of second pattern
            
        Returns:
            Time until next synchronization, or None if patterns don't synchronize
        """
        if (pattern1_name not in self.patterns or 
            pattern2_name not in self.patterns):
            return None
        
        pattern1 = self.patterns[pattern1_name]
        pattern2 = self.patterns[pattern2_name]
        
        # Calculate frequency difference
        freq_diff = abs(pattern1.frequency - pattern2.frequency)
        
        if freq_diff == 0:
            # Same frequency - always synchronized or never synchronized
            return 0.0 if pattern1.phase_offset == pattern2.phase_offset else None
        
        # Calculate beat frequency (how often patterns align)
        beat_frequency = freq_diff
        
        # Time until next alignment
        next_sync_time = 1.0 / beat_frequency
        
        return next_sync_time
    
    def create_civilizational_temporal_model(self) -> Dict[str, TemporalPattern]:
        """Create temporal patterns for standard civilizational cycles.
        
        Returns:
            Dictionary of standard civilizational temporal patterns
        """
        civilizational_patterns = {}
        
        # 600-year civilizational cycle
        main_cycle = TemporalPattern(
            name="civilizational_cycle",
            time_scale=TimeScale.CENTURY,
            frequency=1.0 / 600.0,  # One cycle per 600 years
            amplitude=1.0
        )
        civilizational_patterns["civilizational_cycle"] = main_cycle
        
        # 200-year phase cycle
        phase_cycle = TemporalPattern(
            name="phase_cycle", 
            time_scale=TimeScale.CENTURY,
            frequency=1.0 / 200.0,  # Three phases per 600 years
            amplitude=0.8
        )
        civilizational_patterns["phase_cycle"] = phase_cycle
        
        # Innovation waves (shorter cycles)
        innovation_cycle = TemporalPattern(
            name="innovation_wave",
            time_scale=TimeScale.DECADE,
            frequency=1.0 / 50.0,  # Innovation waves every ~50 years
            amplitude=0.6
        )
        civilizational_patterns["innovation_wave"] = innovation_cycle
        
        # Generational change
        generational_cycle = TemporalPattern(
            name="generational_change",
            time_scale=TimeScale.DECADE,
            frequency=1.0 / 25.0,  # Generational change every ~25 years
            amplitude=0.4
        )
        civilizational_patterns["generational_change"] = generational_cycle
        
        # Set up interactions between patterns
        # Main cycle influences phase cycle
        main_cycle_key = ("civilizational_cycle", "phase_cycle")
        self.pattern_interactions[main_cycle_key] = 0.8
        
        # Phase cycle influences innovation waves
        phase_innovation_key = ("phase_cycle", "innovation_wave")
        self.pattern_interactions[phase_innovation_key] = 0.6
        
        # Innovation waves influence generational change
        innovation_gen_key = ("innovation_wave", "generational_change")
        self.pattern_interactions[innovation_gen_key] = 0.4
        
        return civilizational_patterns
    
    def get_pattern_synchronization_matrix(self) -> Dict[Tuple[str, str], float]:
        """Calculate synchronization matrix for all pattern pairs.
        
        Returns:
            Matrix showing synchronization strength between all pattern pairs
        """
        if not self.start_time:
            return {}
        
        current_time = time.time() - self.start_time
        sync_matrix = {}
        
        pattern_names = list(self.patterns.keys())
        
        for i, name1 in enumerate(pattern_names):
            for name2 in pattern_names[i+1:]:
                pattern1 = self.patterns[name1]
                pattern2 = self.patterns[name2]
                
                # Calculate phase difference
                phase1 = (2 * math.pi * pattern1.frequency * current_time + 
                         pattern1.phase_offset) % (2 * math.pi)
                phase2 = (2 * math.pi * pattern2.frequency * current_time + 
                         pattern2.phase_offset) % (2 * math.pi)
                
                phase_diff = abs(phase1 - phase2)
                phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
                
                # Synchronization strength (1.0 for perfect sync, 0.0 for opposite)
                sync_strength = 1.0 - (phase_diff / math.pi)
                
                sync_matrix[(name1, name2)] = sync_strength
        
        return sync_matrix
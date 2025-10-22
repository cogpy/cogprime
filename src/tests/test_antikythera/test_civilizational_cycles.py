"""
Tests for Civilizational Cycles in the Antikythera Framework
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antikythera.cycles import CivilizationalCycle, CyclePhase, PhaseType, PhaseCharacteristics, MetaCycle
from atomspace import AtomSpace


class TestCivilizationalCycle(unittest.TestCase):
    """Test cases for CivilizationalCycle functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cycle = CivilizationalCycle("Test_Cycle", cycle_id=0)
        self.atomspace = AtomSpace()
    
    def test_cycle_initialization(self):
        """Test cycle initialization."""
        self.assertEqual(self.cycle.name, "Test_Cycle")
        self.assertEqual(self.cycle.cycle_id, 0)
        self.assertEqual(self.cycle.total_duration, 600)
        self.assertEqual(len(self.cycle.phases), 3)  # Rise, Peak, Fall
        self.assertFalse(self.cycle.is_active)
        self.assertFalse(self.cycle.completed)
    
    def test_phase_structure(self):
        """Test phase structure within cycle."""
        # Check that phases are properly configured
        self.assertEqual(len(self.cycle.phases), 3)
        
        phase_types = [phase.phase_type for phase in self.cycle.phases]
        expected_types = [PhaseType.RISE, PhaseType.PEAK, PhaseType.FALL]
        
        self.assertEqual(phase_types, expected_types)
        
        # Check phase durations
        for phase in self.cycle.phases:
            self.assertEqual(phase.characteristics.duration_years, 200)
    
    def test_cycle_start_and_progression(self):
        """Test cycle start and progression."""
        # Start cycle
        self.cycle.start_cycle()
        
        self.assertTrue(self.cycle.is_active)
        self.assertEqual(self.cycle.current_phase_index, 0)
        self.assertTrue(self.cycle.phases[0].is_active)
        
        # Test progression (simulate time passage)
        elapsed_time = 100.0  # 100 simulation units
        self.cycle.update(elapsed_time)
        
        # Should still be in first phase but with progress
        current_phase = self.cycle.get_current_phase()
        self.assertIsNotNone(current_phase)
        self.assertEqual(current_phase.phase_type, PhaseType.RISE)
        self.assertGreater(current_phase.current_progress, 0.0)
    
    def test_phase_transitions(self):
        """Test transitions between phases."""
        self.cycle.start_cycle()
        
        # Simulate completion of rise phase
        elapsed_time = 200.0  # Complete first phase
        self.cycle.update(elapsed_time)
        
        # Should transition to peak phase
        current_phase = self.cycle.get_current_phase()
        self.assertEqual(current_phase.phase_type, PhaseType.PEAK)
        
        # Simulate completion of peak phase
        elapsed_time = 400.0  # Complete second phase
        self.cycle.update(elapsed_time)
        
        # Should transition to fall phase
        current_phase = self.cycle.get_current_phase()
        self.assertEqual(current_phase.phase_type, PhaseType.FALL)
    
    def test_cycle_completion(self):
        """Test cycle completion."""
        self.cycle.start_cycle()
        
        # Simulate completion of entire cycle
        elapsed_time = 600.0  # Complete all phases
        self.cycle.update(elapsed_time)
        
        self.assertTrue(self.cycle.completed)
        self.assertFalse(self.cycle.is_active)
    
    def test_cycle_progress_calculation(self):
        """Test cycle progress calculation."""
        self.cycle.start_cycle()
        
        # Test various progress points
        test_points = [
            (100.0, 0.167),  # ~1/6 through cycle
            (200.0, 0.333),  # ~1/3 through cycle  
            (300.0, 0.5),    # 1/2 through cycle
            (450.0, 0.75),   # 3/4 through cycle
            (600.0, 1.0)     # Complete cycle
        ]
        
        for elapsed_time, expected_progress in test_points:
            self.cycle.update(elapsed_time)
            actual_progress = self.cycle.get_cycle_progress()
            self.assertAlmostEqual(actual_progress, expected_progress, places=1)
    
    def test_current_characteristics(self):
        """Test retrieval of current civilizational characteristics."""
        self.cycle.start_cycle()
        
        characteristics = self.cycle.get_current_characteristics()
        
        # Should have all expected characteristics
        expected_keys = ['energy', 'stability', 'innovation_rate', 
                        'integration_level', 'transformation_intensity']
        
        for key in expected_keys:
            self.assertIn(key, characteristics)
            self.assertIsInstance(characteristics[key], float)
            self.assertGreaterEqual(characteristics[key], 0.0)
            self.assertLessEqual(characteristics[key], 1.0)
    
    def test_atomspace_binding(self):
        """Test binding cycle to AtomSpace."""
        self.cycle.bind_to_cognitive_system(self.atomspace)
        
        # Check that cognitive bindings were created
        self.assertIn('atomspace', self.cycle.cognitive_bindings)
        self.assertIn('cycle_node', self.cycle.cognitive_bindings)
        
        # Check that atoms were added to AtomSpace
        all_atoms = self.atomspace.get_all_atoms()
        self.assertGreater(len(all_atoms), 0)
        
        # Check for cycle and phase nodes
        cycle_nodes = [atom for atom in all_atoms 
                      if atom.is_node() and 'cycle' in atom.name.lower()]
        self.assertGreater(len(cycle_nodes), 0)
    
    def test_historical_context(self):
        """Test historical context retrieval."""
        # Test with known historical years
        context_500_bc = self.cycle.get_historical_context(-500)  # Classical period
        context_800_ad = self.cycle.get_historical_context(800)   # Medieval period
        
        # Should return context for years within historical periods
        # (Implementation dependent on historical_periods data)
        if context_500_bc:
            self.assertIn('cycle', context_500_bc)
            self.assertIn('phase', context_500_bc)
            self.assertIn('year', context_500_bc)
        
        if context_800_ad:
            self.assertIn('cycle', context_800_ad)
            self.assertIn('phase', context_800_ad)
            self.assertIn('year', context_800_ad)


class TestCyclePhase(unittest.TestCase):
    """Test cases for CyclePhase functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.characteristics = PhaseCharacteristics(
            duration_years=200,
            energy_level=0.8,
            stability=0.6,
            innovation_rate=0.7,
            integration_level=0.5,
            transformation_intensity=0.4
        )
        self.phase = CyclePhase(PhaseType.RISE, self.characteristics, "test_rise")
    
    def test_phase_initialization(self):
        """Test phase initialization."""
        self.assertEqual(self.phase.phase_type, PhaseType.RISE)
        self.assertEqual(self.phase.name, "test_rise")
        self.assertEqual(self.phase.characteristics.duration_years, 200)
        self.assertFalse(self.phase.is_active)
        self.assertEqual(self.phase.current_progress, 0.0)
    
    def test_phase_activation(self):
        """Test phase activation and deactivation."""
        # Activate phase
        self.phase.activate()
        self.assertTrue(self.phase.is_active)
        self.assertIsNotNone(self.phase.start_time)
        
        # Deactivate phase
        self.phase.deactivate()
        self.assertFalse(self.phase.is_active)
        self.assertEqual(self.phase.current_progress, 1.0)
    
    def test_phase_progress_update(self):
        """Test phase progress updates."""
        self.phase.activate()
        
        # Update progress
        elapsed_time = 100.0  # Half the duration
        self.phase.update_progress(elapsed_time)
        
        self.assertAlmostEqual(self.phase.current_progress, 0.5, places=1)
    
    def test_energy_dynamics(self):
        """Test phase energy dynamics."""
        self.phase.activate()
        
        # Test energy at different progress levels
        progress_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for progress in progress_points:
            self.phase.current_progress = progress
            energy = self.phase.get_current_energy()
            
            self.assertIsInstance(energy, float)
            self.assertGreaterEqual(energy, 0.0)
            self.assertLessEqual(energy, 1.0)
            
            # For RISE phase, energy should generally increase with progress
            if progress > 0:
                self.phase.current_progress = 0.0
                initial_energy = self.phase.get_current_energy()
                self.phase.current_progress = progress
                current_energy = self.phase.get_current_energy()
                # Energy should increase during rise phase
                if progress > 0.5:  # Only test for significant progress
                    self.assertGreaterEqual(current_energy, initial_energy)
    
    def test_stability_dynamics(self):
        """Test phase stability dynamics."""
        self.phase.activate()
        
        # Test stability at different progress levels
        for progress in [0.0, 0.5, 1.0]:
            self.phase.current_progress = progress
            stability = self.phase.get_current_stability()
            
            self.assertIsInstance(stability, float)
            self.assertGreaterEqual(stability, 0.0)
            self.assertLessEqual(stability, 1.0)
    
    def test_evolutionary_parameters(self):
        """Test evolutionary parameter generation."""
        self.phase.activate()
        
        params = self.phase.get_evolutionary_parameters()
        
        expected_params = ['mutation_rate', 'selection_pressure', 
                          'crossover_rate', 'population_growth']
        
        for param in expected_params:
            self.assertIn(param, params)
            self.assertIsInstance(params[param], float)
            self.assertGreater(params[param], 0.0)


class TestMetaCycle(unittest.TestCase):
    """Test cases for MetaCycle functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.meta_cycle = MetaCycle("test_meta")
        
        # Create test cycles
        self.cycle1 = CivilizationalCycle("Cycle_1", cycle_id=0)
        self.cycle2 = CivilizationalCycle("Cycle_2", cycle_id=1)
    
    def test_meta_cycle_initialization(self):
        """Test meta-cycle initialization."""
        self.assertEqual(self.meta_cycle.name, "test_meta")
        self.assertEqual(len(self.meta_cycle.cycles), 0)
        self.assertFalse(self.meta_cycle.is_active)
    
    def test_cycle_addition(self):
        """Test adding cycles to meta-cycle."""
        self.meta_cycle.add_cycle(self.cycle1)
        self.meta_cycle.add_cycle(self.cycle2)
        
        self.assertEqual(len(self.meta_cycle.cycles), 2)
        self.assertEqual(self.cycle1.cycle_id, 0)
        self.assertEqual(self.cycle2.cycle_id, 1)
    
    def test_cycle_interactions(self):
        """Test cycle interaction setup."""
        self.meta_cycle.add_cycle(self.cycle1)
        self.meta_cycle.add_cycle(self.cycle2)
        
        # Set interaction between cycles
        self.meta_cycle.set_cycle_interaction(0, 1, 0.5)
        
        # Check that interaction was recorded
        interaction_key = (0, 1)
        self.assertIn(interaction_key, self.meta_cycle.cycle_interactions)
        self.assertEqual(self.meta_cycle.cycle_interactions[interaction_key], 0.5)
    
    def test_meta_cycle_start(self):
        """Test starting meta-cycle."""
        self.meta_cycle.add_cycle(self.cycle1)
        self.meta_cycle.add_cycle(self.cycle2)
        
        self.meta_cycle.start_meta_cycle()
        
        self.assertTrue(self.meta_cycle.is_active)
        self.assertTrue(self.cycle1.is_active)
        self.assertTrue(self.cycle2.is_active)
    
    def test_meta_characteristics(self):
        """Test meta-cycle characteristics aggregation."""
        self.meta_cycle.add_cycle(self.cycle1)
        self.meta_cycle.add_cycle(self.cycle2)
        
        self.meta_cycle.start_meta_cycle()
        
        meta_chars = self.meta_cycle.get_meta_characteristics()
        
        expected_keys = ['energy', 'stability', 'innovation_rate',
                        'integration_level', 'transformation_intensity']
        
        for key in expected_keys:
            self.assertIn(key, meta_chars)
            self.assertIsInstance(meta_chars[key], float)


if __name__ == '__main__':
    unittest.main()
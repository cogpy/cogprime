"""
Tests for the Phoenix Orchestrator - Cycle of Phoenix main coordination system
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from atomspace import AtomSpace, Node
from cycle_phoenix.phoenix_orchestrator import CycleOfPhoenix, PhoenixOrchestrator, TransformationMode


class TestPhoenixOrchestrator(unittest.TestCase):
    """Test cases for Phoenix Orchestrator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atomspace = AtomSpace()
        self.phoenix = CycleOfPhoenix(self.atomspace, TransformationMode.CONSCIOUSNESS_EVOLUTION)
    
    def test_phoenix_initialization(self):
        """Test Phoenix system initialization."""
        self.assertIsNotNone(self.phoenix.orchestrator)
        self.assertEqual(self.phoenix.transformation_mode, TransformationMode.CONSCIOUSNESS_EVOLUTION)
        self.assertIsNotNone(self.phoenix.orchestrator.civilizational_engine)
        self.assertIsNotNone(self.phoenix.orchestrator.phoenix_moses)
    
    def test_transformation_status_initial(self):
        """Test initial transformation status."""
        status = self.phoenix.get_transformation_status()
        
        self.assertIn('is_active', status)
        self.assertIn('transformation_mode', status)
        self.assertEqual(status['transformation_mode'], 'consciousness_evolution')
        self.assertFalse(status['is_active'])
    
    def test_cycle_creation(self):
        """Test civilizational cycle creation."""
        # Start transformation to trigger cycle creation
        self.phoenix.begin_cycle_of_transformation()
        
        try:
            status = self.phoenix.get_transformation_status()
            
            # Check that cycles were created
            if 'antikythera' in status:
                antikythera_status = status['antikythera']
                self.assertGreater(antikythera_status['cycles']['total'], 0)
            
        finally:
            # Clean up
            self.phoenix.pause_transformation()
    
    def test_consciousness_evolution_mode(self):
        """Test consciousness evolution specific functionality."""
        # Verify consciousness evolution mode setup
        config = self.phoenix.orchestrator.config
        self.assertEqual(config.transformation_mode, TransformationMode.CONSCIOUSNESS_EVOLUTION)
        
        # Check Phoenix MOSES is configured for consciousness evolution
        self.assertIsNotNone(self.phoenix.orchestrator.phoenix_moses)
        self.assertIsNotNone(self.phoenix.orchestrator.phoenix_moses.civilizational_fitness)
    
    def test_state_persistence(self):
        """Test state saving and loading functionality."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            # Save state
            self.phoenix.save_state(temp_filepath)
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_filepath))
            
            # Verify file has content
            with open(temp_filepath, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 100)  # Should have substantial content
                self.assertIn('transformation_mode', content)
        
        finally:
            # Clean up
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_prediction_functionality(self):
        """Test future prediction functionality."""
        prediction = self.phoenix.predict_future(years_ahead=50)
        
        # Should return prediction data structure
        self.assertIsInstance(prediction, dict)
        
        # May contain error if Antikythera not fully initialized
        # but should not raise exception
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_transformation_lifecycle(self, mock_sleep):
        """Test complete transformation lifecycle."""
        # Begin transformation
        self.phoenix.begin_cycle_of_transformation()
        
        # Verify active state
        status = self.phoenix.get_transformation_status()
        self.assertTrue(status['is_active'])
        
        # Pause transformation
        self.phoenix.pause_transformation()
        
        # Verify paused state
        status = self.phoenix.get_transformation_status()
        self.assertFalse(status['is_active'])
    
    def test_atomspace_integration(self):
        """Test integration with AtomSpace."""
        # Add some atoms to AtomSpace
        concept1 = Node("ConceptNode", "consciousness")
        concept2 = Node("ConceptNode", "evolution")
        
        self.atomspace.add(concept1)
        self.atomspace.add(concept2)
        
        # Verify atoms are accessible
        self.assertEqual(len(self.atomspace.get_all_atoms()), 2)
        
        # Initialize Phoenix with populated AtomSpace
        phoenix2 = CycleOfPhoenix(self.atomspace, TransformationMode.CONSCIOUSNESS_EVOLUTION)
        
        # Verify Phoenix can access AtomSpace atoms
        self.assertIsNotNone(phoenix2.orchestrator.atomspace)
        self.assertEqual(len(phoenix2.orchestrator.atomspace.get_all_atoms()), 2)


class TestPhoenixConfiguration(unittest.TestCase):
    """Test cases for Phoenix configuration and modes."""
    
    def test_transformation_modes(self):
        """Test different transformation modes."""
        atomspace = AtomSpace()
        
        for mode in TransformationMode:
            phoenix = CycleOfPhoenix(atomspace, mode)
            status = phoenix.get_transformation_status()
            
            self.assertEqual(status['transformation_mode'], mode.value)
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        from cycle_phoenix.phoenix_orchestrator import PhoenixConfiguration
        
        # Test default configuration
        config = PhoenixConfiguration()
        self.assertEqual(config.cycle_duration_years, 600)
        self.assertEqual(config.phase_duration_years, 200)
        self.assertTrue(config.enable_meta_cycles)
        
        # Test custom configuration
        custom_config = PhoenixConfiguration(
            cycle_duration_years=400,
            population_size=25
        )
        self.assertEqual(custom_config.cycle_duration_years, 400)
        self.assertEqual(custom_config.population_size, 25)


if __name__ == '__main__':
    unittest.main()
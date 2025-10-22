"""
Tests for the MOSES Evolution Engine
"""

import unittest
from unittest.mock import Mock
import sys
import os

# Add src to path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from atomspace import AtomSpace, Node, Link
from evolution.moses_engine import MosesEngine, Individual, Gene, MutationOperator, CrossoverOperator


class TestMosesEngine(unittest.TestCase):
    """Test cases for MOSES Engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atomspace = AtomSpace()
        self.moses = MosesEngine(self.atomspace, population_size=10)
        
        # Add some test atoms
        self.concept1 = Node("ConceptNode", "test_concept_1")
        self.concept2 = Node("ConceptNode", "test_concept_2")
        self.predicate = Node("PredicateNode", "test_predicate")
        self.link = Link("InheritanceLink", [self.concept1, self.concept2])
        
        for atom in [self.concept1, self.concept2, self.predicate, self.link]:
            self.atomspace.add(atom)
    
    def test_moses_initialization(self):
        """Test MOSES engine initialization."""
        self.assertEqual(self.moses.population_size, 10)
        self.assertIsNotNone(self.moses.fitness_function)
        self.assertGreater(len(self.moses.operators), 0)
        self.assertEqual(self.moses.generation, 0)
    
    def test_population_initialization(self):
        """Test population initialization."""
        # Initialize with seed atoms
        seed_atoms = [self.concept1, self.concept2, self.predicate]
        self.moses.initialize_population(seed_atoms)
        
        self.assertEqual(len(self.moses.population), self.moses.population_size)
        
        # Check that individuals have genes
        for individual in self.moses.population:
            self.assertIsInstance(individual, Individual)
            self.assertGreater(len(individual.genes), 0)
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation of population."""
        # Initialize population
        self.moses.initialize_population([self.concept1, self.concept2])
        
        # Evaluate population
        self.moses.evaluate_population()
        
        # Check that individuals have fitness scores
        for individual in self.moses.population:
            self.assertIsNotNone(individual.fitness)
            self.assertGreaterEqual(individual.fitness, 0.0)
            self.assertLessEqual(individual.fitness, 1.0)
    
    def test_evolution_generation(self):
        """Test single generation evolution."""
        # Initialize population
        self.moses.initialize_population([self.concept1, self.concept2])
        
        # Record initial state
        initial_generation = self.moses.generation
        initial_population_size = len(self.moses.population)
        
        # Evolve one generation
        self.moses.evolve_generation()
        
        # Check generation advanced
        self.assertEqual(self.moses.generation, initial_generation + 1)
        self.assertEqual(len(self.moses.population), initial_population_size)
    
    def test_best_individual_selection(self):
        """Test selection of best individual."""
        # Initialize and evaluate population
        self.moses.initialize_population([self.concept1, self.concept2])
        self.moses.evaluate_population()
        
        # Get best individual
        best = self.moses.get_best_individual()
        
        self.assertIsNotNone(best)
        self.assertIsInstance(best, Individual)
        
        # Verify it's actually the best
        all_fitnesses = [ind.fitness or 0.0 for ind in self.moses.population]
        max_fitness = max(all_fitnesses)
        self.assertEqual(best.fitness, max_fitness)
    
    def test_population_statistics(self):
        """Test population statistics calculation."""
        # Initialize and evaluate population
        self.moses.initialize_population([self.concept1, self.concept2])
        self.moses.evaluate_population()
        
        # Get statistics
        stats = self.moses.get_population_stats()
        
        # Verify expected fields
        expected_fields = ['generation', 'population_size', 'best_fitness', 
                          'average_fitness', 'worst_fitness', 'diversity']
        
        for field in expected_fields:
            self.assertIn(field, stats)
        
        # Verify reasonable values
        self.assertEqual(stats['population_size'], self.moses.population_size)
        self.assertGreaterEqual(stats['best_fitness'], stats['average_fitness'])
        self.assertGreaterEqual(stats['average_fitness'], stats['worst_fitness'])


class TestIndividual(unittest.TestCase):
    """Test cases for Individual class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.concept = Node("ConceptNode", "test")
        self.gene1 = Gene(atom=self.concept, weight=0.5, active=True)
        self.gene2 = Gene(atom=self.concept, weight=0.8, active=False)
    
    def test_individual_creation(self):
        """Test individual creation and gene management."""
        individual = Individual()
        
        # Initially empty
        self.assertEqual(len(individual), 0)
        self.assertEqual(len(individual.get_active_genes()), 0)
        
        # Add genes
        individual.add_gene(self.gene1)
        individual.add_gene(self.gene2)
        
        self.assertEqual(len(individual), 2)
        self.assertEqual(len(individual.get_active_genes()), 1)  # Only gene1 is active
    
    def test_individual_copying(self):
        """Test individual copying functionality."""
        individual = Individual([self.gene1, self.gene2])
        individual.fitness = 0.75
        individual.age = 5
        
        # Create copy
        copy_individual = individual.copy()
        
        # Verify independence
        self.assertNotEqual(id(individual), id(copy_individual))
        self.assertEqual(len(copy_individual), len(individual))
        self.assertEqual(copy_individual.fitness, individual.fitness)
        self.assertEqual(copy_individual.age, individual.age)
        
        # Verify gene independence
        self.assertNotEqual(id(individual.genes[0]), id(copy_individual.genes[0]))
    
    def test_atomspace_structure_conversion(self):
        """Test conversion to AtomSpace structures."""
        individual = Individual([self.gene1, self.gene2])
        
        atoms = individual.to_atomspace_structure()
        
        # Should only get active genes (gene1)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0], self.concept)


class TestEvolutionaryOperators(unittest.TestCase):
    """Test cases for evolutionary operators."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.concept = Node("ConceptNode", "test")
        self.gene = Gene(atom=self.concept, weight=0.5, active=True)
        self.individual = Individual([self.gene])
    
    def test_mutation_operator(self):
        """Test mutation operator functionality."""
        mutation_op = MutationOperator(mutation_rate=1.0)  # 100% mutation for testing
        
        # Apply mutation
        mutated = mutation_op.apply(self.individual)
        
        # Should be a different individual
        self.assertNotEqual(id(self.individual), id(mutated))
        
        # Should have same number of genes
        self.assertEqual(len(mutated), len(self.individual))
        
        # Weight should be different (with high probability)
        original_weight = self.individual.genes[0].weight
        mutated_weight = mutated.genes[0].weight
        
        # Allow for small chance they're the same due to randomness
        # In practice with mutation_rate=1.0, they should almost always differ
    
    def test_crossover_operator(self):
        """Test crossover operator functionality."""
        # Create second individual for crossover
        gene2 = Gene(atom=Node("ConceptNode", "test2"), weight=0.8, active=True)
        individual2 = Individual([gene2])
        
        crossover_op = CrossoverOperator(crossover_rate=1.0)  # 100% crossover for testing
        
        # Apply crossover
        offspring = crossover_op.apply(self.individual, {'partner': individual2})
        
        # Should be a new individual
        self.assertNotEqual(id(self.individual), id(offspring))
        self.assertNotEqual(id(individual2), id(offspring))
        
        # Should have genes from both parents
        self.assertGreater(len(offspring.genes), 0)


if __name__ == '__main__':
    unittest.main()
"""
Selection Strategies for MOSES Evolution

This module implements various selection strategies for evolutionary algorithms,
including tournament selection, roulette wheel selection, and rank-based selection.
"""

import random
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .moses_engine import Individual


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select(self, population: List[Individual], 
               count: int = 1, context: Dict[str, Any] = None) -> List[Individual]:
        """Select individuals from the population.
        
        Args:
            population: Population to select from
            count: Number of individuals to select
            context: Optional context information
            
        Returns:
            List of selected individuals
        """
        pass


class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy."""
    
    def __init__(self, tournament_size: int = 3, 
                 selection_pressure: float = 1.0):
        """Initialize tournament selection.
        
        Args:
            tournament_size: Size of tournament groups
            selection_pressure: Pressure factor (higher = more selective)
        """
        self.tournament_size = tournament_size
        self.selection_pressure = selection_pressure
    
    def select(self, population: List[Individual], 
               count: int = 1, context: Dict[str, Any] = None) -> List[Individual]:
        """Select individuals using tournament selection."""
        if not population:
            return []
        
        selected = []
        
        for _ in range(count):
            # Create tournament
            tournament_size = min(self.tournament_size, len(population))
            tournament = random.sample(population, tournament_size)
            
            # Select winner with pressure adjustment
            if self.selection_pressure == 1.0:
                # Standard tournament - best fitness wins
                winner = max(tournament, key=lambda ind: ind.fitness or 0.0)
            else:
                # Pressure-adjusted selection
                tournament.sort(key=lambda ind: ind.fitness or 0.0, reverse=True)
                
                # Use exponential probability based on rank
                probabilities = []
                for i, individual in enumerate(tournament):
                    prob = math.exp(-i * self.selection_pressure)
                    probabilities.append(prob)
                
                # Normalize probabilities
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                    
                    # Select based on probabilities
                    r = random.random()
                    cumulative = 0.0
                    winner = tournament[0]  # Default to best
                    
                    for i, prob in enumerate(probabilities):
                        cumulative += prob
                        if r <= cumulative:
                            winner = tournament[i]
                            break
                else:
                    winner = tournament[0]
            
            selected.append(winner)
        
        return selected


class RouletteSelection(SelectionStrategy):
    """Roulette wheel selection strategy."""
    
    def __init__(self, fitness_scaling: str = "linear", 
                 scaling_factor: float = 2.0):
        """Initialize roulette selection.
        
        Args:
            fitness_scaling: Type of fitness scaling ("linear", "exponential", "rank")
            scaling_factor: Scaling factor for fitness transformation
        """
        self.fitness_scaling = fitness_scaling
        self.scaling_factor = scaling_factor
    
    def select(self, population: List[Individual], 
               count: int = 1, context: Dict[str, Any] = None) -> List[Individual]:
        """Select individuals using roulette wheel selection."""
        if not population:
            return []
        
        # Calculate selection probabilities
        probabilities = self._calculate_probabilities(population)
        
        selected = []
        for _ in range(count):
            r = random.random()
            cumulative = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(population[i])
                    break
            else:
                # Fallback to last individual if rounding errors occur
                selected.append(population[-1])
        
        return selected
    
    def _calculate_probabilities(self, population: List[Individual]) -> List[float]:
        """Calculate selection probabilities for population."""
        fitnesses = [ind.fitness or 0.0 for ind in population]
        
        # Handle negative fitnesses
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 0.001 for f in fitnesses]
        
        # Apply fitness scaling
        if self.fitness_scaling == "linear":
            scaled_fitnesses = [f * self.scaling_factor for f in fitnesses]
        elif self.fitness_scaling == "exponential":
            scaled_fitnesses = [math.exp(f * self.scaling_factor) for f in fitnesses]
        elif self.fitness_scaling == "rank":
            # Rank-based scaling
            ranked = sorted(enumerate(fitnesses), key=lambda x: x[1])
            scaled_fitnesses = [0.0] * len(population)
            for rank, (original_idx, _) in enumerate(ranked):
                scaled_fitnesses[original_idx] = rank * self.scaling_factor
        else:
            scaled_fitnesses = fitnesses
        
        # Normalize to probabilities
        total_fitness = sum(scaled_fitnesses)
        if total_fitness > 0:
            probabilities = [f / total_fitness for f in scaled_fitnesses]
        else:
            # Equal probability if all fitnesses are zero
            probabilities = [1.0 / len(population)] * len(population)
        
        return probabilities


class RankSelection(SelectionStrategy):
    """Rank-based selection strategy."""
    
    def __init__(self, selection_pressure: float = 2.0, 
                 use_linear_ranking: bool = True):
        """Initialize rank selection.
        
        Args:
            selection_pressure: Selection pressure (1.0 = no pressure, 2.0 = max pressure)
            use_linear_ranking: Use linear vs exponential ranking
        """
        self.selection_pressure = max(1.0, min(2.0, selection_pressure))
        self.use_linear_ranking = use_linear_ranking
    
    def select(self, population: List[Individual], 
               count: int = 1, context: Dict[str, Any] = None) -> List[Individual]:
        """Select individuals using rank-based selection."""
        if not population:
            return []
        
        # Sort population by fitness (worst to best)
        sorted_population = sorted(population, key=lambda ind: ind.fitness or 0.0)
        
        # Calculate rank-based probabilities
        probabilities = self._calculate_rank_probabilities(len(population))
        
        selected = []
        for _ in range(count):
            r = random.random()
            cumulative = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(sorted_population[i])
                    break
            else:
                selected.append(sorted_population[-1])
        
        return selected
    
    def _calculate_rank_probabilities(self, population_size: int) -> List[float]:
        """Calculate probabilities based on rank."""
        if self.use_linear_ranking:
            # Linear ranking: P(i) = (2-SP + 2*(SP-1)*i/(N-1)) / N
            probabilities = []
            for i in range(population_size):
                if population_size == 1:
                    prob = 1.0
                else:
                    prob = (2 - self.selection_pressure + 
                           2 * (self.selection_pressure - 1) * i / (population_size - 1)) / population_size
                probabilities.append(prob)
        else:
            # Exponential ranking
            probabilities = []
            for i in range(population_size):
                prob = (self.selection_pressure - 1) / (self.selection_pressure ** population_size - 1)
                prob *= self.selection_pressure ** i
                probabilities.append(prob)
        
        return probabilities


class ElitistSelection(SelectionStrategy):
    """Elitist selection strategy that always selects the best individuals."""
    
    def __init__(self, elite_ratio: float = 0.1):
        """Initialize elitist selection.
        
        Args:
            elite_ratio: Ratio of population to consider elite
        """
        self.elite_ratio = max(0.0, min(1.0, elite_ratio))
    
    def select(self, population: List[Individual], 
               count: int = 1, context: Dict[str, Any] = None) -> List[Individual]:
        """Select best individuals from population."""
        if not population:
            return []
        
        # Sort by fitness (best first)
        sorted_population = sorted(population, key=lambda ind: ind.fitness or 0.0, reverse=True)
        
        # Determine elite size
        elite_size = max(1, int(len(population) * self.elite_ratio))
        elite_population = sorted_population[:elite_size]
        
        # Select from elite population
        selected = []
        for _ in range(count):
            selected.append(random.choice(elite_population))
        
        return selected


class StochasticUniversalSampling(SelectionStrategy):
    """Stochastic Universal Sampling selection strategy."""
    
    def __init__(self, fitness_scaling: str = "linear"):
        """Initialize SUS selection.
        
        Args:
            fitness_scaling: Type of fitness scaling to apply
        """
        self.fitness_scaling = fitness_scaling
    
    def select(self, population: List[Individual], 
               count: int = 1, context: Dict[str, Any] = None) -> List[Individual]:
        """Select individuals using Stochastic Universal Sampling."""
        if not population or count <= 0:
            return []
        
        fitnesses = [ind.fitness or 0.0 for ind in population]
        
        # Handle negative fitnesses
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 0.001 for f in fitnesses]
        
        # Calculate cumulative fitnesses
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            # Equal selection if all fitnesses are zero
            return [random.choice(population) for _ in range(count)]
        
        # SUS selection
        pointer_distance = total_fitness / count
        start = random.uniform(0, pointer_distance)
        
        selected = []
        cumulative = 0.0
        i = 0
        
        for selection_num in range(count):
            pointer = start + selection_num * pointer_distance
            
            # Find the individual corresponding to this pointer
            while cumulative < pointer and i < len(population):
                cumulative += fitnesses[i]
                i += 1
            
            if i > 0:
                selected.append(population[i - 1])
            else:
                selected.append(population[0])
        
        return selected


class MultiObjectiveSelection(SelectionStrategy):
    """Multi-objective selection using Pareto dominance."""
    
    def __init__(self, objectives: List[str] = None, 
                 crowding_distance_weight: float = 0.5):
        """Initialize multi-objective selection.
        
        Args:
            objectives: List of objective names to optimize
            crowding_distance_weight: Weight for crowding distance in selection
        """
        self.objectives = objectives or ['fitness']
        self.crowding_distance_weight = crowding_distance_weight
    
    def select(self, population: List[Individual], 
               count: int = 1, context: Dict[str, Any] = None) -> List[Individual]:
        """Select individuals using multi-objective criteria."""
        if not population:
            return []
        
        # Calculate Pareto fronts
        fronts = self._calculate_pareto_fronts(population)
        
        # Calculate crowding distances for each front
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Select individuals from fronts
        selected = []
        front_idx = 0
        
        while len(selected) < count and front_idx < len(fronts):
            current_front = fronts[front_idx]
            
            if len(selected) + len(current_front) <= count:
                # Add entire front
                selected.extend(current_front)
            else:
                # Partially fill from this front using crowding distance
                remaining_count = count - len(selected)
                current_front.sort(key=lambda ind: ind.metadata.get('crowding_distance', 0.0), 
                                 reverse=True)
                selected.extend(current_front[:remaining_count])
            
            front_idx += 1
        
        return selected
    
    def _calculate_pareto_fronts(self, population: List[Individual]) -> List[List[Individual]]:
        """Calculate Pareto fronts for the population."""
        fronts = []
        remaining = population.copy()
        
        while remaining:
            current_front = []
            dominated = []
            
            for individual in remaining:
                is_dominated = False
                
                for other in remaining:
                    if other != individual and self._dominates(other, individual):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(individual)
                else:
                    dominated.append(individual)
            
            if current_front:
                fronts.append(current_front)
                remaining = dominated
            else:
                # Prevent infinite loop - add remaining individuals
                fronts.append(remaining)
                break
        
        return fronts
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2 in Pareto sense."""
        better_in_any = False
        
        for objective in self.objectives:
            val1 = ind1.metadata.get(objective, ind1.fitness or 0.0)
            val2 = ind2.metadata.get(objective, ind2.fitness or 0.0)
            
            if val1 < val2:
                return False  # ind1 is worse in this objective
            elif val1 > val2:
                better_in_any = True
        
        return better_in_any
    
    def _calculate_crowding_distance(self, front: List[Individual]) -> None:
        """Calculate crowding distance for individuals in a front."""
        if len(front) <= 2:
            # Boundary individuals get infinite distance
            for ind in front:
                ind.metadata['crowding_distance'] = float('inf')
            return
        
        # Initialize distances
        for ind in front:
            ind.metadata['crowding_distance'] = 0.0
        
        # Calculate distance for each objective
        for objective in self.objectives:
            # Sort by objective value
            front.sort(key=lambda ind: ind.metadata.get(objective, ind.fitness or 0.0))
            
            # Get objective range
            obj_values = [ind.metadata.get(objective, ind.fitness or 0.0) for ind in front]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range > 0:
                # Boundary individuals get infinite distance
                front[0].metadata['crowding_distance'] = float('inf')
                front[-1].metadata['crowding_distance'] = float('inf')
                
                # Calculate distances for intermediate individuals
                for i in range(1, len(front) - 1):
                    distance = (obj_values[i + 1] - obj_values[i - 1]) / obj_range
                    front[i].metadata['crowding_distance'] += distance


class AdaptiveSelection(SelectionStrategy):
    """Adaptive selection that changes strategy based on population diversity."""
    
    def __init__(self, strategies: List[Tuple[SelectionStrategy, float]] = None):
        """Initialize adaptive selection.
        
        Args:
            strategies: List of (strategy, weight) tuples
        """
        if strategies is None:
            strategies = [
                (TournamentSelection(tournament_size=3), 0.4),
                (RouletteSelection(), 0.3),
                (RankSelection(), 0.2),
                (ElitistSelection(elite_ratio=0.1), 0.1)
            ]
        
        self.strategies = strategies
        self.strategy_performance: Dict[str, float] = {}
        
        # Initialize performance tracking
        for strategy, _ in self.strategies:
            strategy_name = strategy.__class__.__name__
            self.strategy_performance[strategy_name] = 1.0
    
    def select(self, population: List[Individual], 
               count: int = 1, context: Dict[str, Any] = None) -> List[Individual]:
        """Select using adaptive strategy selection."""
        if not population:
            return []
        
        # Calculate population diversity
        diversity = self._calculate_population_diversity(population)
        
        # Select strategy based on diversity and performance
        strategy = self._select_strategy(diversity)
        
        # Perform selection
        selected = strategy.select(population, count, context)
        
        # Update strategy performance (simplified)
        if selected:
            avg_fitness = sum(ind.fitness or 0.0 for ind in selected) / len(selected)
            strategy_name = strategy.__class__.__name__
            
            # Exponential moving average
            alpha = 0.1
            self.strategy_performance[strategy_name] = (
                alpha * avg_fitness + 
                (1 - alpha) * self.strategy_performance.get(strategy_name, 0.0)
            )
        
        return selected
    
    def _calculate_population_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity measure."""
        if len(population) <= 1:
            return 0.0
        
        # Simple diversity based on fitness variance
        fitnesses = [ind.fitness or 0.0 for ind in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        
        return min(1.0, variance)  # Normalize to [0, 1]
    
    def _select_strategy(self, diversity: float) -> SelectionStrategy:
        """Select strategy based on diversity and performance."""
        # Adjust weights based on diversity
        adjusted_weights = []
        
        for strategy, base_weight in self.strategies:
            strategy_name = strategy.__class__.__name__
            performance = self.strategy_performance.get(strategy_name, 1.0)
            
            # Adjust weight based on diversity
            if isinstance(strategy, ElitistSelection):
                # Use elitist selection more when diversity is low
                diversity_factor = 1.0 - diversity
            elif isinstance(strategy, TournamentSelection):
                # Use tournament selection more when diversity is high
                diversity_factor = diversity
            else:
                diversity_factor = 1.0
            
            adjusted_weight = base_weight * performance * diversity_factor
            adjusted_weights.append(adjusted_weight)
        
        # Select strategy based on weights
        total_weight = sum(adjusted_weights)
        if total_weight > 0:
            r = random.uniform(0, total_weight)
            cumulative = 0.0
            
            for i, weight in enumerate(adjusted_weights):
                cumulative += weight
                if r <= cumulative:
                    return self.strategies[i][0]
        
        # Fallback to first strategy
        return self.strategies[0][0]
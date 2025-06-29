from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import numpy as np
import networkx as nx

class NetworkType(Enum):
    """Types of neural networks as described by Vervaeke"""
    REGULAR = "regular"      # Short-distance connections
    RANDOM = "random"        # Mixed connections
    SMALL_WORLD = "small_world"  # Optimal balance

@dataclass
class NetworkMetrics:
    """Metrics for analyzing network properties"""
    mean_path_length: float  # Efficiency measure
    clustering_coeff: float  # Resiliency measure
    small_world_index: float  # Balance measure

class NeuralDynamics:
    """Handles neural dynamics of relevance realization.
    
    Implements Vervaeke's model of neural dynamics based on:
    - Self-Organizing Criticality (SOC) in firing patterns
    - Small World Networks (SWN) in wiring patterns
    """
    
    def __init__(self, n_nodes: int = 100):
        self.n_nodes = n_nodes
        self.network = self._init_network()
        self.firing_states = np.zeros(n_nodes)
        self.sync_threshold = 0.7
        
    def update_dynamics(self, 
                       input_pattern: np.ndarray,
                       context: Optional[Dict] = None) -> Tuple[np.ndarray, NetworkMetrics]:
        """Update neural dynamics based on input and context.
        
        Args:
            input_pattern: Activity pattern to process
            context: Optional contextual information
            
        Returns:
            Tuple of (output pattern, network metrics)
        """
        # Update firing patterns (SOC)
        output = self._update_firing(input_pattern, context)
        
        # Update network structure (SWN)
        self._update_wiring(output, context)
        
        # Compute network metrics
        metrics = self._compute_metrics()
        
        return output, metrics
        
    def _init_network(self) -> nx.Graph:
        """Initialize network with regular structure."""
        # Start with regular network (k=4 nearest neighbors)
        k = 4
        network = nx.watts_strogatz_graph(
            n=self.n_nodes,
            k=k,
            p=0  # No rewiring initially
        )
        return network
        
    def _update_firing(self, 
                      input_pattern: np.ndarray,
                      context: Optional[Dict]) -> np.ndarray:
        """Update firing patterns using SOC dynamics."""
        # Current firing states
        states = self.firing_states
        
        # Add input influence
        states = 0.3 * states + 0.7 * input_pattern
        
        # Apply thresholding for avalanche dynamics
        over_threshold = states > self.sync_threshold
        if np.mean(over_threshold) > 0.6:  # Critical state
            # Neural avalanche
            states[over_threshold] *= 0.5
            # Reorganize remaining activity
            states[~over_threshold] += np.mean(states[over_threshold]) * 0.1
            
        # Update firing states
        self.firing_states = states
        
        return states
        
    def _update_wiring(self,
                      activity_pattern: np.ndarray,
                      context: Optional[Dict]) -> None:
        """Update network structure based on activity."""
        # Get current network stats
        metrics = self._compute_metrics()
        
        # If too regular, add some long-range connections
        if metrics.small_world_index < 0.8:
            self._add_long_range_connections()
            
        # If too random, strengthen local clustering
        elif metrics.small_world_index > 1.2:
            self._strengthen_local_clusters()
            
    def _compute_metrics(self) -> NetworkMetrics:
        """Compute current network metrics."""
        # Get base metrics
        mpl = nx.average_shortest_path_length(self.network)
        cc = nx.average_clustering(self.network)
        
        # Generate equivalent random graph
        random_graph = nx.erdos_renyi_graph(
            n=self.n_nodes,
            p=nx.density(self.network)
        )
        
        # Compute small-world index
        random_mpl = nx.average_shortest_path_length(random_graph)
        random_cc = nx.average_clustering(random_graph)
        
        small_world = (cc/random_cc) / (mpl/random_mpl)
        
        return NetworkMetrics(
            mean_path_length=mpl,
            clustering_coeff=cc,
            small_world_index=small_world
        )
        
    def _add_long_range_connections(self) -> None:
        """Add some long-range connections to increase efficiency."""
        # Randomly select nodes
        nodes = list(self.network.nodes())
        n_new = int(0.1 * self.n_nodes)  # Add 10% new connections
        
        for _ in range(n_new):
            source = np.random.choice(nodes)
            # Select distant target
            distances = [
                nx.shortest_path_length(self.network, source, target)
                for target in nodes
            ]
            distant = [
                n for n, d in enumerate(distances)
                if d > np.mean(distances)
            ]
            if distant:
                target = np.random.choice(distant)
                self.network.add_edge(source, target)
                
    def _strengthen_local_clusters(self) -> None:
        """Strengthen local clustering to increase resiliency."""
        # For each node
        for node in self.network.nodes():
            neighbors = list(self.network.neighbors(node))
            if len(neighbors) >= 2:
                # Connect random pair of neighbors
                pair = np.random.choice(
                    neighbors, size=2, replace=False
                )
                self.network.add_edge(pair[0], pair[1])
                
    def get_network_type(self) -> NetworkType:
        """Determine current network type based on metrics."""
        metrics = self._compute_metrics()
        
        if metrics.small_world_index < 0.8:
            return NetworkType.REGULAR
        elif metrics.small_world_index > 1.2:
            return NetworkType.RANDOM
        else:
            return NetworkType.SMALL_WORLD 
"""
Cross-Domain Knowledge Graph Integration

This module implements knowledge graph structures and algorithms for integrating
information from multiple domains and modalities into a coherent graph representation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import torch
import networkx as nx


class RelationType(Enum):
    """Types of relationships in the knowledge graph"""
    IS_A = "is_a"  # Taxonomic
    PART_OF = "part_of"  # Mereological
    CAUSES = "causes"  # Causal
    SIMILAR_TO = "similar_to"  # Analogical
    TRANSFORMS_TO = "transforms_to"  # Dynamic
    REPRESENTS = "represents"  # Semantic
    INSTANTIATES = "instantiates"  # Abstract-Concrete
    CROSS_MODAL = "cross_modal"  # Cross-modality link


@dataclass
class KnowledgeNode:
    """
    Node in the cross-domain knowledge graph.
    
    Represents a concept, entity, or pattern that may have instantiations
    across multiple domains.
    """
    node_id: str
    content: Any
    domain: 'Domain'
    embedding: torch.Tensor
    node_type: str = "concept"
    properties: Dict[str, Any] = field(default_factory=dict)
    activation: float = 1.0
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, KnowledgeNode):
            return False
        return self.node_id == other.node_id


@dataclass
class KnowledgeEdge:
    """
    Edge in the cross-domain knowledge graph.
    
    Represents relationships and connections between concepts across domains.
    """
    source: KnowledgeNode
    target: KnowledgeNode
    relation_type: RelationType
    strength: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False
    
    def __hash__(self):
        return hash((self.source.node_id, self.target.node_id, self.relation_type))


class CrossDomainKnowledgeGraph:
    """
    Knowledge graph that integrates information from multiple cognitive domains.
    
    Provides graph-based reasoning, querying, and knowledge retrieval that
    spans domain boundaries.
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.domain_subgraphs: Dict['Domain', nx.DiGraph] = {}
        
    def add_node(self, node: KnowledgeNode):
        """Add a node to the knowledge graph"""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            **node.properties,
            domain=node.domain.value,
            node_type=node.node_type,
            activation=node.activation
        )
        
        # Update domain subgraph
        if node.domain not in self.domain_subgraphs:
            self.domain_subgraphs[node.domain] = nx.DiGraph()
        self.domain_subgraphs[node.domain].add_node(node.node_id)
    
    def add_edge(self, edge: KnowledgeEdge):
        """Add an edge to the knowledge graph"""
        self.edges.append(edge)
        
        # Add both nodes if not present
        if edge.source.node_id not in self.nodes:
            self.add_node(edge.source)
        if edge.target.node_id not in self.nodes:
            self.add_node(edge.target)
        
        # Add edge to graph
        self.graph.add_edge(
            edge.source.node_id,
            edge.target.node_id,
            relation=edge.relation_type.value,
            strength=edge.strength,
            **edge.properties
        )
        
        # Add to domain subgraphs
        if edge.source.domain == edge.target.domain:
            # Intra-domain edge
            if edge.source.domain in self.domain_subgraphs:
                self.domain_subgraphs[edge.source.domain].add_edge(
                    edge.source.node_id,
                    edge.target.node_id
                )
        
        # Add reverse edge if bidirectional
        if edge.bidirectional:
            self.graph.add_edge(
                edge.target.node_id,
                edge.source.node_id,
                relation=edge.relation_type.value,
                strength=edge.strength,
                **edge.properties
            )
    
    def find_cross_domain_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """
        Find paths connecting nodes across domains.
        
        Args:
            source_id: Starting node ID
            target_id: Target node ID
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_domain_neighborhood(
        self,
        node_id: str,
        target_domain: 'Domain',
        radius: int = 2
    ) -> Set[str]:
        """
        Get nodes in target domain within graph distance radius.
        
        Args:
            node_id: Center node ID
            target_domain: Domain to search in
            radius: Maximum graph distance
            
        Returns:
            Set of node IDs in target domain
        """
        if node_id not in self.nodes:
            return set()
        
        # BFS to find nodes within radius
        visited = {node_id}
        current_level = {node_id}
        result = set()
        
        for _ in range(radius):
            next_level = set()
            for node in current_level:
                # Get neighbors
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        
                        # Check if in target domain
                        if self.nodes[neighbor].domain == target_domain:
                            result.add(neighbor)
                
                # Also check predecessors
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        
                        if self.nodes[neighbor].domain == target_domain:
                            result.add(neighbor)
            
            current_level = next_level
            if not current_level:
                break
        
        return result
    
    def activate_pattern(
        self,
        seed_nodes: List[str],
        activation_threshold: float = 0.1,
        decay: float = 0.8,
        max_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Spread activation through the graph from seed nodes.
        
        Args:
            seed_nodes: Initial nodes to activate
            activation_threshold: Minimum activation to consider
            decay: Activation decay factor per hop
            max_iterations: Maximum spreading iterations
            
        Returns:
            Dictionary mapping node IDs to activation values
        """
        activations = {node_id: 0.0 for node_id in self.nodes}
        
        # Initialize seed nodes
        for node_id in seed_nodes:
            if node_id in activations:
                activations[node_id] = 1.0
        
        # Spread activation
        for _ in range(max_iterations):
            new_activations = activations.copy()
            
            for node_id, activation in activations.items():
                if activation < activation_threshold:
                    continue
                
                # Spread to neighbors
                for neighbor in self.graph.successors(node_id):
                    edge_data = self.graph.get_edge_data(node_id, neighbor)
                    # Get first edge if multiple
                    if isinstance(edge_data, dict):
                        edge_strength = list(edge_data.values())[0].get('strength', 0.5)
                    else:
                        edge_strength = 0.5
                    
                    spread_amount = activation * decay * edge_strength
                    new_activations[neighbor] = max(
                        new_activations[neighbor],
                        spread_amount
                    )
            
            activations = new_activations
        
        # Filter by threshold
        return {
            node_id: act for node_id, act in activations.items()
            if act >= activation_threshold
        }
    
    def query_cross_domain(
        self,
        query_embedding: torch.Tensor,
        source_domain: 'Domain',
        target_domains: Optional[List['Domain']] = None,
        top_k: int = 5
    ) -> List[Tuple[KnowledgeNode, float]]:
        """
        Query knowledge graph across domains using embedding similarity.
        
        Args:
            query_embedding: Query vector
            source_domain: Domain of the query
            target_domains: Domains to search (None = all domains)
            top_k: Number of results to return
            
        Returns:
            List of (node, similarity) tuples
        """
        if target_domains is None:
            target_domains = list(self.domain_subgraphs.keys())
        
        candidates = []
        for domain in target_domains:
            if domain == source_domain:
                continue  # Skip same domain
            
            # Get nodes in target domain
            for node_id in self.domain_subgraphs.get(domain, []):
                node = self.nodes[node_id]
                
                # Compute similarity
                similarity = torch.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    node.embedding.unsqueeze(0)
                ).item()
                
                candidates.append((node, similarity))
        
        # Sort by similarity and return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def extract_subgraph(
        self,
        node_ids: List[str],
        include_connecting: bool = True
    ) -> 'CrossDomainKnowledgeGraph':
        """
        Extract subgraph containing specified nodes.
        
        Args:
            node_ids: Nodes to include
            include_connecting: Whether to include paths connecting the nodes
            
        Returns:
            New knowledge graph with the subgraph
        """
        subgraph = CrossDomainKnowledgeGraph()
        
        if include_connecting:
            # Find connecting nodes
            connecting_nodes = set(node_ids)
            for i, source in enumerate(node_ids):
                for target in node_ids[i+1:]:
                    paths = self.find_cross_domain_paths(source, target, max_length=3)
                    for path in paths:
                        connecting_nodes.update(path)
            node_ids = list(connecting_nodes)
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        # Add edges between included nodes
        for edge in self.edges:
            if (edge.source.node_id in node_ids and 
                edge.target.node_id in node_ids):
                subgraph.add_edge(edge)
        
        return subgraph
    
    def compute_centrality(self, domain: Optional['Domain'] = None) -> Dict[str, float]:
        """
        Compute node centrality measures.
        
        Args:
            domain: Optional domain to restrict computation
            
        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        if domain and domain in self.domain_subgraphs:
            graph = self.domain_subgraphs[domain]
        else:
            graph = self.graph
        
        # Use PageRank as centrality measure
        centrality = nx.pagerank(graph, alpha=0.85)
        return centrality
    
    def detect_communities(
        self,
        resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Detect communities/clusters in the knowledge graph.
        
        Args:
            resolution: Resolution parameter for community detection
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Use Louvain method
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.louvain_communities(undirected, resolution=resolution)
        
        # Create mapping
        node_to_community = {}
        for comm_id, community in enumerate(communities):
            for node_id in community:
                node_to_community[node_id] = comm_id
        
        return node_to_community


class KnowledgeGraphIntegrator:
    """
    Integrates multiple knowledge sources into unified cross-domain graph.
    
    Handles merging, deduplication, and conflict resolution when integrating
    knowledge from different modalities and domains.
    """
    
    def __init__(self):
        self.main_graph = CrossDomainKnowledgeGraph()
        self.source_graphs: Dict[str, CrossDomainKnowledgeGraph] = {}
        
    def add_source(self, source_id: str, graph: CrossDomainKnowledgeGraph):
        """Add a knowledge source"""
        self.source_graphs[source_id] = graph
    
    def merge_sources(
        self,
        similarity_threshold: float = 0.85,
        conflict_resolution: str = "average"
    ):
        """
        Merge all source graphs into main integrated graph.
        
        Args:
            similarity_threshold: Threshold for considering nodes equivalent
            conflict_resolution: Strategy for resolving conflicts ("average", "vote", "latest")
        """
        from .cross_domain_core import Domain
        
        # Node deduplication mapping
        node_mapping: Dict[Tuple[str, str], str] = {}  # (source_id, node_id) -> merged_id
        
        # Process each source graph
        for source_id, source_graph in self.source_graphs.items():
            for node_id, node in source_graph.nodes.items():
                # Check for equivalent nodes in main graph
                merged_id = None
                for existing_id, existing_node in self.main_graph.nodes.items():
                    if existing_node.domain == node.domain:
                        # Compute similarity
                        sim = torch.cosine_similarity(
                            node.embedding.unsqueeze(0),
                            existing_node.embedding.unsqueeze(0)
                        ).item()
                        
                        if sim >= similarity_threshold:
                            merged_id = existing_id
                            break
                
                if merged_id is None:
                    # Add as new node
                    merged_id = f"{source_id}_{node_id}"
                    new_node = KnowledgeNode(
                        node_id=merged_id,
                        content=node.content,
                        domain=node.domain,
                        embedding=node.embedding,
                        node_type=node.node_type,
                        properties={**node.properties, 'source': source_id}
                    )
                    self.main_graph.add_node(new_node)
                
                # Record mapping
                node_mapping[(source_id, node_id)] = merged_id
        
        # Merge edges
        for source_id, source_graph in self.source_graphs.items():
            for edge in source_graph.edges:
                source_key = (source_id, edge.source.node_id)
                target_key = (source_id, edge.target.node_id)
                
                if source_key in node_mapping and target_key in node_mapping:
                    merged_source = self.main_graph.nodes[node_mapping[source_key]]
                    merged_target = self.main_graph.nodes[node_mapping[target_key]]
                    
                    merged_edge = KnowledgeEdge(
                        source=merged_source,
                        target=merged_target,
                        relation_type=edge.relation_type,
                        strength=edge.strength,
                        properties={**edge.properties, 'source': source_id},
                        bidirectional=edge.bidirectional
                    )
                    self.main_graph.add_edge(merged_edge)
    
    def validate_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency of integrated knowledge graph.
        
        Returns:
            Dictionary of consistency metrics
        """
        metrics = {
            'num_nodes': len(self.main_graph.nodes),
            'num_edges': len(self.main_graph.edges),
            'num_domains': len(self.main_graph.domain_subgraphs),
            'is_connected': nx.is_weakly_connected(self.main_graph.graph),
            'num_components': nx.number_weakly_connected_components(self.main_graph.graph)
        }
        
        # Check for cycles (potential inconsistencies)
        try:
            cycles = list(nx.simple_cycles(self.main_graph.graph))
            metrics['num_cycles'] = len(cycles)
        except:
            metrics['num_cycles'] = 0
        
        # Compute density
        if len(self.main_graph.nodes) > 1:
            metrics['density'] = nx.density(self.main_graph.graph)
        else:
            metrics['density'] = 0.0
        
        return metrics

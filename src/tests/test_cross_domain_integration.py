"""
Tests for Cross-Domain Integration Framework

Tests unified representations, cross-modal attention, domain adaptation,
abstract concept mapping, cross-domain reasoning, and knowledge graph integration.
"""

import sys
import os
import pytest
import torch
import numpy as np

# Add src to path to avoid circular import issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from integration.cross_domain_core import (
    Domain, ModalityType, UnifiedRepresentation, CrossDomainMapping,
    CrossModalAttention, DomainAdaptation, AbstractConcept,
    CrossDomainReasoner, CrossDomainIntegrationFramework
)
from integration.cross_domain_knowledge_graph import (
    RelationType, KnowledgeNode, KnowledgeEdge,
    CrossDomainKnowledgeGraph, KnowledgeGraphIntegrator
)


class TestUnifiedRepresentation:
    """Test unified representation framework"""
    
    def test_representation_creation(self):
        """Test creating unified representations"""
        features = torch.randn(256)
        abstract_features = torch.randn(256)
        semantic_embedding = torch.randn(256)
        
        rep = UnifiedRepresentation(
            domain=Domain.VISUAL,
            features=features,
            abstract_features=abstract_features,
            semantic_embedding=semantic_embedding
        )
        
        assert rep.domain == Domain.VISUAL
        assert rep.features.shape == (256,)
        assert rep.abstract_features.shape == (256,)
        assert rep.semantic_embedding.shape == (256,)
    
    def test_representation_similarity(self):
        """Test similarity computation between representations"""
        # Create similar representations
        base_features = torch.randn(256)
        abstract1 = torch.randn(256)
        abstract2 = abstract1 + torch.randn(256) * 0.1  # Very similar
        
        rep1 = UnifiedRepresentation(
            domain=Domain.VISUAL,
            features=base_features,
            abstract_features=abstract1,
            semantic_embedding=abstract1
        )
        
        rep2 = UnifiedRepresentation(
            domain=Domain.AUDITORY,
            features=base_features,
            abstract_features=abstract2,
            semantic_embedding=abstract2
        )
        
        similarity = rep1.similarity(rep2)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.8  # Should be high for similar representations
    
    def test_cross_domain_links(self):
        """Test cross-domain linking"""
        rep = UnifiedRepresentation(
            domain=Domain.VISUAL,
            features=torch.randn(256),
            abstract_features=torch.randn(256),
            semantic_embedding=torch.randn(256),
            cross_domain_links={
                Domain.AUDITORY: 0.8,
                Domain.LINGUISTIC: 0.6
            }
        )
        
        assert len(rep.cross_domain_links) == 2
        assert Domain.AUDITORY in rep.cross_domain_links
        assert rep.cross_domain_links[Domain.AUDITORY] == 0.8


class TestCrossDomainMapping:
    """Test cross-domain mapping and transfer"""
    
    def test_mapping_creation(self):
        """Test creating cross-domain mappings"""
        transformation = torch.eye(256)
        
        mapping = CrossDomainMapping(
            source_domain=Domain.VISUAL,
            target_domain=Domain.LINGUISTIC,
            source_concept="image",
            target_concept="description",
            mapping_strength=0.9,
            transformation_matrix=transformation,
            bidirectional=True
        )
        
        assert mapping.source_domain == Domain.VISUAL
        assert mapping.target_domain == Domain.LINGUISTIC
        assert mapping.bidirectional
    
    def test_mapping_application(self):
        """Test applying mapping to transform representations"""
        features = torch.randn(256)
        abstract_features = torch.randn(256)
        
        rep = UnifiedRepresentation(
            domain=Domain.VISUAL,
            features=features,
            abstract_features=abstract_features,
            semantic_embedding=abstract_features
        )
        
        transformation = torch.randn(256, 256)
        mapping = CrossDomainMapping(
            source_domain=Domain.VISUAL,
            target_domain=Domain.LINGUISTIC,
            source_concept="image",
            target_concept="description",
            mapping_strength=0.9,
            transformation_matrix=transformation
        )
        
        transformed = mapping.apply(rep)
        
        assert transformed.domain == Domain.LINGUISTIC
        assert transformed.features.shape == features.shape
        # Abstract features should be preserved
        assert torch.allclose(transformed.abstract_features, abstract_features)
        # Should have link to source domain
        assert Domain.VISUAL in transformed.cross_domain_links


class TestCrossModalAttention:
    """Test cross-modal attention mechanisms"""
    
    def test_attention_initialization(self):
        """Test initializing cross-modal attention"""
        attention = CrossModalAttention(
            feature_dim=256,
            num_modalities=3,
            num_heads=4
        )
        
        assert attention.feature_dim == 256
        assert attention.num_modalities == 3
        assert attention.num_heads == 4
    
    def test_modality_integration(self):
        """Test integrating multiple modalities"""
        attention = CrossModalAttention(
            feature_dim=256,
            num_modalities=3,
            num_heads=4
        )
        
        # Create dummy modality features
        batch_size = 8
        modality_features = [
            torch.randn(batch_size, 256) for _ in range(3)
        ]
        
        integrated, weights = attention(modality_features)
        
        assert integrated.shape == (batch_size, 256)
        assert weights is not None
        # Check attention weights sum approximately to 1
        assert weights.shape[0] == batch_size


class TestDomainAdaptation:
    """Test domain adaptation mechanisms"""
    
    def test_adaptation_initialization(self):
        """Test initializing domain adaptation"""
        adaptation = DomainAdaptation(
            feature_dim=256,
            num_domains=len(Domain)
        )
        
        assert adaptation.feature_dim == 256
        assert adaptation.num_domains == len(Domain)
    
    def test_domain_encoding(self):
        """Test encoding features into domain-specific and invariant"""
        adaptation = DomainAdaptation(
            feature_dim=256,
            num_domains=len(Domain)
        )
        
        features = torch.randn(8, 256)
        domain_id = 0
        
        domain_specific, domain_invariant = adaptation.encode(features, domain_id)
        
        assert domain_specific.shape == features.shape
        assert domain_invariant.shape == features.shape
    
    def test_domain_alignment(self):
        """Test aligning features across domains"""
        adaptation = DomainAdaptation(
            feature_dim=256,
            num_domains=len(Domain)
        )
        
        source_features = torch.randn(8, 256)
        target_features = torch.randn(8, 256)
        
        results = adaptation.align_domains(
            source_features,
            target_features,
            source_domain=0,
            target_domain=1
        )
        
        assert 'alignment_loss' in results
        assert 'source_invariant' in results
        assert 'target_invariant' in results
        assert results['alignment_loss'].item() >= 0


class TestAbstractConcept:
    """Test abstract concept mapping"""
    
    def test_concept_creation(self):
        """Test creating abstract concepts"""
        abstract_features = torch.randn(256)
        
        concept = AbstractConcept(
            concept_id="test_concept",
            name="Test Concept",
            abstract_features=abstract_features,
            abstraction_level=2
        )
        
        assert concept.concept_id == "test_concept"
        assert concept.abstraction_level == 2
        assert len(concept.domain_instantiations) == 0
    
    def test_domain_instantiation(self):
        """Test instantiating concept in different domains"""
        abstract_features = torch.randn(256)
        
        concept = AbstractConcept(
            concept_id="movement",
            name="Movement",
            abstract_features=abstract_features
        )
        
        # Instantiate in visual domain
        visual_rep = concept.instantiate_in_domain(Domain.VISUAL)
        assert visual_rep.domain == Domain.VISUAL
        assert 'concept_id' in visual_rep.metadata
        
        # Instantiate in auditory domain
        auditory_rep = concept.instantiate_in_domain(Domain.AUDITORY)
        assert auditory_rep.domain == Domain.AUDITORY
        
        # Both should share abstract features (approximately)
        similarity = visual_rep.similarity(auditory_rep)
        assert similarity > 0.7  # Should be similar due to shared abstract features
    
    def test_conceptual_relationships(self):
        """Test adding relationships between concepts"""
        concept1 = AbstractConcept(
            concept_id="cause",
            name="Cause",
            abstract_features=torch.randn(256)
        )
        
        concept1.add_relationship("effect", 0.9)
        concept1.add_relationship("related", 0.5)
        
        assert len(concept1.conceptual_relationships) == 2
        assert concept1.conceptual_relationships[0] == ("effect", 0.9)


class TestCrossDomainReasoner:
    """Test cross-domain reasoning capabilities"""
    
    def test_reasoner_initialization(self):
        """Test initializing reasoner"""
        reasoner = CrossDomainReasoner(feature_dim=256)
        
        assert reasoner.feature_dim == 256
        assert len(reasoner.abstract_concepts) == 0
        assert len(reasoner.domain_mappings) == 0
    
    def test_concept_registration(self):
        """Test registering concepts"""
        reasoner = CrossDomainReasoner(feature_dim=256)
        
        concept = AbstractConcept(
            concept_id="test",
            name="Test",
            abstract_features=torch.randn(256)
        )
        
        reasoner.register_concept(concept)
        assert "test" in reasoner.abstract_concepts
    
    def test_mapping_registration(self):
        """Test registering mappings"""
        reasoner = CrossDomainReasoner(feature_dim=256)
        
        mapping = CrossDomainMapping(
            source_domain=Domain.VISUAL,
            target_domain=Domain.LINGUISTIC,
            source_concept="image",
            target_concept="description",
            mapping_strength=0.9,
            transformation_matrix=torch.eye(256),
            bidirectional=True
        )
        
        reasoner.register_mapping(mapping)
        
        key = (Domain.VISUAL, Domain.LINGUISTIC)
        assert key in reasoner.domain_mappings
        # Should also have reverse mapping
        reverse_key = (Domain.LINGUISTIC, Domain.VISUAL)
        assert reverse_key in reasoner.domain_mappings
    
    def test_knowledge_transfer(self):
        """Test transferring knowledge across domains"""
        reasoner = CrossDomainReasoner(feature_dim=256)
        
        # Create and register mapping
        mapping = CrossDomainMapping(
            source_domain=Domain.VISUAL,
            target_domain=Domain.LINGUISTIC,
            source_concept="image",
            target_concept="description",
            mapping_strength=0.9,
            transformation_matrix=torch.eye(256)
        )
        reasoner.register_mapping(mapping)
        
        # Create source representation
        rep = UnifiedRepresentation(
            domain=Domain.VISUAL,
            features=torch.randn(256),
            abstract_features=torch.randn(256),
            semantic_embedding=torch.randn(256)
        )
        
        # Transfer to linguistic domain
        transferred = reasoner.transfer_knowledge(rep, Domain.LINGUISTIC)
        
        assert transferred is not None
        assert transferred.domain == Domain.LINGUISTIC
        assert Domain.VISUAL in transferred.cross_domain_links
    
    def test_analogy_finding(self):
        """Test finding analogies across domains"""
        reasoner = CrossDomainReasoner(feature_dim=256)
        
        # Create source representation
        base_abstract = torch.randn(256)
        source = UnifiedRepresentation(
            domain=Domain.VISUAL,
            features=torch.randn(256),
            abstract_features=base_abstract,
            semantic_embedding=base_abstract
        )
        
        # Create candidates with varying similarity
        candidates = []
        for i, domain in enumerate([Domain.AUDITORY, Domain.LINGUISTIC, Domain.SPATIAL]):
            # Make some similar, some different
            if i == 0:
                abstract = base_abstract + torch.randn(256) * 0.1  # Very similar
            else:
                abstract = torch.randn(256)  # Different
            
            candidates.append(UnifiedRepresentation(
                domain=domain,
                features=torch.randn(256),
                abstract_features=abstract,
                semantic_embedding=abstract
            ))
        
        # Find analogies
        analogies = reasoner.find_analogies(source, candidates, top_k=2)
        
        assert len(analogies) == 2
        # Should be sorted by similarity
        assert analogies[0][1] >= analogies[1][1]


class TestCrossDomainIntegrationFramework:
    """Test main integration framework"""
    
    def test_framework_initialization(self):
        """Test initializing integration framework"""
        framework = CrossDomainIntegrationFramework(
            feature_dim=256,
            num_domains=len(Domain)
        )
        
        assert framework.feature_dim == 256
        assert framework.cross_modal_attention is not None
        assert framework.domain_adaptation is not None
        assert framework.reasoner is not None
    
    def test_modality_integration(self):
        """Test integrating multiple modalities"""
        framework = CrossDomainIntegrationFramework(feature_dim=256)
        
        # Create multi-modal input
        modality_inputs = {
            ModalityType.VISION: torch.randn(8, 256),
            ModalityType.AUDIO: torch.randn(8, 256),
            ModalityType.TEXT: torch.randn(8, 256)
        }
        
        integrated, attention = framework.integrate_modalities(modality_inputs)
        
        assert integrated.shape == (8, 256)
        assert attention is not None
    
    def test_domain_adaptation(self):
        """Test adapting representations across domains"""
        framework = CrossDomainIntegrationFramework(feature_dim=256)
        
        rep = UnifiedRepresentation(
            domain=Domain.VISUAL,
            features=torch.randn(256),
            abstract_features=torch.randn(256),
            semantic_embedding=torch.randn(256)
        )
        
        adapted = framework.adapt_domain(rep, Domain.LINGUISTIC)
        
        assert adapted.domain == Domain.LINGUISTIC
        assert adapted.features.shape == rep.features.shape
        assert 'adapted_from' in adapted.metadata
        assert adapted.metadata['adapted_from'] == Domain.VISUAL.value
    
    def test_consistency_validation(self):
        """Test validating cross-domain consistency"""
        framework = CrossDomainIntegrationFramework(feature_dim=256)
        
        # Create representations
        base_features = torch.randn(256)
        representations = []
        for domain in [Domain.VISUAL, Domain.AUDITORY, Domain.LINGUISTIC]:
            rep = UnifiedRepresentation(
                domain=domain,
                features=base_features + torch.randn(256) * 0.1,
                abstract_features=base_features,
                semantic_embedding=base_features
            )
            representations.append(rep)
        
        metrics = framework.validate_consistency(representations)
        
        assert 'consistency_score' in metrics
        assert 'semantic_coherence' in metrics
        assert 'num_representations' in metrics
        assert metrics['num_representations'] == 3
        # Should have high consistency for similar representations
        assert metrics['consistency_score'] > 0.7


class TestCrossDomainKnowledgeGraph:
    """Test knowledge graph integration"""
    
    def test_graph_creation(self):
        """Test creating knowledge graph"""
        graph = CrossDomainKnowledgeGraph()
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_node_addition(self):
        """Test adding nodes to graph"""
        graph = CrossDomainKnowledgeGraph()
        
        node = KnowledgeNode(
            node_id="test_node",
            content="Test content",
            domain=Domain.VISUAL,
            embedding=torch.randn(256)
        )
        
        graph.add_node(node)
        
        assert "test_node" in graph.nodes
        assert Domain.VISUAL in graph.domain_subgraphs
    
    def test_edge_addition(self):
        """Test adding edges to graph"""
        graph = CrossDomainKnowledgeGraph()
        
        node1 = KnowledgeNode(
            node_id="node1",
            content="Content 1",
            domain=Domain.VISUAL,
            embedding=torch.randn(256)
        )
        
        node2 = KnowledgeNode(
            node_id="node2",
            content="Content 2",
            domain=Domain.LINGUISTIC,
            embedding=torch.randn(256)
        )
        
        edge = KnowledgeEdge(
            source=node1,
            target=node2,
            relation_type=RelationType.REPRESENTS,
            strength=0.8
        )
        
        graph.add_edge(edge)
        
        assert len(graph.edges) == 1
        assert "node1" in graph.nodes
        assert "node2" in graph.nodes
    
    def test_cross_domain_paths(self):
        """Test finding paths across domains"""
        graph = CrossDomainKnowledgeGraph()
        
        # Create chain of nodes across domains
        nodes = []
        for i, domain in enumerate([Domain.VISUAL, Domain.LINGUISTIC, Domain.SEMANTIC]):
            node = KnowledgeNode(
                node_id=f"node{i}",
                content=f"Content {i}",
                domain=domain,
                embedding=torch.randn(256)
            )
            nodes.append(node)
            graph.add_node(node)
        
        # Connect them
        for i in range(len(nodes) - 1):
            edge = KnowledgeEdge(
                source=nodes[i],
                target=nodes[i+1],
                relation_type=RelationType.TRANSFORMS_TO,
                strength=0.9
            )
            graph.add_edge(edge)
        
        # Find path from first to last
        paths = graph.find_cross_domain_paths("node0", "node2", max_length=5)
        
        assert len(paths) > 0
        assert paths[0] == ["node0", "node1", "node2"]
    
    def test_domain_neighborhood(self):
        """Test getting neighborhood in target domain"""
        graph = CrossDomainKnowledgeGraph()
        
        # Create nodes in different domains
        center = KnowledgeNode(
            node_id="center",
            content="Center",
            domain=Domain.VISUAL,
            embedding=torch.randn(256)
        )
        graph.add_node(center)
        
        # Add neighbors in different domains
        for i in range(3):
            node = KnowledgeNode(
                node_id=f"neighbor{i}",
                content=f"Neighbor {i}",
                domain=Domain.LINGUISTIC,
                embedding=torch.randn(256)
            )
            graph.add_node(node)
            
            edge = KnowledgeEdge(
                source=center,
                target=node,
                relation_type=RelationType.SIMILAR_TO,
                strength=0.8
            )
            graph.add_edge(edge)
        
        # Get linguistic neighbors
        neighbors = graph.get_domain_neighborhood("center", Domain.LINGUISTIC, radius=2)
        
        assert len(neighbors) == 3
        assert all(nid.startswith("neighbor") for nid in neighbors)
    
    def test_activation_spreading(self):
        """Test spreading activation through graph"""
        graph = CrossDomainKnowledgeGraph()
        
        # Create simple chain
        for i in range(5):
            node = KnowledgeNode(
                node_id=f"node{i}",
                content=f"Content {i}",
                domain=Domain.VISUAL if i % 2 == 0 else Domain.LINGUISTIC,
                embedding=torch.randn(256)
            )
            graph.add_node(node)
        
        for i in range(4):
            edge = KnowledgeEdge(
                source=graph.nodes[f"node{i}"],
                target=graph.nodes[f"node{i+1}"],
                relation_type=RelationType.TRANSFORMS_TO,
                strength=0.8
            )
            graph.add_edge(edge)
        
        # Spread activation from first node
        activations = graph.activate_pattern(
            seed_nodes=["node0"],
            activation_threshold=0.1,
            decay=0.8
        )
        
        assert "node0" in activations
        assert activations["node0"] == 1.0
        # Should spread to neighbors
        assert len(activations) > 1
    
    def test_cross_domain_query(self):
        """Test querying across domains"""
        graph = CrossDomainKnowledgeGraph()
        
        # Create nodes in different domains
        query_embedding = torch.randn(256)
        
        for domain in [Domain.VISUAL, Domain.AUDITORY, Domain.LINGUISTIC]:
            for i in range(3):
                # Make some similar to query
                if i == 0:
                    embedding = query_embedding + torch.randn(256) * 0.1
                else:
                    embedding = torch.randn(256)
                
                node = KnowledgeNode(
                    node_id=f"{domain.value}_{i}",
                    content=f"Content {i}",
                    domain=domain,
                    embedding=embedding
                )
                graph.add_node(node)
        
        # Query from visual domain
        results = graph.query_cross_domain(
            query_embedding=query_embedding,
            source_domain=Domain.VISUAL,
            target_domains=[Domain.AUDITORY, Domain.LINGUISTIC],
            top_k=2
        )
        
        assert len(results) == 2
        # Results should be sorted by similarity
        assert results[0][1] >= results[1][1]


class TestKnowledgeGraphIntegrator:
    """Test knowledge graph integration and merging"""
    
    def test_integrator_creation(self):
        """Test creating integrator"""
        integrator = KnowledgeGraphIntegrator()
        
        assert integrator.main_graph is not None
        assert len(integrator.source_graphs) == 0
    
    def test_source_addition(self):
        """Test adding knowledge sources"""
        integrator = KnowledgeGraphIntegrator()
        
        graph1 = CrossDomainKnowledgeGraph()
        graph2 = CrossDomainKnowledgeGraph()
        
        integrator.add_source("source1", graph1)
        integrator.add_source("source2", graph2)
        
        assert len(integrator.source_graphs) == 2
        assert "source1" in integrator.source_graphs
    
    def test_source_merging(self):
        """Test merging multiple knowledge sources"""
        integrator = KnowledgeGraphIntegrator()
        
        # Create two source graphs with overlapping content
        graph1 = CrossDomainKnowledgeGraph()
        graph2 = CrossDomainKnowledgeGraph()
        
        # Add similar nodes to both
        base_embedding = torch.randn(256)
        for i, graph in enumerate([graph1, graph2]):
            node = KnowledgeNode(
                node_id=f"node{i}",
                content="Shared concept",
                domain=Domain.VISUAL,
                embedding=base_embedding + torch.randn(256) * 0.05  # Very similar
            )
            graph.add_node(node)
        
        integrator.add_source("graph1", graph1)
        integrator.add_source("graph2", graph2)
        
        # Merge with high similarity threshold
        integrator.merge_sources(similarity_threshold=0.9)
        
        # Should have merged similar nodes
        assert len(integrator.main_graph.nodes) >= 1
    
    def test_consistency_validation(self):
        """Test validating integrated graph consistency"""
        integrator = KnowledgeGraphIntegrator()
        
        # Create simple source graph
        graph = CrossDomainKnowledgeGraph()
        for i in range(5):
            node = KnowledgeNode(
                node_id=f"node{i}",
                content=f"Content {i}",
                domain=Domain.VISUAL,
                embedding=torch.randn(256)
            )
            graph.add_node(node)
        
        integrator.add_source("source", graph)
        integrator.merge_sources()
        
        # Validate consistency
        metrics = integrator.validate_consistency()
        
        assert 'num_nodes' in metrics
        assert 'num_edges' in metrics
        assert 'num_domains' in metrics
        assert metrics['num_nodes'] == 5

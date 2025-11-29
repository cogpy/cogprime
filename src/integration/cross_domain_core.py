"""
Cross-Domain Integration Framework Core

This module implements the core framework for integrating cognitive processing
across different domains and modalities, enabling synergistic effects through:
- Unified representation systems
- Cross-modal attention and integration
- Domain adaptation and alignment
- Abstract concept mapping
- Cross-domain reasoning and inference
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import torch
import torch.nn as nn

# Configuration constants
DOMAIN_SPECIFIC_WEIGHT = 0.3  # Weight for domain-specific features in adaptation
DEFAULT_ATTENTION_HEADS = [8, 4, 2, 1]  # Candidate values for attention heads


class Domain(Enum):
    """Cognitive processing domains"""

    VISUAL = "visual"
    AUDITORY = "auditory"
    LINGUISTIC = "linguistic"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    ABSTRACT = "abstract"
    SENSORIMOTOR = "sensorimotor"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class ModalityType(Enum):
    """Types of sensory modalities"""

    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"
    INTEROCEPTION = "interoception"


@dataclass
class UnifiedRepresentation:
    """
    Unified representation that transcends domain boundaries.

    Implements a glocal (global + local) representation strategy where
    representations maintain both domain-specific features and cross-domain
    abstractions.
    """

    domain: Domain
    features: torch.Tensor
    abstract_features: torch.Tensor
    semantic_embedding: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    attention_weights: Optional[torch.Tensor] = None
    cross_domain_links: Dict[Domain, float] = field(default_factory=dict)

    def similarity(self, other: "UnifiedRepresentation") -> float:
        """Compute similarity between representations"""
        # Combine abstract and domain-specific similarity
        abstract_sim = torch.cosine_similarity(
            self.abstract_features.unsqueeze(0), other.abstract_features.unsqueeze(0)
        ).item()

        semantic_sim = torch.cosine_similarity(
            self.semantic_embedding.unsqueeze(0), other.semantic_embedding.unsqueeze(0)
        ).item()

        # Weighted combination
        return 0.6 * abstract_sim + 0.4 * semantic_sim


@dataclass
class CrossDomainMapping:
    """
    Mapping between concepts in different domains.

    Enables analogical reasoning and knowledge transfer across domains.
    """

    source_domain: Domain
    target_domain: Domain
    source_concept: str
    target_concept: str
    mapping_strength: float
    transformation_matrix: torch.Tensor
    bidirectional: bool = True
    semantic_preservation: float = 0.0

    def apply(self, representation: UnifiedRepresentation) -> UnifiedRepresentation:
        """Apply mapping to transform representation to target domain"""
        if representation.domain != self.source_domain:
            raise ValueError(
                f"Representation domain {representation.domain} "
                f"does not match source domain {self.source_domain}"
            )

        # Transform features
        transformed_features = torch.matmul(representation.features, self.transformation_matrix)

        # Preserve abstract features (they transcend domains)
        return UnifiedRepresentation(
            domain=self.target_domain,
            features=transformed_features,
            abstract_features=representation.abstract_features,
            semantic_embedding=representation.semantic_embedding,
            metadata={
                **representation.metadata,
                "mapped_from": self.source_domain.value,
                "mapping_strength": self.mapping_strength,
            },
            cross_domain_links={
                **representation.cross_domain_links,
                self.source_domain: self.mapping_strength,
            },
        )


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for integrating information across modalities.

    Implements attention-based fusion that learns which modalities are most
    relevant for different tasks and contexts.
    """

    def __init__(self, feature_dim: int, num_modalities: int, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads

        # Multi-head attention for cross-modal integration
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )

        # Modality-specific projections
        self.modality_projections = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)]
        )

        # Fusion layer - takes concatenation of mean-pooled and max-pooled features across modalities
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(
        self, modality_features: List[torch.Tensor], masks: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate features across modalities.

        Args:
            modality_features: List of feature tensors, one per modality
            masks: Optional attention masks for each modality

        Returns:
            integrated_features: Fused cross-modal features
            attention_weights: Attention weights showing modality importance
        """
        # Project each modality
        projected = []
        for i, features in enumerate(modality_features):
            proj = self.modality_projections[i](features)
            projected.append(proj)

        # Stack for attention
        stacked = torch.stack(projected, dim=1)  # [batch, num_modalities, feature_dim]

        # Apply multi-head attention
        attended, attention_weights = self.multihead_attn(query=stacked, key=stacked, value=stacked)

        # Pool across modalities instead of concatenating
        pooled = attended.mean(dim=1)  # [batch, feature_dim]

        # Apply final fusion with concatenation of pooled and max-pooled
        max_pooled = attended.max(dim=1)[0]  # [batch, feature_dim]
        concatenated = torch.cat([pooled, max_pooled], dim=1)  # [batch, 2*feature_dim]
        integrated = self.fusion(concatenated)

        return integrated, attention_weights


class DomainAdaptation(nn.Module):
    """
    Domain adaptation module for aligning representations across domains.

    Uses adversarial training and alignment losses to learn domain-invariant
    representations while preserving domain-specific information.
    """

    def __init__(self, feature_dim: int, num_domains: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains

        # Domain-specific encoders
        self.domain_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, feature_dim),
                )
                for _ in range(num_domains)
            ]
        )

        # Shared encoder for domain-invariant features
        self.shared_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # Domain discriminator (for adversarial training)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_domains),
        )

    def encode(self, features: torch.Tensor, domain_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode features into domain-specific and domain-invariant components.

        Args:
            features: Input features
            domain_id: ID of the source domain

        Returns:
            domain_specific: Domain-specific features
            domain_invariant: Domain-invariant features
        """
        # Extract domain-specific features
        domain_specific = self.domain_encoders[domain_id](features)

        # Extract domain-invariant features
        domain_invariant = self.shared_encoder(features)

        return domain_specific, domain_invariant

    def align_domains(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_domain: int,
        target_domain: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Align features from source and target domains.

        Returns alignment losses and adapted features.
        """
        # Encode both domains
        source_specific, source_invariant = self.encode(source_features, source_domain)
        target_specific, target_invariant = self.encode(target_features, target_domain)

        # Alignment loss (minimize distance between invariant features)
        alignment_loss = torch.nn.functional.mse_loss(source_invariant, target_invariant)

        # Domain classification loss (for adversarial training)
        source_domain_pred = self.domain_discriminator(source_invariant)
        target_domain_pred = self.domain_discriminator(target_invariant)

        return {
            "source_specific": source_specific,
            "source_invariant": source_invariant,
            "target_specific": target_specific,
            "target_invariant": target_invariant,
            "alignment_loss": alignment_loss,
            "source_domain_pred": source_domain_pred,
            "target_domain_pred": target_domain_pred,
        }


@dataclass
class AbstractConcept:
    """
    Abstract concept that can be instantiated across multiple domains.

    Represents high-level patterns and relationships that transcend
    domain boundaries, enabling conceptual transfer and generalization.
    """

    concept_id: str
    name: str
    abstract_features: torch.Tensor
    domain_instantiations: Dict[Domain, UnifiedRepresentation] = field(default_factory=dict)
    conceptual_relationships: List[Tuple[str, float]] = field(default_factory=list)
    abstraction_level: int = 0

    def instantiate_in_domain(
        self, domain: Domain, context: Optional[Dict] = None
    ) -> UnifiedRepresentation:
        """
        Instantiate this abstract concept in a specific domain.

        Args:
            domain: Target domain for instantiation
            context: Optional context for guided instantiation

        Returns:
            Domain-specific representation of the concept
        """
        if domain in self.domain_instantiations:
            # Use cached instantiation
            return self.domain_instantiations[domain]

        # Create new instantiation
        # Start with abstract features and add domain-specific variance
        domain_features = self.abstract_features + torch.randn_like(self.abstract_features) * 0.1

        representation = UnifiedRepresentation(
            domain=domain,
            features=domain_features,
            abstract_features=self.abstract_features,
            semantic_embedding=self.abstract_features,  # Initialize same as abstract
            metadata={
                "concept_id": self.concept_id,
                "concept_name": self.name,
                "abstraction_level": self.abstraction_level,
            },
        )

        # Cache for future use
        self.domain_instantiations[domain] = representation

        return representation

    def add_relationship(self, related_concept_id: str, strength: float):
        """Add a conceptual relationship to another abstract concept"""
        self.conceptual_relationships.append((related_concept_id, strength))


class CrossDomainReasoner:
    """
    Cross-domain reasoning engine.

    Performs logical inference and reasoning that spans multiple domains,
    enabling transfer of knowledge and insights across domain boundaries.
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.abstract_concepts: Dict[str, AbstractConcept] = {}
        self.domain_mappings: Dict[Tuple[Domain, Domain], List[CrossDomainMapping]] = {}

    def register_concept(self, concept: AbstractConcept):
        """Register an abstract concept for cross-domain reasoning"""
        self.abstract_concepts[concept.concept_id] = concept

    def register_mapping(self, mapping: CrossDomainMapping):
        """Register a cross-domain mapping"""
        key = (mapping.source_domain, mapping.target_domain)
        if key not in self.domain_mappings:
            self.domain_mappings[key] = []
        self.domain_mappings[key].append(mapping)

        # Add reverse mapping if bidirectional
        if mapping.bidirectional:
            reverse_key = (mapping.target_domain, mapping.source_domain)
            if reverse_key not in self.domain_mappings:
                self.domain_mappings[reverse_key] = []
            # Create reverse mapping
            reverse_mapping = CrossDomainMapping(
                source_domain=mapping.target_domain,
                target_domain=mapping.source_domain,
                source_concept=mapping.target_concept,
                target_concept=mapping.source_concept,
                mapping_strength=mapping.mapping_strength,
                transformation_matrix=mapping.transformation_matrix.T,
                bidirectional=False,  # Avoid infinite recursion
                semantic_preservation=mapping.semantic_preservation,
            )
            self.domain_mappings[reverse_key].append(reverse_mapping)

    def transfer_knowledge(
        self,
        representation: UnifiedRepresentation,
        target_domain: Domain,
        confidence_threshold: float = 0.5,
    ) -> Optional[UnifiedRepresentation]:
        """
        Transfer knowledge from one domain to another.

        Args:
            representation: Source representation
            target_domain: Target domain for transfer
            confidence_threshold: Minimum mapping strength required

        Returns:
            Transferred representation in target domain, or None if no valid mapping
        """
        key = (representation.domain, target_domain)

        if key not in self.domain_mappings:
            return None

        # Find best mapping
        best_mapping = None
        best_strength = 0.0

        for mapping in self.domain_mappings[key]:
            if mapping.mapping_strength > best_strength:
                best_mapping = mapping
                best_strength = mapping.mapping_strength

        if best_mapping and best_strength >= confidence_threshold:
            return best_mapping.apply(representation)

        return None

    def find_analogies(
        self, source: UnifiedRepresentation, candidates: List[UnifiedRepresentation], top_k: int = 3
    ) -> List[Tuple[UnifiedRepresentation, float]]:
        """
        Find analogical matches across domains.

        Args:
            source: Source representation
            candidates: Candidate representations to match against
            top_k: Number of top matches to return

        Returns:
            List of (representation, similarity_score) tuples
        """
        scores = []
        for candidate in candidates:
            similarity = source.similarity(candidate)
            scores.append((candidate, similarity))

        # Sort by similarity and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def infer_cross_domain(
        self, premises: List[UnifiedRepresentation], target_domain: Optional[Domain] = None
    ) -> List[UnifiedRepresentation]:
        """
        Perform cross-domain inference from given premises.

        Args:
            premises: Input representations from potentially different domains
            target_domain: Optional target domain for conclusions

        Returns:
            Inferred representations
        """
        conclusions = []

        # Find abstract concepts matching premises
        matching_concepts = []
        for premise in premises:
            for concept in self.abstract_concepts.values():
                # Check if concept matches premise
                if premise.domain in concept.domain_instantiations:
                    inst = concept.domain_instantiations[premise.domain]
                    if premise.similarity(inst) > 0.7:
                        matching_concepts.append(concept)

        # Generate conclusions based on conceptual relationships
        for concept in matching_concepts:
            for related_id, strength in concept.conceptual_relationships:
                if strength > 0.5 and related_id in self.abstract_concepts:
                    related_concept = self.abstract_concepts[related_id]

                    # Instantiate in target domain or first premise domain
                    domain = target_domain if target_domain else premises[0].domain
                    conclusion = related_concept.instantiate_in_domain(domain)
                    conclusions.append(conclusion)

        return conclusions


class CrossDomainIntegrationFramework:
    """
    Main framework for cross-domain cognitive integration.

    Coordinates all cross-domain integration mechanisms to enable synergistic
    cognitive processing across domains and modalities.
    """

    def __init__(self, feature_dim: int = 512, num_domains: int = len(Domain)):
        self.feature_dim = feature_dim
        self.num_domains = num_domains

        # Core components
        self.cross_modal_attention = CrossModalAttention(
            feature_dim=feature_dim, num_modalities=len(ModalityType), num_heads=8
        )

        self.domain_adaptation = DomainAdaptation(feature_dim=feature_dim, num_domains=num_domains)

        self.reasoner = CrossDomainReasoner(feature_dim=feature_dim)

        # State tracking
        self.active_representations: Dict[Domain, List[UnifiedRepresentation]] = {
            domain: [] for domain in Domain
        }

    def integrate_modalities(
        self, modality_inputs: Dict[ModalityType, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate multiple sensory modalities into unified representation.

        Args:
            modality_inputs: Dictionary mapping modality types to feature tensors

        Returns:
            integrated_features: Fused multi-modal features
            attention_weights: Attention weights showing modality importance
        """
        # Sort by value to get consistent ordering
        features = [
            modality_inputs[mod] for mod in sorted(modality_inputs.keys(), key=lambda x: x.value)
        ]

        # Create temporary attention module with correct number of modalities
        num_modalities = len(modality_inputs)
        # Find a num_heads that divides feature_dim evenly
        num_heads = next((h for h in DEFAULT_ATTENTION_HEADS if self.feature_dim % h == 0), 1)

        temp_attention = CrossModalAttention(
            feature_dim=self.feature_dim, num_modalities=num_modalities, num_heads=num_heads
        )

        return temp_attention(features)

    def adapt_domain(
        self, representation: UnifiedRepresentation, target_domain: Domain
    ) -> UnifiedRepresentation:
        """
        Adapt representation to target domain while maintaining accuracy.

        Args:
            representation: Source representation
            target_domain: Target domain for adaptation

        Returns:
            Adapted representation in target domain
        """
        # Get domain IDs
        source_id = list(Domain).index(representation.domain)
        target_id = list(Domain).index(target_domain)

        # Encode with domain adaptation
        domain_specific, domain_invariant = self.domain_adaptation.encode(
            representation.features, source_id
        )

        # Decode in target domain
        target_specific, _ = self.domain_adaptation.encode(representation.features, target_id)

        # Combine invariant features with target-specific (weighted to preserve domain-invariance)
        adapted_features = domain_invariant + DOMAIN_SPECIFIC_WEIGHT * target_specific

        return UnifiedRepresentation(
            domain=target_domain,
            features=adapted_features,
            abstract_features=representation.abstract_features,
            semantic_embedding=representation.semantic_embedding,
            metadata={**representation.metadata, "adapted_from": representation.domain.value},
            cross_domain_links={
                **representation.cross_domain_links,
                representation.domain: 0.9,  # High link strength to source
            },
        )

    def validate_consistency(
        self, representations: List[UnifiedRepresentation]
    ) -> Dict[str, float]:
        """
        Validate cross-domain knowledge consistency.

        Args:
            representations: List of representations to check for consistency

        Returns:
            Dictionary of consistency metrics
        """
        if len(representations) < 2:
            return {"consistency_score": 1.0, "semantic_coherence": 1.0}

        # Compute pairwise similarities
        similarities = []
        for i in range(len(representations)):
            for j in range(i + 1, len(representations)):
                sim = representations[i].similarity(representations[j])
                similarities.append(sim)

        # Compute semantic coherence
        semantic_vectors = [rep.semantic_embedding for rep in representations]
        semantic_stack = torch.stack(semantic_vectors)
        pairwise_semantic = torch.mm(semantic_stack, semantic_stack.T)
        semantic_coherence = pairwise_semantic.mean().item()

        return {
            "consistency_score": float(np.mean(similarities)),
            "semantic_coherence": semantic_coherence,
            "num_representations": len(representations),
            "cross_domain_connections": sum(len(rep.cross_domain_links) for rep in representations),
        }

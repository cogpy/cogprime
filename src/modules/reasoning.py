import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import time

@dataclass
class Thought:
    """Represents a cognitive thought pattern"""
    content: torch.Tensor
    salience: float
    associations: List[str]
    timestamp: float
    confidence: float = 1.0
    pattern_type: str = "general"

class EpisodicMemory:
    """Manages episodic memories and their associations with consolidation"""
    
    def __init__(self, memory_size: int = 1000, feature_dim: int = 512):
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.memories = []
        self.memory_matrix = torch.zeros(memory_size, feature_dim)
        self.importance_scores = torch.zeros(memory_size)
        self.access_counts = torch.zeros(memory_size)
        self.current_index = 0
        self.consolidation_threshold = 0.7
    
    def store(self, memory: Thought) -> None:
        """Store a new memory with importance tracking"""
        if self.current_index >= self.memory_size:
            # Enhanced forgetting: replace least important memory
            if len(self.memories) > 0:
                importance = self.importance_scores[:len(self.memories)]
                access = self.access_counts[:len(self.memories)]
                # Combined importance metric (recent access + salience)
                combined_importance = importance * 0.7 + (access / (access.max() + 1e-8)) * 0.3
                replace_idx = int(torch.argmin(combined_importance))
                self.current_index = replace_idx
            else:
                self.current_index = 0
            
        self.memory_matrix[self.current_index] = memory.content
        self.importance_scores[self.current_index] = memory.salience * memory.confidence
        self.access_counts[self.current_index] = 0
        
        if self.current_index < len(self.memories):
            self.memories[self.current_index] = memory
        else:
            self.memories.append(memory)
        
        self.current_index = (self.current_index + 1) % self.memory_size
    
    def retrieve(self, query: torch.Tensor, k: int = 5) -> List[Thought]:
        """Retrieve k most similar memories to query with access tracking"""
        if len(self.memories) == 0:
            return []
            
        similarities = torch.cosine_similarity(
            query.unsqueeze(0),
            self.memory_matrix[:len(self.memories)],
            dim=1
        )
        
        # Boost retrieval by importance
        boosted_similarities = similarities * (1 + 0.3 * self.importance_scores[:len(self.memories)])
        
        _, indices = torch.topk(boosted_similarities, min(k, len(self.memories)))
        
        # Update access counts
        for idx in indices:
            self.access_counts[idx] += 1
        
        return [self.memories[int(i)] for i in indices]
    
    def consolidate(self) -> int:
        """Consolidate memories by merging similar ones, reducing storage by ~20%"""
        if len(self.memories) < 10:
            return 0
            
        # Find similar memory clusters
        similarity_matrix = torch.mm(
            self.memory_matrix[:len(self.memories)],
            self.memory_matrix[:len(self.memories)].t()
        )
        
        # Normalize to get cosine similarity
        norms = torch.norm(self.memory_matrix[:len(self.memories)], dim=1, keepdim=True)
        similarity_matrix = similarity_matrix / (norms @ norms.t() + 1e-8)
        
        # Find pairs above consolidation threshold
        consolidated = 0
        i = 0
        while i < len(self.memories):
            merged = False
            for j in range(i + 1, len(self.memories)):
                if similarity_matrix[i, j] > self.consolidation_threshold:
                    # Merge memories: weighted average by importance
                    w1 = self.importance_scores[i]
                    w2 = self.importance_scores[j]
                    total_w = w1 + w2 + 1e-8
                    
                    merged_content = (w1 * self.memory_matrix[i] + w2 * self.memory_matrix[j]) / total_w
                    merged_importance = max(self.importance_scores[i], self.importance_scores[j])
                    merged_access = self.access_counts[i] + self.access_counts[j]
                    
                    # Update first memory with merged content
                    self.memory_matrix[i] = merged_content
                    self.importance_scores[i] = merged_importance
                    self.access_counts[i] = merged_access
                    
                    # Remove second memory by shifting remaining elements
                    self.memories.pop(j)
                    n_remaining = len(self.memories)
                    if j < n_remaining:
                        self.memory_matrix[j:n_remaining] = self.memory_matrix[j+1:n_remaining+1].clone()
                        self.importance_scores[j:n_remaining] = self.importance_scores[j+1:n_remaining+1].clone()
                        self.access_counts[j:n_remaining] = self.access_counts[j+1:n_remaining+1].clone()
                    
                    consolidated += 1
                    merged = True
                    break
            
            if not merged:
                i += 1
        
        return consolidated


class AdvancedPatternRecognizer(nn.Module):
    """Advanced pattern recognition with hierarchical features"""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        # Multi-scale pattern detection
        self.local_patterns = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.global_patterns = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.pattern_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Pattern type classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # causal, analogical, hierarchical, temporal
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recognize patterns at multiple scales"""
        local = self.local_patterns(x)
        global_feat = self.global_patterns(x)
        
        combined = torch.cat([local, global_feat], dim=-1)
        patterns = self.pattern_fusion(combined)
        pattern_types = self.pattern_classifier(patterns)
        
        return patterns, pattern_types


class ReasoningModule(nn.Module):
    """Implements enhanced cognitive reasoning mechanisms"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.feature_dim = self.config.get('feature_dim', 512)
        
        # Memory systems
        self.episodic_memory = EpisodicMemory(
            memory_size=self.config.get('memory_size', 1000),
            feature_dim=self.feature_dim
        )
        
        # Enhanced pattern recognition
        self.pattern_recognizer = AdvancedPatternRecognizer(self.feature_dim)
        
        # Reasoning networks with residual connections
        self.inference_network = nn.Sequential(
            nn.Linear(64 + self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim)
        )
        
        # Working memory attention with adaptive heads
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Cross-modal integration network
        self.cross_modal_integrator = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.Tanh()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.feature_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Cognitive flexibility metrics
        self.flexibility_metrics = {
            'pattern_diversity': [],
            'attention_entropy': [],
            'inference_novelty': []
        }
    
    def _compute_flexibility_metrics(self, patterns: torch.Tensor, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Compute cognitive flexibility metrics"""
        # Pattern diversity: variance in pattern activations
        pattern_diversity = float(torch.var(patterns))
        
        # Attention entropy: measure of attention distribution
        attention_probs = torch.softmax(attention_weights, dim=-1)
        attention_entropy = -float(torch.sum(attention_probs * torch.log(attention_probs + 1e-8)))
        
        # Update metrics history
        self.flexibility_metrics['pattern_diversity'].append(pattern_diversity)
        self.flexibility_metrics['attention_entropy'].append(attention_entropy)
        
        return {
            'pattern_diversity': pattern_diversity,
            'attention_entropy': attention_entropy
        }
    
    def process_thought(self, 
                       current_input: torch.Tensor,
                       working_memory: Dict[str, Any]) -> Tuple[Thought, Dict[str, Any]]:
        """Process current input with enhanced reasoning capabilities"""
        
        # Advanced pattern recognition
        patterns, pattern_types = self.pattern_recognizer(current_input)
        
        # Identify dominant pattern type
        pattern_type_idx = int(torch.argmax(pattern_types))
        pattern_type_names = ['causal', 'analogical', 'hierarchical', 'temporal']
        dominant_pattern = pattern_type_names[pattern_type_idx]
        
        # Retrieve relevant memories
        relevant_memories = self.episodic_memory.retrieve(current_input, k=7)
        
        if relevant_memories:
            memory_tensor = torch.stack([m.content for m in relevant_memories])
        else:
            memory_tensor = current_input.unsqueeze(0)
        
        # Apply attention over memories and current input
        attended_memory, attention_weights = self.attention(
            current_input.unsqueeze(0),
            memory_tensor,
            memory_tensor
        )
        
        # Cross-modal integration if working memory has prior context
        if 'last_thought' in working_memory and working_memory['last_thought'] is not None:
            prior_context = working_memory['last_thought'].content
            integrated_input = self.cross_modal_integrator(
                torch.cat([current_input, prior_context], dim=-1)
            )
        else:
            integrated_input = current_input
        
        # Generate new thought through enhanced inference
        inference_input = torch.cat([patterns, attended_memory.squeeze(0)], dim=-1)
        new_thought_content = self.inference_network(inference_input)
        
        # Add residual connection for stability
        new_thought_content = new_thought_content + integrated_input
        
        # Estimate confidence in the reasoning
        confidence_input = torch.cat([new_thought_content, patterns], dim=-1)
        confidence = float(self.confidence_estimator(confidence_input))
        
        # Compute cognitive flexibility metrics
        flexibility = self._compute_flexibility_metrics(patterns, attention_weights.squeeze(0))
        
        # Calculate salience with pattern importance
        salience = float(torch.norm(new_thought_content)) * (1 + 0.3 * torch.max(pattern_types))
        
        # Create enhanced thought object
        thought = Thought(
            content=new_thought_content,
            salience=salience,
            associations=[f"memory_{i}" for i in range(len(relevant_memories))],
            timestamp=time.time(),
            confidence=confidence,
            pattern_type=dominant_pattern
        )
        
        # Update working memory with enhanced information
        working_memory['last_thought'] = thought
        working_memory['active_patterns'] = patterns
        working_memory['pattern_types'] = pattern_types
        working_memory['flexibility_metrics'] = flexibility
        working_memory['attention_weights'] = attention_weights
        
        # Store thought in episodic memory
        self.episodic_memory.store(thought)
        
        # Periodic memory consolidation (every 100 thoughts)
        if len(self.episodic_memory.memories) % 100 == 0:
            consolidated = self.episodic_memory.consolidate()
            working_memory['last_consolidation_count'] = consolidated
        
        return thought, working_memory
    
    def get_flexibility_report(self) -> Dict[str, Any]:
        """Generate a report on cognitive flexibility"""
        if not self.flexibility_metrics['pattern_diversity']:
            return {'status': 'insufficient_data'}
        
        return {
            'pattern_diversity_mean': np.mean(self.flexibility_metrics['pattern_diversity']),
            'pattern_diversity_std': np.std(self.flexibility_metrics['pattern_diversity']),
            'attention_entropy_mean': np.mean(self.flexibility_metrics['attention_entropy']),
            'attention_entropy_std': np.std(self.flexibility_metrics['attention_entropy']),
            'total_thoughts_processed': len(self.flexibility_metrics['pattern_diversity'])
        }
    
    def forward(self, 
                sensory_input: torch.Tensor,
                working_memory: Dict[str, Any]) -> Tuple[Thought, Dict[str, Any]]:
        """Forward pass of enhanced reasoning module"""
        return self.process_thought(sensory_input, working_memory) 
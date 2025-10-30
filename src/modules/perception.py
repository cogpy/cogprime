import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class SensoryInput:
    """Represents different types of sensory inputs"""
    visual: torch.Tensor = None
    auditory: torch.Tensor = None
    proprioceptive: torch.Tensor = None
    text: str = None

class AdaptiveAttentionAllocator(nn.Module):
    """Dynamic attention allocation based on environmental demands"""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Attention policy network
        self.attention_policy = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),  # Current + context
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Softmax(dim=-1)
        )
        
        # Saliency detector
        self.saliency_detector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Context buffer for temporal adaptation
        self.context_buffer = None
        self.attention_history = []
    
    def forward(self, features: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive attention weights"""
        if context is None:
            if self.context_buffer is not None:
                context = self.context_buffer
            else:
                context = torch.zeros_like(features)
        
        # Compute saliency map
        saliency = self.saliency_detector(features)
        
        # Compute attention weights considering context
        combined = torch.cat([features, context], dim=-1)
        attention_weights = self.attention_policy(combined)
        
        # Modulate by saliency
        modulated_weights = attention_weights * (1 + 0.5 * saliency)
        modulated_weights = modulated_weights / (modulated_weights.sum() + 1e-8)
        
        # Update context buffer
        self.context_buffer = features.detach()
        self.attention_history.append(modulated_weights.detach())
        
        # Keep only recent history
        if len(self.attention_history) > 50:
            self.attention_history.pop(0)
        
        return modulated_weights, saliency
    
    def get_attention_dynamics(self) -> Dict[str, float]:
        """Analyze attention allocation dynamics"""
        if len(self.attention_history) < 2:
            return {'entropy': 0.0, 'stability': 1.0}
        
        recent_weights = torch.stack(self.attention_history[-10:])
        
        # Compute entropy (diversity of attention)
        mean_weights = recent_weights.mean(dim=0)
        entropy = -float(torch.sum(mean_weights * torch.log(mean_weights + 1e-8)))
        
        # Compute stability (low variance = stable)
        stability = 1.0 / (1.0 + float(torch.var(recent_weights)))
        
        return {
            'entropy': entropy,
            'stability': stability
        }


class CrossModalIntegrator(nn.Module):
    """Integrates information across sensory modalities"""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        
        # Modal-specific transformations
        self.visual_transform = nn.Linear(256, feature_dim // 2)
        self.auditory_transform = nn.Linear(128, feature_dim // 2)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Integration network
        self.integration_network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Accuracy tracking
        self.integration_scores = []
    
    def forward(self, visual_features: Optional[torch.Tensor], 
                auditory_features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """Integrate cross-modal information"""
        modalities = []
        
        if visual_features is not None:
            visual_transformed = self.visual_transform(visual_features)
            modalities.append(visual_transformed)
        
        if auditory_features is not None:
            auditory_transformed = self.auditory_transform(auditory_features)
            modalities.append(auditory_transformed)
        
        if not modalities:
            return torch.zeros(512), 0.0
        
        # Concatenate available modalities
        if len(modalities) == 1:
            combined = torch.cat([modalities[0], torch.zeros(256)], dim=-1)
            integration_quality = 0.5
        else:
            combined = torch.cat(modalities, dim=-1)
            integration_quality = 1.0
        
        # Apply cross-modal attention
        attended, attention_weights = self.cross_attention(
            combined.unsqueeze(0),
            combined.unsqueeze(0),
            combined.unsqueeze(0)
        )
        
        # Final integration
        integrated = self.integration_network(attended.squeeze(0))
        
        # Track integration quality
        self.integration_scores.append(integration_quality)
        
        return integrated, integration_quality
    
    def get_integration_performance(self) -> Dict[str, float]:
        """Get cross-modal integration performance metrics"""
        if not self.integration_scores:
            return {'mean_quality': 0.0, 'improvement': 0.0}
        
        mean_quality = np.mean(self.integration_scores)
        
        # Calculate improvement trend
        if len(self.integration_scores) > 10:
            recent = np.mean(self.integration_scores[-10:])
            historical = np.mean(self.integration_scores[:-10])
            improvement = ((recent - historical) / (historical + 1e-8)) * 100
        else:
            improvement = 0.0
        
        return {
            'mean_quality': mean_quality,
            'improvement': improvement
        }


class SensoryEncoder(nn.Module):
    """Encodes different types of sensory inputs with enhanced cross-modal integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # Visual processing pathway - enhanced
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.config.get('visual_dim', 784), 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Auditory processing pathway - enhanced
        self.auditory_encoder = nn.Sequential(
            nn.Linear(self.config.get('audio_dim', 256), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # Cross-modal integration
        self.cross_modal_integrator = CrossModalIntegrator(512)
        
    def forward(self, sensory_input: SensoryInput) -> Tuple[torch.Tensor, float]:
        """Process and fuse sensory inputs with quality metric"""
        visual_features = None
        auditory_features = None
        
        if sensory_input.visual is not None:
            visual_features = self.visual_encoder(sensory_input.visual)
            
        if sensory_input.auditory is not None:
            auditory_features = self.auditory_encoder(sensory_input.auditory)
        
        # Cross-modal integration
        integrated, integration_quality = self.cross_modal_integrator(
            visual_features, auditory_features
        )
        
        return integrated, integration_quality


class PerceptionModule:
    """Enhanced perception module with adaptive attention and cross-modal integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.encoder = SensoryEncoder(config)
        self.attention_allocator = AdaptiveAttentionAllocator(512)
        self.perception_accuracy_history = []
        
    def process_input(self, sensory_input: SensoryInput, 
                     context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process sensory input with enhanced attention and cross-modal integration
        
        Returns:
            Tuple of (attended_features, perception_metrics)
        """
        # Encode sensory input with cross-modal integration
        encoded_input, integration_quality = self.encoder(sensory_input)
        
        # Apply adaptive attention allocation
        attention_weights, saliency = self.attention_allocator(encoded_input, context)
        
        # Apply attention to features
        attended_features = encoded_input * attention_weights
        
        # Calculate perception accuracy estimate
        perception_accuracy = float(integration_quality * saliency.mean())
        self.perception_accuracy_history.append(perception_accuracy)
        
        # Compile metrics
        attention_dynamics = self.attention_allocator.get_attention_dynamics()
        integration_perf = self.encoder.cross_modal_integrator.get_integration_performance()
        
        metrics = {
            'attention_weights': attention_weights,
            'saliency': saliency,
            'integration_quality': integration_quality,
            'perception_accuracy': perception_accuracy,
            'attention_entropy': attention_dynamics['entropy'],
            'attention_stability': attention_dynamics['stability'],
            'integration_mean_quality': integration_perf['mean_quality'],
            'integration_improvement': integration_perf['improvement']
        }
        
        return attended_features, metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive perception performance report"""
        if len(self.perception_accuracy_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_accuracy = np.mean(self.perception_accuracy_history[-50:])
        if len(self.perception_accuracy_history) > 100:
            baseline_accuracy = np.mean(self.perception_accuracy_history[:50])
            improvement = ((recent_accuracy - baseline_accuracy) / (baseline_accuracy + 1e-8)) * 100
        else:
            baseline_accuracy = recent_accuracy
            improvement = 0.0
        
        return {
            'current_accuracy': recent_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'improvement_percentage': improvement,
            'total_samples_processed': len(self.perception_accuracy_history)
        }
    
    def reset_attention(self) -> None:
        """Reset attention allocation to initial state"""
        self.attention_allocator.context_buffer = None
        self.attention_allocator.attention_history = [] 
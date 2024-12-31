import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class SensoryInput:
    """Represents different types of sensory inputs"""
    visual: torch.Tensor = None
    auditory: torch.Tensor = None
    proprioceptive: torch.Tensor = None
    text: str = None

class SensoryEncoder(nn.Module):
    """Encodes different types of sensory inputs into a unified representation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # Visual processing pathway
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.config.get('visual_dim', 784), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Auditory processing pathway
        self.auditory_encoder = nn.Sequential(
            nn.Linear(self.config.get('audio_dim', 256), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion_layer = nn.Linear(384, 512)  # 256 (visual) + 128 (audio) = 384
        
    def forward(self, sensory_input: SensoryInput) -> torch.Tensor:
        """Process and fuse sensory inputs into a unified representation"""
        encoded_features = []
        
        if sensory_input.visual is not None:
            visual_features = self.visual_encoder(sensory_input.visual)
            encoded_features.append(visual_features)
            
        if sensory_input.auditory is not None:
            audio_features = self.auditory_encoder(sensory_input.auditory)
            encoded_features.append(audio_features)
            
        # Concatenate available features
        if encoded_features:
            combined = torch.cat(encoded_features, dim=-1)
            return self.fusion_layer(combined)
        else:
            return torch.zeros(512)  # Default attention vector size

class PerceptionModule:
    """Main perception module that processes sensory inputs"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.encoder = SensoryEncoder(config)
        self.attention_weights = torch.ones(512) / 512  # Uniform attention initially
        
    def process_input(self, sensory_input: SensoryInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sensory input and return encoded representation and attention weights
        """
        # Encode sensory input
        encoded_input = self.encoder(sensory_input)
        
        # Apply attention mechanism
        attended_features = encoded_input * self.attention_weights
        
        # Update attention weights based on feature salience
        self.attention_weights = torch.softmax(torch.abs(encoded_input), dim=0)
        
        return attended_features, self.attention_weights
    
    def reset_attention(self) -> None:
        """Reset attention weights to uniform distribution"""
        self.attention_weights = torch.ones(512) / 512 
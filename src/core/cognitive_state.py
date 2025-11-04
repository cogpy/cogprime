"""
CogPrime Cognitive State Module

Defines the CognitiveState dataclass used throughout the system.
Separated to avoid circular imports between core and memory modules.
"""

import torch
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class CognitiveState:
    """Represents the current cognitive state of the system"""
    attention_focus: torch.Tensor
    working_memory: Dict[str, Any]
    emotional_valence: float
    goal_stack: List[str]
    sensory_buffer: Dict[str, torch.Tensor]
    current_thought: Any = None  # Thought type from reasoning module
    last_action: Any = None  # Action type from action module
    last_reward: float = 0.0
    total_reward: float = 0.0

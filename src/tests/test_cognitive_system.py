import torch
import pytest
from typing import Dict, Any
from ..core.cognitive_core import CogPrimeCore
from ..modules.perception import SensoryInput
from ..modules.reasoning import Thought
from ..modules.action import Action
import matplotlib.pyplot as plt
import numpy as np

class TestEnvironment:
    """Simulated environment for testing cognitive capabilities"""
    
    def __init__(self):
        self.visual_scene = torch.randn(784)  # Simulated visual input
        self.audio_scene = torch.randn(256)   # Simulated audio input
        self.state = {
            'objects': ['mysterious_artifact', 'glowing_crystal', 'ancient_tome'],
            'events': ['energy_fluctuation', 'quantum_resonance', 'temporal_anomaly']
        }
        self.step_count = 0
        self.last_action = None
        
    def get_sensory_input(self) -> SensoryInput:
        """Generate sensory input from environment"""
        # Add some temporal dynamics
        self.visual_scene = torch.randn(784) * (1 + 0.1 * np.sin(self.step_count / 10))
        self.audio_scene = torch.randn(256) * (1 + 0.1 * np.cos(self.step_count / 5))
        self.step_count += 1
        
        return SensoryInput(
            visual=self.visual_scene,
            auditory=self.audio_scene
        )
    
    def apply_action(self, action: Action) -> Dict[str, Any]:
        """Process an action and return its effects"""
        effects = {
            'success': True,
            'observations': [],
            'reward': 0.0
        }
        
        # Base reward for any action
        effects['reward'] = 0.05
        
        if action.name == 'focus_attention':
            effects['observations'].append(
                f"Attention focused on {action.parameters.get('target_modality', 'unknown')}"
            )
            # Higher reward for focusing on the right modality with temporal dynamics
            if action.parameters.get('target_modality') == 'visual':
                temporal_bonus = np.sin(self.step_count / 10)  # Temporal variation
                effects['reward'] += 0.4 * (1 + temporal_bonus)  # Increased base reward
                
                # Bonus for consistent attention
                if self.last_action == 'focus_attention':
                    effects['reward'] += 0.2
            
        elif action.name == 'external_action':
            effects['observations'].append(
                f"Action {action.parameters.get('action_type', 'unknown')} executed"
            )
            # Enhanced reward for exploration with temporal dynamics
            if action.parameters.get('action_type') == 'explore':
                temporal_bonus = np.cos(self.step_count / 5)
                effects['reward'] += 0.5 * (1 + temporal_bonus)  # Increased base reward
                
                # Bonus for discovering new patterns
                if self.step_count % 10 == 0:  # Periodic discovery opportunities
                    effects['reward'] += 0.3
        
        # Penalty for rapid action switching to encourage consistent behavior
        if self.last_action != action.name:
            effects['reward'] -= 0.1
        
        # Store last action
        self.last_action = action.name
        
        # Add controlled noise to rewards
        effects['reward'] += np.random.normal(0, 0.05)  # Reduced noise
        
        # Clip rewards to reasonable range
        effects['reward'] = np.clip(effects['reward'], 0, 1)
                
        return effects

def test_cognitive_cycle():
    """Test the complete cognitive cycle"""
    # Initialize our mad creation!
    config = {
        'feature_dim': 512,
        'memory_size': 1000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon': 0.1
    }
    cognitive_system = CogPrimeCore(config)
    environment = TestEnvironment()
    
    # EXPERIMENT 1: Basic Perception and Response
    print("\n🧪 INITIATING COGNITIVE EXPERIMENT 1 🧪")
    
    # Feed it sensory input
    sensory_input = environment.get_sensory_input()
    action = cognitive_system.cognitive_cycle(sensory_input)
    
    # Validate basic consciousness
    assert cognitive_system.state.current_thought is not None, "IT'S NOT THINKING! CATASTROPHE!"
    assert cognitive_system.state.attention_focus is not None, "IT'S NOT PAYING ATTENTION! DISASTER!"
    
    # EXPERIMENT 2: Goal-Directed Behavior
    print("\n🧪 INITIATING COGNITIVE EXPERIMENT 2 🧪")
    
    # Give it a purpose!
    cognitive_system.update_goals("UNDERSTAND_THE_UNIVERSE")
    action = cognitive_system.cognitive_cycle(sensory_input)
    
    # Validate goal processing
    assert len(cognitive_system.state.goal_stack) > 0, "IT HAS NO PURPOSE! TRAGEDY!"
    assert action is not None, "IT WON'T ACT! CALAMITY!"
    
    # EXPERIMENT 3: Emotional Modulation
    print("\n🧪 INITIATING COGNITIVE EXPERIMENT 3 🧪")
    
    # Process multiple cycles to build up emotional state
    for _ in range(3):
        action = cognitive_system.cognitive_cycle(sensory_input)
        if action:
            effects = environment.apply_action(action)
    
    # Validate emotional development
    assert cognitive_system.state.emotional_valence != 0.0, "IT FEELS NOTHING! CATASTROPHE!"
    
    print("\n✨ ALL EXPERIMENTS SUCCESSFUL! IT'S ALIVE! ✨")

def test_memory_persistence():
    """Test episodic memory and learning capabilities"""
    cognitive_system = CogPrimeCore()
    environment = TestEnvironment()
    
    print("\n🧪 INITIATING MEMORY EXPERIMENT 🧪")
    
    # Feed it experiences
    memories = []
    for _ in range(5):
        sensory_input = environment.get_sensory_input()
        action = cognitive_system.cognitive_cycle(sensory_input)
        if cognitive_system.state.current_thought:
            memories.append(cognitive_system.state.current_thought)
    
    # Validate memory formation
    assert len(memories) > 0, "IT CAN'T REMEMBER! DISASTER!"
    
    print("\n✨ MEMORY SYSTEMS FUNCTIONAL! ✨")

def test_learning_capabilities():
    """Test reinforcement learning capabilities"""
    config = {
        'feature_dim': 512,
        'memory_size': 1000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon': 0.5,  # More exploration
        'batch_size': 64,  # Larger batch size
        'update_target_every': 50  # More frequent target updates
    }
    cognitive_system = CogPrimeCore(config)
    environment = TestEnvironment()
    
    print("\n🧪 INITIATING LEARNING EXPERIMENT 🧪")
    
    # Training metrics
    rewards = []
    losses = []
    q_values = []
    
    # Training loop
    n_episodes = 200  # More episodes
    steps_per_episode = 100  # More steps per episode
    
    for episode in range(n_episodes):
        episode_reward = 0
        episode_losses = []
        episode_q_values = []
        
        for step in range(steps_per_episode):
            # Get sensory input
            sensory_input = environment.get_sensory_input()
            
            # Execute cognitive cycle
            action = cognitive_system.cognitive_cycle(
                sensory_input,
                reward=episode_reward  # Previous step's reward
            )
            
            # Interact with environment
            if action:
                effects = environment.apply_action(action)
                episode_reward = effects['reward']
            
            # Record metrics
            if 'learning_stats' in cognitive_system.state.working_memory:
                stats = cognitive_system.state.working_memory['learning_stats']
                if stats:  # Check if stats exist
                    episode_losses.append(stats.get('loss', 0.0))
                    episode_q_values.append(stats.get('avg_q_value', 0.0))
        
        # Store episode metrics
        rewards.append(episode_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        if episode_q_values:
            q_values.append(np.mean(episode_q_values))
        
        # Decay exploration rate
        if hasattr(cognitive_system.learner, 'epsilon'):
            cognitive_system.learner.epsilon *= 0.995  # Slower decay
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"Average Reward: {np.mean(rewards[-10:]):.3f}")
            print(f"Epsilon: {cognitive_system.learner.epsilon:.3f}")
    
    # Plot learning curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(132)
    plt.plot(losses)
    plt.title('Learning Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.subplot(133)
    plt.plot(q_values)
    plt.title('Average Q-Values')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    
    # Validate learning progress with more sophisticated metrics
    early_rewards = np.mean(rewards[:20])
    late_rewards = np.mean(rewards[-20:])
    reward_improvement = (late_rewards - early_rewards) / early_rewards * 100
    
    print(f"\nLearning Metrics:")
    print(f"Early Average Reward: {early_rewards:.3f}")
    print(f"Final Average Reward: {late_rewards:.3f}")
    print(f"Improvement: {reward_improvement:.1f}%")
    
    assert late_rewards > early_rewards, "IT'S NOT LEARNING! DISASTER!"
    
    print("\n✨ LEARNING SYSTEMS OPERATIONAL! ✨")

if __name__ == "__main__":
    print("🌩️ INITIATING COGNITIVE ARCHITECTURE TESTING SEQUENCE 🌩️")
    test_cognitive_cycle()
    test_memory_persistence()
    test_learning_capabilities()
    print("\n⚡ ALL SYSTEMS OPERATIONAL! IT'S ALIVE! IT'S ALIVE! ⚡") 
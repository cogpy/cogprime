# Quick Start Guide

This guide will help you get started with CogPrime quickly and understand its core capabilities.

## Installation and Setup

### Prerequisites Check

Ensure you have the required dependencies:

```bash
python --version  # Should be 3.8+
pip --version
```

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/drzo/cogprime.git
cd cogprime

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Basic Examples

### Example 1: Simple Cognitive Cycle

```python
#!/usr/bin/env python3
"""
Basic cognitive cycle example demonstrating perception, reasoning, and action.
"""

import torch
from src.core.cognitive_core import CogPrimeCore
from src.modules.perception import SensoryInput

def basic_cognitive_cycle():
    # Initialize the cognitive system
    config = {
        'visual_dim': 784,
        'audio_dim': 256,
        'memory_size': 100
    }
    
    cognitive_system = CogPrimeCore(config)
    
    # Simulate a series of cognitive cycles
    for cycle in range(5):
        print(f"\n--- Cognitive Cycle {cycle + 1} ---")
        
        # Create random sensory input (simulating real-world input)
        sensory_input = SensoryInput(
            visual=torch.randn(784),
            auditory=torch.randn(256),
            text=f"Input text for cycle {cycle + 1}"
        )
        
        # Provide reward signal (simulating environment feedback)
        reward = 1.0 if cycle % 2 == 0 else 0.5
        
        # Execute cognitive cycle
        action = cognitive_system.cognitive_cycle(sensory_input, reward)
        
        print(f"Reward received: {reward}")
        print(f"Action selected: {action}")
        print(f"Total accumulated reward: {cognitive_system.state.total_reward}")

if __name__ == "__main__":
    basic_cognitive_cycle()
```

### Example 2: Multi-Modal Perception

```python
#!/usr/bin/env python3
"""
Demonstrates multi-modal sensory processing and attention mechanisms.
"""

import torch
import numpy as np
from src.modules.perception import PerceptionModule, SensoryInput

def multimodal_perception_demo():
    config = {
        'visual_dim': 784,
        'audio_dim': 256,
        'attention_heads': 4
    }
    
    perception_module = PerceptionModule(config)
    
    # Create multi-modal sensory inputs
    inputs = [
        SensoryInput(
            visual=torch.randn(784),
            auditory=torch.randn(256),
            text="Visual and auditory input"
        ),
        SensoryInput(
            visual=torch.randn(784),
            text="Visual input only"
        ),
        SensoryInput(
            auditory=torch.randn(256),
            text="Auditory input only"
        )
    ]
    
    print("Multi-Modal Perception Processing:")
    
    for i, sensory_input in enumerate(inputs):
        print(f"\n--- Input {i + 1} ---")
        
        # Process the input
        processed = perception_module.process(sensory_input)
        
        print(f"Input modalities: {[k for k, v in sensory_input.__dict__.items() if v is not None]}")
        print(f"Processed representation shape: {processed.shape}")
        print(f"Attention focus: {torch.norm(processed[:10]):.3f}")  # First 10 dimensions as attention proxy

if __name__ == "__main__":
    multimodal_perception_demo()
```

### Example 3: Learning and Adaptation

```python
#!/usr/bin/env python3
"""
Demonstrates learning capabilities and adaptive behavior.
"""

import torch
from src.modules.learning import ReinforcementLearner, Experience

def learning_demo():
    config = {
        'state_dim': 512,
        'action_dim': 10,
        'learning_rate': 0.01,
        'exploration_rate': 0.1
    }
    
    learner = ReinforcementLearner(config)
    
    print("Learning and Adaptation Demo:")
    
    # Simulate learning episodes
    for episode in range(10):
        # Random state and action for demonstration
        state = torch.randn(512)
        action_idx = episode % 10  # Cycle through actions
        reward = np.random.normal(0.5, 0.2)  # Random reward with some noise
        next_state = torch.randn(512)
        
        # Create experience
        experience = Experience(
            state=state,
            action=action_idx,
            reward=reward,
            next_state=next_state,
            done=False
        )
        
        # Learn from experience
        loss = learner.learn(experience)
        
        print(f"Episode {episode + 1}: Reward = {reward:.3f}, Loss = {loss:.6f}")
        
        # Show policy evolution every few episodes
        if (episode + 1) % 3 == 0:
            action_probs = learner.get_action_probabilities(state)
            top_action = torch.argmax(action_probs).item()
            print(f"  Current preferred action: {top_action}, confidence: {action_probs[top_action]:.3f}")

if __name__ == "__main__":
    learning_demo()
```

### Example 4: Relevance Realization

```python
#!/usr/bin/env python3
"""
Demonstrates relevance realization and meaning-making processes.
"""

import torch
from src.vervaeke.relevance_realization import RelevanceProcessor

def relevance_demo():
    # Note: This is a conceptual example - actual implementation may vary
    processor = RelevanceProcessor()
    
    # Simulate different contexts and stimuli
    contexts = [
        {"goal": "learning", "environment": "academic", "time_pressure": "low"},
        {"goal": "survival", "environment": "dangerous", "time_pressure": "high"},
        {"goal": "social", "environment": "party", "time_pressure": "medium"}
    ]
    
    stimulus = torch.randn(256)  # Simulated perceptual input
    
    print("Relevance Realization Demo:")
    
    for i, context in enumerate(contexts):
        print(f"\n--- Context {i + 1}: {context['goal']} scenario ---")
        
        # Calculate relevance in different contexts
        relevance_score = processor.calculate_relevance(stimulus, context)
        salience_map = processor.generate_salience_map(stimulus, context)
        meaning = processor.generate_meaning(stimulus, context, relevance_score)
        
        print(f"Relevance Score: {relevance_score:.3f}")
        print(f"Salience Map (top 5): {salience_map[:5]}")
        print(f"Generated Meaning: {meaning}")

if __name__ == "__main__":
    relevance_demo()
```

## Configuration Examples

### Basic Configuration

```python
basic_config = {
    'visual_dim': 784,
    'audio_dim': 256,
    'memory_size': 1000,
    'learning_rate': 0.001
}
```

### Advanced Configuration

```python
advanced_config = {
    'perception': {
        'visual_dim': 1024,
        'audio_dim': 512,
        'attention_heads': 8,
        'fusion_layers': [512, 256, 128]
    },
    'reasoning': {
        'memory_size': 5000,
        'working_memory_capacity': 10,
        'attention_decay': 0.95,
        'inference_steps': 5
    },
    'action': {
        'action_dim': 20,
        'planning_horizon': 5,
        'exploration_strategy': 'epsilon_greedy'
    },
    'learning': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'exploration_rate': 0.1,
        'meta_learning_enabled': True,
        'replay_buffer_size': 10000
    },
    'vervaeke': {
        'relevance_threshold': 0.5,
        'meaning_integration_depth': 3,
        'salience_decay': 0.9
    }
}
```

### Research Configuration

```python
research_config = {
    'experiment_name': 'cognitive_synergy_study',
    'logging_level': 'DEBUG',
    'save_checkpoints': True,
    'checkpoint_interval': 100,
    'tensorboard_logging': True,
    'meta_learning': {
        'enabled': True,
        'strategy_pool_size': 10,
        'adaptation_rate': 0.01,
        'strategy_evaluation_episodes': 50
    },
    'cognitive_synergy': {
        'cross_module_communication': True,
        'emergence_detection': True,
        'synergy_optimization': True
    }
}
```

## Testing Your Setup

### Quick Functionality Test

```python
#!/usr/bin/env python3
"""
Quick test to verify all components are working.
"""

def test_setup():
    try:
        from src.core.cognitive_core import CogPrimeCore
        from src.modules.perception import SensoryInput
        print("✓ Core imports successful")
        
        # Test basic initialization
        cognitive_system = CogPrimeCore({'visual_dim': 10, 'audio_dim': 10})
        print("✓ Cognitive system initialization successful")
        
        # Test basic processing
        input_data = SensoryInput(visual=torch.randn(10), auditory=torch.randn(10))
        action = cognitive_system.cognitive_cycle(input_data)
        print("✓ Cognitive cycle execution successful")
        
        print("\n🎉 Setup verification complete! CogPrime is ready to use.")
        
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        print("Please check your installation and dependencies.")

if __name__ == "__main__":
    test_setup()
```

## Next Steps

1. **Explore the Examples**: Run the examples above to understand basic functionality
2. **Read the Architecture**: Review [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical information
3. **Experiment**: Modify the examples to explore different configurations
4. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md) for ways to contribute to the project

## Common Issues and Solutions

### Import Errors
```bash
# If you get import errors, ensure you're in the project root:
cd /path/to/cogprime
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### PyTorch Issues
```bash
# For CPU-only installations:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA support:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
```python
# For systems with limited memory, use smaller configurations:
minimal_config = {
    'visual_dim': 128,
    'audio_dim': 64,
    'memory_size': 100
}
```

## Getting Help

- **Documentation**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Issues**: [GitHub Issues](https://github.com/drzo/cogprime/issues)
- **Discussions**: [GitHub Discussions](https://github.com/drzo/cogprime/discussions)

Happy experimenting with CogPrime! 🧠✨
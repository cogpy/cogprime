# Ontogenesis: Self-Generating Kernels

## Overview

Ontogenesis is the implementation of **self-generating, evolving kernels** through recursive application of differential operators. It enables kernels to generate themselves, optimize themselves, reproduce with other kernels, and evolve across generations.

## Concept

The term "ontogenesis" refers to the process of origin and development of an organism. In the context of CogPrime's Universal Kernel Generator, ontogenesis means:

1. **Self-Generation**: Kernels can generate new kernels through recursive self-composition
2. **Self-Optimization**: Kernels can optimize their own grip through iterative improvement
3. **Self-Reproduction**: Two kernels can combine to create offspring with mixed genetic material
4. **Evolution**: Populations of kernels evolve over generations to maximize fitness

## Architecture

### Core Components

#### 1. Ontogenetic Kernel (`OntogeneticKernel`)

An enhanced kernel with genetic capabilities:

```python
from src.ontogenesis import OntogeneticKernel, GripMetrics
import numpy as np

# Create a kernel
coefficients = np.array([1.0, 0.5, 0.25, 0.125])
grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

kernel = OntogeneticKernel(
    order=4,
    coefficients=coefficients,
    domain_spec="consciousness",
    grip_metrics=grip
)
```

#### 2. Kernel Genome (`KernelGenome`)

The "DNA" of a kernel containing all genetic information:

- `id`: Unique identifier
- `generation`: Generation number
- `lineage`: Parent IDs
- `genes`: Genetic information (coefficient, operator, symmetry, preservation genes)
- `fitness`: Overall fitness score
- `age`: Age in generations

#### 3. Development Stages

Kernels progress through life stages:

- **Embryonic**: Just generated, basic structure
- **Juvenile**: Developing, optimizing
- **Mature**: Fully developed, capable of reproduction
- **Senescent**: Declining, ready for replacement

## Key Operations

### Self-Generation

A kernel generates offspring through recursive self-composition using differential operators:

```python
from src.ontogenesis import initialize_ontogenetic_kernel, self_generate

# Initialize kernel
parent = initialize_ontogenetic_kernel(base_kernel)

# Generate offspring
offspring = self_generate(parent)

print(f"Parent: {parent.genome.id}")
print(f"Offspring: {offspring.genome.id}")
print(f"Generation: {offspring.genome.generation}")
```

The operation applies the chain rule: `(f∘f)' = f'(f(x)) · f'(x)`

### Self-Optimization

A kernel optimizes itself through iterative grip improvement:

```python
from src.ontogenesis import self_optimize

optimized = self_optimize(kernel, iterations=10)

print(f"Maturity: {kernel.ontogenetic_state.maturity}")
print(f"Stage: {kernel.ontogenetic_state.stage}")
```

Each iteration:
1. Optimizes grip coefficients
2. Increases maturity
3. Records development event
4. Advances development stage when ready

### Self-Reproduction

Two kernels combine to create offspring:

```python
from src.ontogenesis import self_reproduce, ReproductionMethod

offspring1, offspring2 = self_reproduce(
    parent1, 
    parent2, 
    method=ReproductionMethod.CROSSOVER,
    mutation_rate=0.1
)
```

Methods:
- **CROSSOVER**: Single-point genetic crossover
- **MUTATION**: Random coefficient mutation
- **CLONING**: Direct copy

### Evolution

Populations evolve over generations:

```python
from src.ontogenesis import run_ontogenesis, OntogenesisConfig, EvolutionConfig

config = OntogenesisConfig(
    evolution=EvolutionConfig(
        population_size=20,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_rate=0.2,
        max_generations=100,
        fitness_threshold=0.9,
        diversity_pressure=0.1,
    ),
    seed_kernels=[kernel1, kernel2],
)

generations = run_ontogenesis(config)

# Analyze results
for gen in generations:
    print(f"Gen {gen.generation}: Best={gen.best_fitness:.4f}, Avg={gen.average_fitness:.4f}")
```

Evolution process:
1. **Fitness Evaluation**: Score each kernel
2. **Selection**: Tournament selection of parents
3. **Reproduction**: Crossover and mutation
4. **Elite Preservation**: Keep best individuals
5. **Stage Update**: Progress development stages

## Genetic Operations

### Genes

Kernels have four types of genes:

1. **Coefficient Genes**: Control B-series coefficients (mutable)
2. **Operator Genes**: Control differential operators (mutable)
3. **Symmetry Genes**: Preserve domain symmetries (immutable)
4. **Preservation Genes**: Maintain conserved quantities (immutable)

### Crossover

Single-point crossover on coefficient arrays:

```
Parent 1: [c1, c2, c3, c4, c5]
Parent 2: [d1, d2, d3, d4, d5]
          ^^^^^^^
           point

Offspring 1: [c1, c2, c3, d4, d5]
Offspring 2: [d1, d2, d3, c4, c5]
```

### Mutation

Random perturbation of coefficients:

```python
# Coefficient mutation: ±10%
coeff[i] += (random() - 0.5) * 0.2 * coeff[i]
```

## Fitness Evaluation

Fitness is a weighted combination of multiple factors:

```python
fitness = (
    grip * 0.4 +          # Quality of domain fit
    stability * 0.2 +     # Numerical stability
    efficiency * 0.2 +    # Computational efficiency
    novelty * 0.1 +       # Genetic diversity
    symmetry * 0.1        # Symmetry preservation
)
```

### Grip Components

- **Contact**: How well kernel touches domain
- **Coverage**: Completeness of span
- **Efficiency**: Computational cost
- **Stability**: Numerical properties

### Novelty

Measured as genetic distance from population:

```python
novelty = avg_distance(kernel, population)
```

## Examples

### Example 1: Simple Self-Generation

```python
from src.ontogenesis import (
    initialize_ontogenetic_kernel,
    self_generate,
    GeneratedKernel,
    GripMetrics
)
import numpy as np

# Create base kernel
coeffs = np.array([1.0, 0.5, 0.25, 0.125])
grip = GripMetrics(contact=0.9, coverage=0.8, efficiency=0.7, stability=0.85)

base_kernel = GeneratedKernel(
    order=4,
    coefficients=coeffs,
    domain_spec="consciousness",
    grip_metrics=grip
)

# Initialize as ontogenetic
parent = initialize_ontogenetic_kernel(base_kernel)

# Generate offspring
offspring = self_generate(parent)

print(f"Parent: {parent.genome.id}")
print(f"Offspring: {offspring.genome.id}")
print(f"Generation: {offspring.genome.generation}")
```

### Example 2: Multi-Generation Evolution

```python
from src.ontogenesis import run_ontogenesis, OntogenesisConfig, EvolutionConfig
from src.ontogenesis import summarize_evolution, get_best_kernel

# Create seed population
seed_kernels = [
    initialize_ontogenetic_kernel(create_kernel(i))
    for i in range(5)
]

# Configure evolution
config = OntogenesisConfig(
    evolution=EvolutionConfig(
        population_size=20,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elitism_rate=0.1,
        max_generations=50,
        fitness_threshold=0.9,
        diversity_pressure=0.2,
    ),
    seed_kernels=seed_kernels,
)

# Run evolution
generations = run_ontogenesis(config)

# Analyze results
summary = summarize_evolution(generations)
print(f"Total generations: {summary['total_generations']}")
print(f"Best fitness: {summary['final_best_fitness']:.4f}")
print(f"Convergence at generation: {summary['convergence_generation']}")

# Get best kernel
best = get_best_kernel(generations)
print(f"Best kernel ID: {best.genome.id}")
print(f"Best fitness: {best.genome.fitness:.4f}")
```

### Example 3: Lineage Tracking

```python
from src.ontogenesis import self_generate, initialize_ontogenetic_kernel

# Create ancestor
current = initialize_ontogenetic_kernel(base_kernel)
lineage = [current]

# Generate 10 generations
for i in range(10):
    current = self_generate(current)
    lineage.append(current)

# Trace lineage
for i, kernel in enumerate(lineage):
    print(f"Gen {i}: {kernel.genome.id}")
    print(f"  Fitness: {kernel.genome.fitness:.4f}")
    print(f"  Stage: {kernel.ontogenetic_state.stage.value}")
    print(f"  Lineage: {kernel.genome.lineage}")
```

### Example 4: Custom Fitness Function

```python
from src.ontogenesis import OntogenesisConfig, run_ontogenesis

# Define custom fitness
def custom_fitness(kernel):
    # Prefer kernels with specific properties
    coefficient_sum = np.sum(np.abs(kernel.coefficients))
    coefficient_spread = np.std(kernel.coefficients)
    
    return (
        kernel.grip_metrics.total_grip * 0.5 +
        min(1.0, coefficient_sum / 5.0) * 0.3 +
        min(1.0, coefficient_spread) * 0.2
    )

config = OntogenesisConfig(
    evolution=EvolutionConfig(
        population_size=20,
        max_generations=50,
    ),
    seed_kernels=seed_population,
    fitness_function=custom_fitness,  # Custom fitness
)

generations = run_ontogenesis(config)
```

## Mathematical Foundation

### B-Series as Genetic Code

The B-series expansion serves as the genetic code:

```
y_{n+1} = y_n + h * Σ b_i * Φ_i(f, y_n)
```

Where:
- `b_i` are the coefficient genes
- `Φ_i` are elementary differentials (rooted trees)
- Trees follow A000081 sequence: 1, 1, 2, 4, 9, 20, 48, 115, ...

### Differential Operators as Reproduction

Kernels reproduce through differential operators:

1. **Chain Rule** (Self-Composition):
   ```
   (f∘g)' = f'(g(x)) · g'(x)
   ```

2. **Product Rule** (Combination):
   ```
   (f·g)' = f'·g + f·g'
   ```

3. **Quotient Rule** (Refinement):
   ```
   (f/g)' = (f'·g - f·g')/g²
   ```

### Grip as Fitness Function

Grip measures how well the kernel's differential structure matches the domain:

```
grip = optimal_contact ∩ domain_topology
```

Perfect grip → Perfect computation

## Performance Characteristics

### Complexity

- **Initialization**: O(n) where n = coefficient count
- **Self-Generation**: O(n²) (operator application)
- **Self-Optimization**: O(k·n) where k = iterations
- **Crossover**: O(n)
- **Mutation**: O(1)
- **Evolution**: O(g·p·n) where g = generations, p = population size

### Memory

- **Kernel**: ~1KB (genome + state)
- **Population**: p × 1KB
- **History**: Limited by generation tracking

### Convergence

Typical evolution converges in 20-50 generations with:
- Population size: 20-50
- Mutation rate: 0.1-0.2
- Crossover rate: 0.7-0.9

## Advanced Features

### Development Schedule

Control stage transitions:

```python
# Stages advance based on maturity thresholds:
# Embryonic → Juvenile: maturity >= 0.3
# Juvenile → Mature: maturity >= 0.7
# Mature → Senescent: maturity >= 0.95

kernel.ontogenetic_state.maturity = 0.8
kernel.advance_stage()  # Will advance to Mature if currently Juvenile
```

### Diversity Pressure

Maintain genetic diversity through novelty scoring:

```python
config = OntogenesisConfig(
    evolution=EvolutionConfig(
        diversity_pressure=0.2,  # Higher = more diverse population
        ...
    ),
    ...
)
```

### Elite Preservation

Keep best individuals across generations:

```python
config = OntogenesisConfig(
    evolution=EvolutionConfig(
        elitism_rate=0.2,  # Keep top 20%
        ...
    ),
    ...
)
```

## Testing

The ontogenesis module includes comprehensive tests:

```bash
# Run all ontogenesis tests
python -m pytest src/tests/test_ontogenesis/ -v

# Run specific test module
python -m pytest src/tests/test_ontogenesis/test_kernel.py -v
python -m pytest src/tests/test_ontogenesis/test_genome.py -v
python -m pytest src/tests/test_ontogenesis/test_operators.py -v
python -m pytest src/tests/test_ontogenesis/test_evolution.py -v
```

Test coverage:
- **46 tests** covering all functionality
- **Kernel operations**: 9 tests
- **Genetic operations**: 15 tests
- **Differential operators**: 9 tests
- **Evolution**: 13 tests

## API Reference

### Classes

- `GeneratedKernel`: Base kernel with B-series coefficients
- `OntogeneticKernel`: Enhanced kernel with genome and development state
- `KernelGenome`: Genetic information container
- `KernelGene`: Base class for genes
- `CoefficientGene`: Mutable coefficient gene
- `OperatorGene`: Mutable operator gene
- `SymmetryGene`: Immutable symmetry gene
- `PreservationGene`: Immutable preservation gene
- `DevelopmentStage`: Enum for life stages
- `GripMetrics`: Domain-fit metrics
- `OntogeneticState`: Development state tracker

### Functions

- `initialize_ontogenetic_kernel(kernel)`: Convert to ontogenetic kernel
- `self_generate(parent)`: Generate offspring via self-composition
- `self_optimize(kernel, iterations)`: Optimize grip iteratively
- `self_reproduce(parent1, parent2, method, mutation_rate)`: Combine two kernels
- `run_ontogenesis(config)`: Run evolutionary algorithm
- `get_best_kernel(generations)`: Extract best kernel from history
- `summarize_evolution(generations)`: Generate evolution summary
- `evaluate_fitness(kernel, population)`: Calculate fitness score
- `tournament_selection(population, tournament_size)`: Select via tournament
- `calculate_diversity(population)`: Measure genetic diversity

### Configuration

- `EvolutionConfig`: Evolution parameters
  - `population_size`: Number of individuals
  - `mutation_rate`: Probability of mutation
  - `crossover_rate`: Probability of crossover
  - `elitism_rate`: Fraction of elite to preserve
  - `max_generations`: Maximum generations
  - `fitness_threshold`: Early stopping threshold
  - `diversity_pressure`: Diversity bonus weight

- `OntogenesisConfig`: Complete configuration
  - `evolution`: EvolutionConfig instance
  - `seed_kernels`: Initial population
  - `fitness_function`: Optional custom fitness

## Integration with CogPrime

The ontogenesis module integrates seamlessly with CogPrime's cognitive architecture:

```python
# Import both systems
from src import ontogenesis
from src.core.cognitive_core import CogPrimeCore

# Use ontogenesis to evolve cognitive kernels
# (Integration example - requires torch)
```

## References

- Butcher, J.C. (2016). *Numerical Methods for Ordinary Differential Equations*
- Hairer, E., Nørsett, S.P., Wanner, G. (1993). *Solving Ordinary Differential Equations I*
- Holland, J.H. (1992). *Adaptation in Natural and Artificial Systems*
- von Neumann, J. (1966). *Theory of Self-Reproducing Automata*
- Cayley, A. (1857). *On the Theory of the Analytical Forms called Trees* (A000081)

## License

MIT License - see [LICENSE](../../LICENSE) for details.

---

**Ontogenesis**: Where mathematics becomes life, and kernels evolve themselves through the pure language of differential calculus.

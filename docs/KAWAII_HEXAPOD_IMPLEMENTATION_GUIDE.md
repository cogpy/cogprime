# Kawaii Hexapod System 4: Implementation Guide & API Reference

## Component Interface Definition

### Core Props and State Interfaces

```typescript
interface Position {
  x: number;
  y: number; 
  rotation: number;
}

interface Obstacle {
  x: number;
  y: number;
  size: number;
  distance?: number; // Calculated distance from hexapod
}

interface SetConfig {
  term: 1 | 2 | 4 | 5 | 7 | 8;
  mode: 'E' | 'R'; // Expressive or Regenerative
}

type EmotionalState = 'happy' | 'concerned' | 'curious';

interface TermMeanings {
  1: 'Perception (Need vs Capacity)';
  2: 'Idea Formation';
  4: 'Sensory Input';
  5: 'Action Sequence';
  7: 'Memory Encoding';
  8: 'Balanced Response';
}
```

### System 4 Configuration Matrix

The component implements a 12-step configuration cycle where three consciousness sets operate with phase shifts:

```typescript
const setConfigs: SetConfig[][] = [
  // Step 0: Set1=[T8E], Set2=[T7R], Set3=[T4E]
  [
    { term: 8, mode: 'E' },  // Set 1: Balanced Response (Expressive)
    { term: 7, mode: 'R' },  // Set 2: Memory Encoding (Regenerative)  
    { term: 4, mode: 'E' }   // Set 3: Sensory Input (Expressive)
  ],
  // Step 1: Set1=[T5E], Set2=[T1R], Set3=[T2E]
  [
    { term: 5, mode: 'E' },  // Set 1: Action Sequence (Expressive)
    { term: 1, mode: 'R' },  // Set 2: Perception (Regenerative)
    { term: 2, mode: 'E' }   // Set 3: Idea Formation (Expressive)
  ],
  // ... continuing for 12 total steps
];
```

## Cognitive Processing Algorithms

### Tensional Coupling Detection

```typescript
const detectTensionalCouplings = (sets: SetConfig[]): CouplingResult => {
  const couplings = {
    sensoryMemory: false,    // T4E ↔ T7R
    simulationPlanning: false, // T1R ↔ T2E  
    balancedResponse: false   // T8E
  };
  
  // T4E ↔ T7R: Sensory input coupled with memory recall
  couplings.sensoryMemory = sets.some(s => s.term === 4 && s.mode === 'E') &&
                           sets.some(s => s.term === 7 && s.mode === 'R');
  
  // T1R ↔ T2E: Simulation reconciling with reactive planning  
  couplings.simulationPlanning = sets.some(s => s.term === 1 && s.mode === 'R') &&
                                sets.some(s => s.term === 2 && s.mode === 'E');
  
  // T8E: Executing balanced motor response
  couplings.balancedResponse = sets.some(s => s.term === 8 && s.mode === 'E');
  
  return couplings;
};
```

### Obstacle Detection and Avoidance

```typescript
const findNearestObstacle = (
  hexapodPosition: Position, 
  obstacles: Obstacle[]
): Obstacle | null => {
  let nearest: Obstacle | null = null;
  let minDistance = Infinity;
  
  obstacles.forEach(obstacle => {
    const distance = Math.sqrt(
      Math.pow(obstacle.x - hexapodPosition.x, 2) + 
      Math.pow(obstacle.y - hexapodPosition.y, 2)
    );
    
    if (distance < minDistance) {
      minDistance = distance;
      nearest = { ...obstacle, distance };
    }
  });
  
  return nearest;
};

const planAvoidanceRoute = (
  hexapodPosition: Position,
  obstacle: Obstacle,
  setHexapodPosition: (pos: Position) => void
): void => {
  // Calculate angle from hexapod to obstacle
  const angleToObstacle = Math.atan2(
    obstacle.y - hexapodPosition.y,
    obstacle.x - hexapodPosition.x
  );
  
  const currentAngle = hexapodPosition.rotation * Math.PI / 180;
  const angleDifference = angleToObstacle - currentAngle;
  
  // Turn away from obstacle with damping factor
  const avoidanceRotation = hexapodPosition.rotation - 
                           (angleDifference * 180 / Math.PI) * 0.3;
  
  setHexapodPosition(prev => ({
    ...prev,
    rotation: avoidanceRotation
  }));
};
```

### Movement Execution with Boundary Wrapping

```typescript
const executeMovement = (
  hexapodPosition: Position,
  setHexapodPosition: (pos: Position) => void,
  speed: number = 2
): void => {
  const angle = hexapodPosition.rotation * Math.PI / 180;
  const deltaX = Math.cos(angle) * speed;
  const deltaY = Math.sin(angle) * speed;
  
  setHexapodPosition(prev => ({
    ...prev,
    x: ((prev.x + deltaX) + 600) % 600, // Wrap around canvas width
    y: ((prev.y + deltaY) + 400) % 400  // Wrap around canvas height
  }));
};
```

## Visualization Engine

### Canvas Rendering Pipeline

```typescript
const renderHexapod = (
  ctx: CanvasRenderingContext2D,
  hexapodPosition: Position,
  emotionalState: EmotionalState,
  currentStep: number,
  sets: SetConfig[]
): void => {
  ctx.save();
  
  // Transform to hexapod coordinate system
  ctx.translate(hexapodPosition.x, hexapodPosition.y);
  ctx.rotate(hexapodPosition.rotation * Math.PI / 180);
  
  // Render body with emotional state mapping
  renderBody(ctx, emotionalState);
  
  // Render legs with cognitive state indicators
  renderLegs(ctx, currentStep, sets);
  
  // Render face with emotional expressions
  renderFace(ctx, emotionalState);
  
  // Render antennae with term-based coloring
  renderAntennae(ctx, sets);
  
  ctx.restore();
};
```

### Leg Animation System

```typescript
const renderLegs = (
  ctx: CanvasRenderingContext2D,
  currentStep: number,
  sets: SetConfig[]
): void => {
  const legPhase = (currentStep % 6) * Math.PI / 3;
  
  for (let pairIndex = 0; pairIndex < 3; pairIndex++) {
    const pairOffset = pairIndex * 15 - 15; // Distribute pairs along body
    const setData = sets[pairIndex];
    const isActive = setData.term === 5 || setData.term === 8;
    
    // Color coding based on cognitive activity
    ctx.strokeStyle = isActive ? '#FF1493' : '#DDA0DD';
    ctx.lineWidth = 3;
    
    // Left leg with phase-shifted animation
    ctx.beginPath();
    ctx.moveTo(pairOffset, -20);
    const leftAngle = -Math.PI/3 + Math.sin(legPhase + pairIndex * 2) * 0.3;
    ctx.lineTo(
      pairOffset + Math.cos(leftAngle) * 25,
      -20 + Math.sin(leftAngle) * 25
    );
    ctx.stroke();
    
    // Right leg with inverted phase
    ctx.beginPath();
    ctx.moveTo(pairOffset, 20);
    const rightAngle = Math.PI/3 + Math.sin(legPhase + pairIndex * 2 + Math.PI) * 0.3;
    ctx.lineTo(
      pairOffset + Math.cos(rightAngle) * 25,
      20 + Math.sin(rightAngle) * 25
    );
    ctx.stroke();
  }
};
```

### Emotional Expression Mapping

```typescript
const renderFace = (
  ctx: CanvasRenderingContext2D,
  emotionalState: EmotionalState
): void => {
  // Dynamic eye size based on emotional state
  const eyeSize = {
    happy: 5,
    concerned: 7, 
    curious: 6
  }[emotionalState];
  
  // Render eyes with sparkles
  ctx.fillStyle = '#000';
  [-8, 8].forEach((xOffset, index) => {
    ctx.beginPath();
    ctx.arc(xOffset, -5, eyeSize, 0, 2 * Math.PI);
    ctx.fill();
    
    // Eye sparkles
    ctx.fillStyle = '#FFF';
    ctx.beginPath();
    ctx.arc(xOffset + 2, -7, 2, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#000';
  });
  
  // Emotional mouth expressions
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  switch (emotionalState) {
    case 'happy':
      ctx.arc(0, 0, 8, 0.2 * Math.PI, 0.8 * Math.PI); // Smile
      break;
    case 'concerned':
      ctx.arc(0, 8, 8, 1.2 * Math.PI, 1.8 * Math.PI); // Frown
      break;
    case 'curious':
      ctx.moveTo(-5, 5);
      ctx.lineTo(5, 5); // Straight line
      break;
  }
  ctx.stroke();
};
```

### Cognitive State Antennae

```typescript
const renderAntennae = (
  ctx: CanvasRenderingContext2D,
  sets: SetConfig[]
): void => {
  // Find dominant cognitive term for visualization
  const dominantSet = sets.find(s => 
    s.mode === 'E' && [4, 8].includes(s.term)
  ) || sets[0];
  
  // Color mapping for cognitive terms
  const termColors = {
    1: '#FFD700', // Gold - Perception
    2: '#FF69B4', // Hot Pink - Idea Formation  
    4: '#00CED1', // Dark Turquoise - Sensory Input
    5: '#32CD32', // Lime Green - Action Sequence
    7: '#9370DB', // Medium Purple - Memory Encoding
    8: '#FF6347'  // Tomato - Balanced Response
  };
  
  // Render antennae stalks
  ctx.strokeStyle = dominantSet.mode === 'E' ? '#FF69B4' : '#87CEEB';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(-5, -20);
  ctx.lineTo(-8, -35);
  ctx.moveTo(5, -20);
  ctx.lineTo(8, -35);
  ctx.stroke();
  
  // Render glowing tips indicating active cognitive terms
  ctx.fillStyle = termColors[dominantSet.term];
  ctx.beginPath();
  ctx.arc(-8, -35, 4, 0, 2 * Math.PI);
  ctx.arc(8, -35, 4, 0, 2 * Math.PI);
  ctx.fill();
};
```

## Performance Optimization Guidelines

### Efficient Canvas Rendering

```typescript
const optimizeCanvasRendering = (
  canvas: HTMLCanvasElement,
  devicePixelRatio: number = window.devicePixelRatio || 1
): void => {
  const ctx = canvas.getContext('2d');
  
  // Set up high-DPI canvas
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  
  // Enable optimizations
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
};
```

### Memory Management for Animation Loops

```typescript
const useOptimizedAnimationLoop = (
  isRunning: boolean,
  stepCallback: () => void,
  interval: number = 500
): void => {
  useEffect(() => {
    if (!isRunning) return;
    
    let animationId: number;
    let lastTime = 0;
    
    const animate = (currentTime: number) => {
      if (currentTime - lastTime >= interval) {
        stepCallback();
        lastTime = currentTime;
      }
      animationId = requestAnimationFrame(animate);
    };
    
    animationId = requestAnimationFrame(animate);
    
    return () => cancelAnimationFrame(animationId);
  }, [isRunning, stepCallback, interval]);
};
```

## Testing Strategy

### Unit Tests for Cognitive Functions

```typescript
describe('System 4 Cognitive Processing', () => {
  test('tensional coupling detection', () => {
    const sets: SetConfig[] = [
      { term: 4, mode: 'E' },
      { term: 7, mode: 'R' },
      { term: 2, mode: 'E' }
    ];
    
    const couplings = detectTensionalCouplings(sets);
    expect(couplings.sensoryMemory).toBe(true);
    expect(couplings.simulationPlanning).toBe(false);
  });
  
  test('obstacle avoidance calculation', () => {
    const hexapod: Position = { x: 100, y: 100, rotation: 0 };
    const obstacle: Obstacle = { x: 150, y: 100, size: 30 };
    
    const nearest = findNearestObstacle(hexapod, [obstacle]);
    expect(nearest?.distance).toBeCloseTo(50);
  });
  
  test('boundary wrapping movement', () => {
    const initialPos: Position = { x: 595, y: 395, rotation: 45 };
    
    executeMovement(initialPos, (newPos) => {
      expect(newPos.x).toBeLessThan(600);
      expect(newPos.y).toBeLessThan(400);
    });
  });
});
```

### Integration Tests for Canvas Rendering

```typescript
describe('Canvas Rendering Integration', () => {
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  
  beforeEach(() => {
    canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 400;
    ctx = canvas.getContext('2d')!;
  });
  
  test('hexapod renders without errors', () => {
    const position: Position = { x: 300, y: 200, rotation: 0 };
    const sets: SetConfig[] = [{ term: 8, mode: 'E' }];
    
    expect(() => {
      renderHexapod(ctx, position, 'happy', 0, sets);
    }).not.toThrow();
  });
  
  test('emotional state affects rendering', () => {
    const position: Position = { x: 300, y: 200, rotation: 0 };
    
    // Test different emotional states
    ['happy', 'concerned', 'curious'].forEach(emotion => {
      expect(() => {
        renderFace(ctx, emotion as EmotionalState);
      }).not.toThrow();
    });
  });
});
```

## Integration with Broader CogPrime Architecture

### GGML Tensor Mapping

```typescript
interface GGMLTensorMapping {
  // Map System 4 states to GGML tensor operations
  consciousnessSetToTensor: (sets: SetConfig[]) => Float32Array;
  
  // Convert hexapod position to tensor representation
  positionToTensor: (position: Position) => Float32Array;
  
  // Emotional state to one-hot encoding
  emotionalStateToTensor: (state: EmotionalState) => Float32Array;
  
  // Obstacle field to spatial tensor
  obstacleFieldToTensor: (obstacles: Obstacle[], width: number, height: number) => Float32Array;
}

const mapToGGMLTensors = (
  sets: SetConfig[],
  position: Position,
  emotionalState: EmotionalState,
  obstacles: Obstacle[]
): GGMLTensorMapping => {
  return {
    consciousnessSetToTensor: (sets) => {
      const tensor = new Float32Array(3 * 6 * 2); // [3 sets, 6 terms, 2 modes]
      sets.forEach((set, setIndex) => {
        const termIndex = [1,2,4,5,7,8].indexOf(set.term);
        const modeIndex = set.mode === 'E' ? 0 : 1;
        const flatIndex = setIndex * 12 + termIndex * 2 + modeIndex;
        tensor[flatIndex] = 1.0;
      });
      return tensor;
    },
    
    positionToTensor: (position) => {
      return new Float32Array([
        position.x / 600.0,  // Normalize to [0,1]
        position.y / 400.0,  // Normalize to [0,1] 
        position.rotation / 360.0 // Normalize to [0,1]
      ]);
    },
    
    emotionalStateToTensor: (state) => {
      const mapping = { happy: 0, concerned: 1, curious: 2 };
      const tensor = new Float32Array(3);
      tensor[mapping[state]] = 1.0;
      return tensor;
    },
    
    obstacleFieldToTensor: (obstacles, width, height) => {
      const tensor = new Float32Array(width * height);
      obstacles.forEach(obs => {
        const x = Math.floor(obs.x);
        const y = Math.floor(obs.y);
        if (x >= 0 && x < width && y >= 0 && y < height) {
          tensor[y * width + x] = obs.size / 50.0; // Normalized size
        }
      });
      return tensor;
    }
  };
};
```

### P9ML Membrane Integration

```typescript
interface P9MLMembraneInterface {
  // Map System 4 consciousness sets to P9ML membranes
  setToMembrane: (set: SetConfig) => MembraneDescriptor;
  
  // Generate grammar rules from cognitive transitions
  generateGrammarRules: (fromSet: SetConfig, toSet: SetConfig) => GrammarRule[];
  
  // Create namespace hierarchies for cognitive organization
  createCognitiveNamespace: (sets: SetConfig[]) => NamespaceHierarchy;
}

const integrateWithP9ML = (
  currentSets: SetConfig[],
  previousSets: SetConfig[]
): P9MLMembraneInterface => {
  return {
    setToMembrane: (set) => ({
      id: `membrane_T${set.term}${set.mode}`,
      type: set.mode === 'E' ? 'expressive' : 'regenerative',
      termSemantics: getTermSemantics(set.term),
      activationLevel: 1.0
    }),
    
    generateGrammarRules: (fromSet, toSet) => {
      return [{
        lhs: `T${fromSet.term}${fromSet.mode}`,
        rhs: `T${toSet.term}${toSet.mode}`,
        probability: calculateTransitionProbability(fromSet, toSet),
        semanticTransformation: deriveSemanticMapping(fromSet, toSet)
      }];
    },
    
    createCognitiveNamespace: (sets) => ({
      root: 'kawaii_hexapod_cognition',
      levels: {
        consciousness_sets: sets.map(s => `set_${s.term}_${s.mode}`),
        cognitive_terms: [1,2,4,5,7,8].map(t => `term_${t}`),
        processing_modes: ['expressive', 'regenerative']
      }
    })
  };
};
```

This implementation guide provides the detailed technical foundation for extending and integrating the kawaii-hexapod-system4 component within the broader CogPrime architecture.
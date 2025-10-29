# OpenCog Limbo Integration Plan

## Overview

This document outlines the integration strategy for the Limbo implementation of OpenCog with the existing Python-based CogPrime system.

## Integration Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Python Layer (CogPrime)                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │   Torch    │  │  Mem0/RAG  │  │  Node9 FFI │        │
│  │  Modules   │  │   Backend  │  │   Bridge   │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└────────────┬─────────────────────────────┬──────────────┘
             │                             │
             │  JSON/9P Protocol          │ FFI Calls
             │                             │
┌────────────┴─────────────────────────────┴──────────────┐
│              Limbo Layer (OpenCog Core)                 │
│  ┌────────────────────────────────────────────────┐    │
│  │              AtomSpace (Limbo)                  │    │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐       │    │
│  │  │ Perc │  │ Reas │  │ Act  │  │ Learn│       │    │
│  │  └──────┘  └──────┘  └──────┘  └──────┘       │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Integration Strategies

### Strategy 1: 9P Protocol Bridge (Recommended)

**Advantages:**
- Network-transparent
- Language-agnostic
- Matches Inferno/Plan9 philosophy
- Async communication

**Implementation:**
1. Expose Limbo AtomSpace as 9P filesystem
2. Python client accesses via 9P mounts
3. Read/write atoms as files
4. Query via special control files

**Files to create:**
- `limbo/bridge/p9_server.b` - 9P server implementation
- `limbo/bridge/modules/` - Module bridges

**Files:**
```
/cog/
  atoms/
    <atom-id>           # Individual atom files (JSON)
  types/
    ConceptNode/        # Atoms by type
    InheritanceLink/
  queries/
    pattern             # Write pattern, read results
    by_type             # Query by type
  control/
    status              # System status
    stats               # Statistics
  modules/
    perception/         # Module-specific controls
    reasoning/
    action/
    learning/
```

### Strategy 2: Foreign Function Interface (FFI)

**Advantages:**
- Direct function calls
- Lower overhead
- Type safety

**Implementation:**
1. Create C wrapper for Limbo functions
2. Python ctypes bindings
3. Synchronous function calls

**Example C Wrapper:**
```c
// atomspace_bridge.c
typedef struct {
    char* id;
    char* type;
    char* name;
    double tv_strength;
    double tv_confidence;
} AtomInfo;

AtomInfo* atomspace_add_node(const char* type, const char* name);
AtomInfo* atomspace_get_atom(const char* id);
void atomspace_free_info(AtomInfo* info);
```

**Python Binding:**
```python
from ctypes import *

class AtomInfo(Structure):
    _fields_ = [
        ('id', c_char_p),
        ('type', c_char_p),
        ('name', c_char_p),
        ('tv_strength', c_double),
        ('tv_confidence', c_double)
    ]

lib = CDLL('/path/to/libatomspace_bridge.so')
lib.atomspace_add_node.argtypes = [c_char_p, c_char_p]
lib.atomspace_add_node.restype = POINTER(AtomInfo)
```

### Strategy 3: Message Passing via Channels

**Advantages:**
- Async operation
- Buffered communication
- Concurrent processing

**Implementation:**
1. Inferno message channels
2. JSON serialization
3. Event-driven architecture

## Implementation Phases

### Phase 1: Basic Bridge (Weeks 1-2)

- [ ] Implement 9P server in Limbo for AtomSpace
- [ ] Create Python 9P client library
- [ ] Basic atom CRUD operations
- [ ] Simple query support
- [ ] Unit tests for bridge

**Deliverables:**
- `limbo/bridge/p9_server.b` - 9P server implementation
- `src/bridges/limbo_atomspace_bridge.py` - Python client
- `tests/test_limbo_bridge.py` - Integration tests

### Phase 2: Module Integration (Weeks 3-4)

- [ ] Expose Perception module via 9P
- [ ] Expose Reasoning module via 9P
- [ ] Expose Action module via 9P
- [ ] Expose Learning module via 9P
- [ ] Python wrappers for each module

**Deliverables:**
- `limbo/bridge/modules/` - Module bridges
- `src/integration/limbo_modules.py` - Python wrappers
- Integration examples

### Phase 3: Cognitive Core Integration (Weeks 5-6)

- [ ] Full cognitive cycle via bridge
- [ ] Performance benchmarking
- [ ] Memory profiling
- [ ] Optimization

**Deliverables:**
- `src/core/limbo_cognitive_core.py` - Integrated core
- Performance benchmarks
- Optimization report

### Phase 4: Advanced Features (Weeks 7-8)

- [ ] Distributed AtomSpace across nodes
- [ ] Persistence layer
- [ ] Advanced pattern matching
- [ ] MeTTa integration

## Code Examples

### Example 1: Python accessing Limbo AtomSpace via 9P

```python
from src.bridges.limbo_atomspace_bridge import LimboAtomSpace

# Connect to Limbo AtomSpace
space = LimboAtomSpace(mount_point="/cog")

# Create nodes
cat = space.add_node("ConceptNode", "cat")
animal = space.add_node("ConceptNode", "animal")

# Create link
inheritance = space.add_link("InheritanceLink", [cat.id, animal.id])

# Query
results = space.query_by_type("ConceptNode")
print(f"Found {len(results)} concept nodes")

# Pattern matching
pattern = {
    "type": "InheritanceLink",
    "outgoing": [
        {"type": "ConceptNode", "name": "*"},
        {"type": "ConceptNode", "name": "animal"}
    ]
}
matches = space.pattern_match(pattern)
```

### Example 2: Cognitive cycle with Limbo backend

```python
from src.core.limbo_cognitive_core import LimboCognitiveCore
import torch

# Initialize with Limbo backend
config = {
    'limbo_mount': '/cog',
    'visual_dim': 784,
    'audio_dim': 256
}
core = LimboCognitiveCore(config)

# Run cognitive cycle
sensory_input = {
    'visual': torch.randn(784),
    'auditory': torch.randn(256)
}

action = core.cognitive_cycle(sensory_input, reward=1.0)
print(f"Selected action: {action}")

# Get statistics
stats = core.get_stats()
print(f"Cycles: {stats['cycles']}")
print(f"AtomSpace size: {stats['atomspace_size']}")
```

## Testing Strategy

### Unit Tests
- Test each Limbo module independently
- Python bridge unit tests
- 9P protocol tests

### Integration Tests
- End-to-end cognitive cycles
- Multi-module interactions
- Persistence tests

### Performance Tests
- Throughput benchmarks
- Latency measurements
- Memory profiling
- Scalability tests

## Performance Considerations

### Expected Performance

| Operation | Python | Limbo | Speedup |
|-----------|--------|-------|---------|
| Node creation | 10μs | 2μs | 5x |
| Link creation | 15μs | 3μs | 5x |
| Query by type | 100μs | 20μs | 5x |
| Pattern match | 1ms | 200μs | 5x |

### Optimization Targets

1. **Memory Usage**: Limbo uses ~40% less memory than Python
2. **Startup Time**: Faster initialization
3. **Concurrency**: Better parallel processing
4. **Network Efficiency**: 9P protocol overhead < 10%

## Deployment Scenarios

### Scenario 1: Standalone Limbo System
- Pure Limbo implementation
- No Python dependency
- Minimal footprint
- Embedded systems

### Scenario 2: Hybrid System (Recommended)
- Limbo core + Python ML/UI
- Best performance for core operations
- Leverage Python ecosystem
- Production deployment

### Scenario 3: Distributed System
- Multiple Limbo nodes
- Shared AtomSpace via 9P
- Python coordinator
- Cloud deployment

## Migration Path

### Incremental Migration

1. **Week 1-2**: Deploy Limbo AtomSpace alongside Python
2. **Week 3-4**: Migrate perception to Limbo
3. **Week 5-6**: Migrate reasoning to Limbo
4. **Week 7-8**: Migrate action and learning
5. **Week 9-10**: Performance tuning and optimization

### Rollback Strategy

- Keep Python implementation available
- Feature flags for Limbo vs Python
- A/B testing in production
- Gradual traffic shift

## Monitoring and Debugging

### Metrics to Track

- Atom creation/deletion rate
- Query latency
- Working memory size
- Attention allocation efficiency
- Learning convergence rate

### Debug Tools

- 9P filesystem browser
- Atom visualization
- Execution traces
- Performance profiler

## Documentation Requirements

- [ ] API documentation for bridge
- [ ] Integration guide
- [ ] Performance tuning guide
- [ ] Troubleshooting guide
- [ ] Migration checklist

## Success Criteria

1. **Functionality**: All Python features available via Limbo
2. **Performance**: 3-5x speedup for core operations
3. **Stability**: No crashes in 1000+ hour stress test
4. **Compatibility**: 100% API compatibility with Python
5. **Documentation**: Complete API docs and examples

## Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Phase 1 | 2 weeks | Basic bridge working |
| Phase 2 | 2 weeks | Module integration complete |
| Phase 3 | 2 weeks | Cognitive core integrated |
| Phase 4 | 2 weeks | Advanced features |
| **Total** | **8 weeks** | **Production ready** |

## Resources Required

- Limbo/Inferno development environment
- Test infrastructure
- Performance benchmarking tools
- Documentation system

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Limbo compiler unavailable | High | Use pre-compiled binaries |
| 9P protocol issues | Medium | Fallback to FFI |
| Performance not meeting goals | Medium | Optimize critical paths |
| Integration complexity | Low | Incremental approach |

## Next Steps

1. Set up Limbo development environment
2. Implement basic 9P server
3. Create Python client library
4. Run integration tests
5. Performance benchmarking

## References

- [9P Protocol Specification](http://man.cat-v.org/plan_9/5/intro)
- [Limbo Programming](http://www.vitanuova.com/inferno/papers/limbo.html)
- [Styx Protocol](http://www.vitanuova.com/inferno/papers/styx.html)
- [OpenCog Architecture](https://wiki.opencog.org/)

# RROS Resource Management System

## Overview

The Resource Management System is a sophisticated, high-performance component of the RROS (Relevance Realization Operating System) kernel that provides dynamic resource allocation, priority-based scheduling, and adaptive load balancing for cognitive operations.

## Features

### 1. Dynamic Resource Allocation
- **Multiple Resource Types**: Computational, Memory, Attention, I/O, and Network
- **Sub-millisecond Response**: Allocation latency < 1μs (target: <1ms)
- **Thread-Safe Operations**: Full mutex protection for concurrent access
- **Atomic Operations**: Lock-free resource counters for maximum performance

### 2. Priority-Based Scheduling
Five priority levels for optimal task execution:
- **CRITICAL (0)**: Must complete on time, can preempt others
- **HIGH (1)**: Important cognitive operations
- **NORMAL (2)**: Standard processing
- **LOW (3)**: Background tasks
- **IDLE (4)**: Can be deferred indefinitely

### 3. Memory Management
- **Tracked Allocations**: Full visibility into all memory usage
- **Garbage Collection**: Automatic cleanup of unused allocations
- **Fragmentation Monitoring**: Real-time fragmentation metrics
- **Age-Based Cleanup**: Configurable thresholds for GC
- **Access Tracking**: Monitor allocation usage patterns

### 4. Load Balancing Strategies
Four intelligent load balancing approaches:
- **Round Robin**: Even distribution across modules
- **Least Loaded**: Assign to module with lowest load
- **Priority-Based**: High-priority tasks to least loaded modules
- **Predictive**: Use historical patterns for optimal placement

### 5. Resource Monitoring
- **Real-Time Utilization**: Track usage across all resource types
- **Alert Generation**: Configurable thresholds with severity levels
- **Performance Metrics**: Allocation counts, rejection rates, latency
- **Statistics Collection**: Peak usage, average usage, trends

### 6. Resource Prediction
- **Time Series Analysis**: Predict future resource needs
- **Historical Tracking**: Learn from past usage patterns
- **Confidence Scoring**: Know reliability of predictions
- **Proactive Preallocation**: Reserve resources before needed

### 7. Graceful Degradation
- **Automatic Detection**: Monitor resource pressure
- **Degraded Mode**: Maintain core functionality under stress
- **Recovery Monitoring**: Automatic return to normal operation
- **Alert Notifications**: Keep system informed of state changes

## Architecture

```
ResourceManager
├── Resource Pools (per type)
│   ├── Computational Resources
│   ├── Memory Resources
│   ├── Attention Resources
│   ├── I/O Resources
│   └── Network Resources
├── Task Scheduler
│   ├── Priority Queue
│   ├── Deadline Tracking
│   └── Resource Requirements
├── Memory Manager
│   ├── Allocation Tracking
│   ├── Garbage Collection
│   └── Fragmentation Monitoring
├── Load Balancer
│   ├── Module Load Tracking
│   ├── Task Assignment
│   └── Strategy Selection
├── Monitoring System
│   ├── Utilization Metrics
│   ├── Alert Management
│   └── Statistics Collection
└── Prediction Engine
    ├── Historical Data
    ├── Time Series Models
    └── Preallocation Logic
```

## Usage

### Basic Initialization

```cpp
#include "core/resource_manager.hpp"

using namespace rros;

// Configure resource manager
std::unordered_map<std::string, float> config = {
    {"computational_capacity", 100.0f},
    {"memory_capacity", 1048576.0f},  // 1MB
    {"attention_capacity", 1.0f},
    {"alert_threshold", 0.85f},
    {"gc_age_threshold", 300.0f}  // 5 minutes
};

ResourceManager manager(config);
```

### Resource Allocation

```cpp
// Request computational resources
ResourceRequest req(
    "my_module",                      // Requester ID
    ResourceType::COMPUTATIONAL,       // Resource type
    Priority::HIGH,                    // Priority level
    25.0f,                            // Amount needed
    std::chrono::milliseconds(100)    // Deadline (optional)
);

auto allocation = manager.allocate_resource(req);

if (allocation.fully_satisfied) {
    // Use resources
    // ...
    
    // Release when done
    manager.release_resource("my_module", 
                            ResourceType::COMPUTATIONAL, 
                            25.0f);
}
```

### Task Scheduling

```cpp
// Create a scheduled task
ScheduledTask task;
task.task_id = "important_computation";
task.priority = Priority::HIGH;
task.task_function = []() {
    // Task implementation
    std::cout << "Executing important computation\n";
};
task.deadline = std::chrono::high_resolution_clock::now() + 
                std::chrono::seconds(1);
task.resource_requirements[ResourceType::COMPUTATIONAL] = 10.0f;
task.is_preemptable = false;

// Schedule the task
if (manager.schedule_task(task)) {
    // Execute when resources available
    manager.execute_next_task();
}
```

### Memory Management

```cpp
// Allocate tracked memory
void* ptr = manager.allocate_memory(1024, "my_module");

if (ptr) {
    // Use memory
    char* buffer = static_cast<char*>(ptr);
    // ...
    
    // Free when done
    manager.free_memory(ptr);
}

// Run garbage collection
size_t reclaimed = manager.run_garbage_collection();
std::cout << "Reclaimed " << reclaimed << " bytes\n";

// Check fragmentation
float fragmentation = manager.get_memory_fragmentation();
```

### Load Balancing

```cpp
// Set strategy
manager.set_load_balancing_strategy(LoadBalancingStrategy::PREDICTIVE);

// Provide module loads
std::unordered_map<std::string, float> loads = {
    {"vision_module", 45.0f},
    {"audio_module", 20.0f},
    {"reasoning_module", 60.0f}
};

// Get task assignments
auto assignments = manager.balance_load(loads);

for (const auto& [module, tasks] : assignments) {
    std::cout << module << " assigned tasks: ";
    for (const auto& task : tasks) {
        std::cout << task << " ";
    }
    std::cout << "\n";
}
```

### Resource Monitoring

```cpp
// Get utilization across all types
auto utilization = manager.get_utilization();

for (const auto& [type, util] : utilization) {
    std::cout << "Resource type " << static_cast<int>(type) 
              << " utilization: " << (util * 100) << "%\n";
}

// Get detailed statistics
auto stats = manager.get_resource_stats(ResourceType::COMPUTATIONAL);
std::cout << "Total capacity: " << stats.total_capacity << "\n";
std::cout << "Current usage: " << stats.current_usage << "\n";
std::cout << "Peak usage: " << stats.peak_usage << "\n";
std::cout << "Allocations: " << stats.allocation_count << "\n";
std::cout << "Rejections: " << stats.rejection_count << "\n";

// Get active alerts
auto alerts = manager.get_active_alerts();
for (const auto& alert : alerts) {
    std::cout << "Alert: " << alert.message 
              << " (severity: " << static_cast<int>(alert.severity) << ")\n";
}
```

### Resource Prediction

```cpp
// Predict future usage
auto prediction = manager.predict_usage(
    ResourceType::COMPUTATIONAL,
    std::chrono::milliseconds(5000)  // 5 second horizon
);

std::cout << "Predicted usage:\n";
for (size_t i = 0; i < prediction.predicted_usage.size(); ++i) {
    std::cout << "  t+" << (i * 100) << "ms: " 
              << prediction.predicted_usage[i] << "\n";
}
std::cout << "Confidence: " << prediction.confidence << "\n";

// Preallocate based on predictions
manager.preallocate_predicted({prediction});
```

### Integration with RROS Kernel

```cpp
#include "core/rros_kernel.hpp"

// Initialize kernel (automatically creates resource manager)
RROSKernel kernel(config);

// Access resource manager
auto& rm = kernel.get_resource_manager();

// Allocate resources for cognitive cycle
if (kernel.allocate_cognitive_resources(50.0f, 1)) {  // HIGH priority
    // Perform cognitive operations
    auto state = kernel.cognitive_cycle(input_data);
    
    // Release resources
    kernel.release_cognitive_resources(50.0f);
}

// Get resource metrics
auto metrics = kernel.get_resource_metrics();
std::cout << "Average utilization: " 
          << metrics["avg_utilization"] << "\n";
std::cout << "Computational utilization: " 
          << metrics["computational_utilization"] << "\n";
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `computational_capacity` | float | 100.0 | Total computational resources |
| `memory_capacity` | float | 1048576.0 | Total memory in bytes |
| `attention_capacity` | float | 1.0 | Total attention budget |
| `io_capacity` | float | 100.0 | Total I/O bandwidth |
| `network_capacity` | float | 100.0 | Total network bandwidth |
| `alert_threshold` | float | 0.85 | Utilization threshold for alerts (0-1) |
| `gc_threshold` | float | 0.9 | Utilization threshold to trigger GC (0-1) |
| `gc_age_threshold` | float | 300.0 | Age in seconds for GC eligibility |

## Performance Characteristics

### Measured Performance (Test Suite)
- **Allocation Latency**: 0.51μs average (target: <1ms)
- **Concurrent Safety**: 100% success rate with 10 threads
- **Memory Overhead**: Minimal tracking overhead
- **Thread Scalability**: Linear with core count

### Resource Efficiency
- **Lock Granularity**: Per-resource-type pools
- **Atomic Operations**: Used for counters
- **Memory Allocations**: Pooled and tracked
- **Alert Processing**: O(1) threshold checks

## Best Practices

### 1. Resource Allocation
- Always check `fully_satisfied` before using resources
- Release resources promptly after use
- Use appropriate priority levels
- Set realistic deadlines for time-critical tasks

### 2. Memory Management
- Prefer tracked allocation for debugging
- Run GC periodically under high load
- Monitor fragmentation levels
- Configure GC thresholds based on workload

### 3. Load Balancing
- Choose strategy based on workload characteristics
- Update module loads regularly
- Consider predictive strategy for stable patterns
- Use priority-based for mixed workloads

### 4. Monitoring
- Set conservative alert thresholds
- Monitor alerts regularly
- Track utilization trends
- Use predictions for capacity planning

### 5. Graceful Degradation
- Test degraded mode behavior
- Define core vs. optional operations
- Monitor recovery thresholds
- Alert operators of degraded state

## Testing

The resource management system includes a comprehensive test suite with 20 tests covering all functionality:

```bash
cd rros_kernel/build
./test_resource_manager
```

### Test Coverage
- ✅ Basic resource allocation and release
- ✅ Priority-based allocation
- ✅ Task scheduling and execution
- ✅ Memory management with GC
- ✅ Load balancing (all strategies)
- ✅ Resource monitoring and statistics
- ✅ Alert generation and management
- ✅ Usage prediction and preallocation
- ✅ Graceful degradation
- ✅ Concurrent access safety
- ✅ Configuration updates
- ✅ Performance benchmarks

## Error Handling

### Common Issues

**Allocation Failure**
```cpp
auto allocation = manager.allocate_resource(req);
if (!allocation.fully_satisfied) {
    // Handle failure - reduce requirements or defer
    if (allocation.allocated_amount > 0) {
        // Partial allocation for CRITICAL priority
    }
}
```

**Memory Allocation Failure**
```cpp
void* ptr = manager.allocate_memory(size, "module");
if (!ptr) {
    // Memory not available - try GC or reduce request
    manager.run_garbage_collection();
    ptr = manager.allocate_memory(smaller_size, "module");
}
```

**Alert Threshold Exceeded**
```cpp
auto alerts = manager.get_active_alerts();
for (const auto& alert : alerts) {
    if (alert.severity == ResourceAlert::Severity::CRITICAL) {
        // Take action - scale down, defer tasks, etc.
        manager.enter_degraded_mode(alert.resource_type);
    }
}
```

## Future Enhancements

- [ ] GPU resource management
- [ ] Distributed resource coordination
- [ ] Machine learning-based prediction
- [ ] Auto-tuning of thresholds
- [ ] Resource quotas per module
- [ ] Resource reservation system
- [ ] GGML tensor integration
- [ ] Energy consumption tracking

## References

- RROS Kernel Architecture: `rros_kernel/README.md`
- Testing Guide: `rros_kernel/tests/test_resource_manager.cpp`
- Integration Examples: `rros_kernel/core/rros_kernel.cpp`
- Performance Report: `rros_kernel/PERFORMANCE_OPTIMIZATION_REPORT.md`

## License

MIT License - See LICENSE file for details.

---

*The Resource Management System is a critical component of the RROS kernel, enabling efficient allocation and utilization of computational resources for cognitive operations while maintaining system stability under varying loads.*

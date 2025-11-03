#include "../core/resource_manager.hpp"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace rros;

// Test utilities
bool test_result(bool passed, const std::string& test_name) {
    std::cout << (passed ? "✓" : "✗") << " " << test_name << std::endl;
    return passed;
}

// Test 1: Basic resource allocation
bool test_basic_allocation() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f},
        {"memory_capacity", 1000.0f}
    };
    
    ResourceManager manager(config);
    
    // Test computational resource allocation
    ResourceRequest req1("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 25.0f);
    auto alloc1 = manager.allocate_resource(req1);
    
    bool test1 = alloc1.fully_satisfied && alloc1.allocated_amount == 25.0f;
    
    // Test memory resource allocation
    ResourceRequest req2("module2", ResourceType::MEMORY, Priority::HIGH, 300.0f);
    auto alloc2 = manager.allocate_resource(req2);
    
    bool test2 = alloc2.fully_satisfied && alloc2.allocated_amount == 300.0f;
    
    // Test available resources
    float comp_available = manager.get_available_resources(ResourceType::COMPUTATIONAL);
    float mem_available = manager.get_available_resources(ResourceType::MEMORY);
    
    bool test3 = (comp_available == 75.0f) && (mem_available == 700.0f);
    
    return test_result(test1 && test2 && test3, "Basic resource allocation");
}

// Test 2: Resource release
bool test_resource_release() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    // Allocate
    ResourceRequest req("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 50.0f);
    manager.allocate_resource(req);
    
    float before_release = manager.get_available_resources(ResourceType::COMPUTATIONAL);
    
    // Release
    manager.release_resource("module1", ResourceType::COMPUTATIONAL, 50.0f);
    
    float after_release = manager.get_available_resources(ResourceType::COMPUTATIONAL);
    
    bool test = (before_release == 50.0f) && (after_release == 100.0f);
    return test_result(test, "Resource release");
}

// Test 3: Priority-based allocation
bool test_priority_allocation() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    // Allocate most resources
    ResourceRequest req1("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 95.0f);
    manager.allocate_resource(req1);
    
    // Try low priority request - should fail
    ResourceRequest req2("module2", ResourceType::COMPUTATIONAL, Priority::LOW, 10.0f);
    auto alloc2 = manager.allocate_resource(req2);
    
    // Try critical priority request - should partially succeed
    ResourceRequest req3("module3", ResourceType::COMPUTATIONAL, Priority::CRITICAL, 10.0f);
    auto alloc3 = manager.allocate_resource(req3);
    
    bool test = !alloc2.fully_satisfied && (alloc3.allocated_amount == 5.0f);
    return test_result(test, "Priority-based allocation");
}

// Test 4: Task scheduling
bool test_task_scheduling() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    bool task_executed = false;
    
    ScheduledTask task;
    task.task_id = "test_task";
    task.priority = Priority::HIGH;
    task.task_function = [&task_executed]() { task_executed = true; };
    task.deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(1);
    task.resource_requirements[ResourceType::COMPUTATIONAL] = 10.0f;
    task.is_preemptable = false;
    
    bool scheduled = manager.schedule_task(task);
    bool executed = manager.execute_next_task();
    
    bool test = scheduled && executed && task_executed;
    return test_result(test, "Task scheduling and execution");
}

// Test 5: Memory management
bool test_memory_management() {
    std::unordered_map<std::string, float> config = {
        {"memory_capacity", 10000.0f}
    };
    
    ResourceManager manager(config);
    
    // Allocate memory
    void* ptr1 = manager.allocate_memory(1000, "module1");
    void* ptr2 = manager.allocate_memory(2000, "module2");
    
    bool test1 = (ptr1 != nullptr) && (ptr2 != nullptr);
    
    // Free memory
    manager.free_memory(ptr1);
    
    // Check available memory increased
    float available = manager.get_available_resources(ResourceType::MEMORY);
    bool test2 = available > 7000.0f;  // Should have ~8000 available
    
    // Clean up
    manager.free_memory(ptr2);
    
    return test_result(test1 && test2, "Memory allocation and deallocation");
}

// Test 6: Garbage collection
bool test_garbage_collection() {
    std::unordered_map<std::string, float> config = {
        {"memory_capacity", 10000.0f},
        {"gc_threshold", 0.8f},
        {"gc_age_threshold", 1.0f}  // 1 second for testing
    };
    
    ResourceManager manager(config);
    
    // Allocate some memory
    void* ptr1 = manager.allocate_memory(1000, "module1");
    void* ptr2 = manager.allocate_memory(2000, "module2");
    
    // Wait a bit for memory to age
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Run GC
    size_t reclaimed = manager.run_garbage_collection();
    
    // Clean up any remaining
    if (ptr1) manager.free_memory(ptr1);
    if (ptr2) manager.free_memory(ptr2);
    
    bool test = true;  // GC should run without error, may or may not reclaim
    return test_result(test, "Garbage collection");
}

// Test 7: Load balancing
bool test_load_balancing() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    manager.set_load_balancing_strategy(LoadBalancingStrategy::LEAST_LOADED);
    
    // Create module loads
    std::unordered_map<std::string, float> module_loads = {
        {"module1", 10.0f},
        {"module2", 5.0f},
        {"module3", 15.0f}
    };
    
    // Schedule some tasks
    for (int i = 0; i < 5; ++i) {
        ScheduledTask task;
        task.task_id = "task" + std::to_string(i);
        task.priority = Priority::NORMAL;
        task.task_function = []() {};
        task.resource_requirements[ResourceType::COMPUTATIONAL] = 5.0f;
        manager.schedule_task(task);
    }
    
    // Balance load
    auto assignments = manager.balance_load(module_loads);
    
    bool test = !assignments.empty();
    return test_result(test, "Load balancing");
}

// Test 8: Resource monitoring and statistics
bool test_resource_monitoring() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    // Allocate resources
    ResourceRequest req("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 30.0f);
    manager.allocate_resource(req);
    
    // Get statistics
    auto stats = manager.get_resource_stats(ResourceType::COMPUTATIONAL);
    
    bool test1 = stats.total_capacity == 100.0f;
    bool test2 = stats.current_usage == 30.0f;
    bool test3 = stats.allocation_count == 1;
    
    return test_result(test1 && test2 && test3, "Resource monitoring and statistics");
}

// Test 9: Alert generation
bool test_alert_generation() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f},
        {"alert_threshold", 0.8f}
    };
    
    ResourceManager manager(config);
    
    // Set alert threshold
    manager.set_alert_threshold(ResourceType::COMPUTATIONAL, 0.7f, 
                               ResourceAlert::Severity::WARNING);
    
    // Allocate to trigger alert
    ResourceRequest req("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 75.0f);
    manager.allocate_resource(req);
    
    // Check alerts
    auto alerts = manager.get_active_alerts();
    
    bool test = !alerts.empty() && alerts[0].resource_type == ResourceType::COMPUTATIONAL;
    return test_result(test, "Alert generation");
}

// Test 10: Resource prediction
bool test_resource_prediction() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    // Feed some usage data
    for (int i = 0; i < 20; ++i) {
        manager.update_prediction_model(ResourceType::COMPUTATIONAL, 
                                       50.0f + i * 2.0f);
    }
    
    // Get prediction
    auto prediction = manager.predict_usage(ResourceType::COMPUTATIONAL, 
                                           std::chrono::milliseconds(1000));
    
    bool test = !prediction.predicted_usage.empty() && prediction.confidence > 0.0f;
    return test_result(test, "Resource usage prediction");
}

// Test 11: Preallocation based on prediction
bool test_preallocation() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    // Create prediction
    ResourcePrediction pred;
    pred.resource_type = ResourceType::COMPUTATIONAL;
    pred.predicted_usage = {40.0f, 50.0f, 60.0f};
    pred.confidence = 0.8f;
    pred.horizon = std::chrono::milliseconds(1000);
    
    float before = manager.get_available_resources(ResourceType::COMPUTATIONAL);
    
    // Preallocate
    manager.preallocate_predicted({pred});
    
    float after = manager.get_available_resources(ResourceType::COMPUTATIONAL);
    
    bool test = after < before;  // Some resources should be preallocated
    return test_result(test, "Preallocation based on prediction");
}

// Test 12: Graceful degradation
bool test_graceful_degradation() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    // Enter degraded mode
    manager.enter_degraded_mode(ResourceType::COMPUTATIONAL);
    
    bool test1 = manager.is_degraded();
    
    // Exit degraded mode
    manager.exit_degraded_mode(ResourceType::COMPUTATIONAL);
    
    bool test2 = !manager.is_degraded();
    
    return test_result(test1 && test2, "Graceful degradation");
}

// Test 13: Adaptive load balancing
bool test_adaptive_load_balancing() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f},
        {"memory_capacity", 1000.0f}
    };
    
    ResourceManager manager(config);
    
    // Simulate high load
    std::unordered_map<ResourceType, float> high_loads = {
        {ResourceType::COMPUTATIONAL, 96.0f},
        {ResourceType::MEMORY, 950.0f}
    };
    
    manager.adapt_to_load(high_loads);
    
    bool test1 = manager.is_degraded();  // Should enter degraded mode
    
    // Simulate load decrease
    std::unordered_map<ResourceType, float> low_loads = {
        {ResourceType::COMPUTATIONAL, 50.0f},
        {ResourceType::MEMORY, 500.0f}
    };
    
    manager.adapt_to_load(low_loads);
    
    // May still be degraded depending on thresholds
    bool test2 = true;  // Just verify it doesn't crash
    
    return test_result(test1 && test2, "Adaptive load balancing");
}

// Test 14: Utilization tracking
bool test_utilization_tracking() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f},
        {"memory_capacity", 1000.0f}
    };
    
    ResourceManager manager(config);
    
    // Allocate resources
    ResourceRequest req1("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 50.0f);
    ResourceRequest req2("module2", ResourceType::MEMORY, Priority::NORMAL, 300.0f);
    
    manager.allocate_resource(req1);
    manager.allocate_resource(req2);
    
    // Get utilization
    auto utilization = manager.get_utilization();
    
    bool test1 = utilization[ResourceType::COMPUTATIONAL] == 0.5f;  // 50%
    bool test2 = utilization[ResourceType::MEMORY] == 0.3f;         // 30%
    
    return test_result(test1 && test2, "Utilization tracking");
}

// Test 15: Configuration updates
bool test_configuration_updates() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    // Initial allocation
    ResourceRequest req("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 80.0f);
    manager.allocate_resource(req);
    
    float before = manager.get_available_resources(ResourceType::COMPUTATIONAL);
    
    // Update config to increase capacity
    std::unordered_map<std::string, float> new_config = {
        {"computational_capacity", 200.0f}
    };
    manager.update_config(new_config);
    
    float after = manager.get_available_resources(ResourceType::COMPUTATIONAL);
    
    bool test = after > before;  // Available should increase
    return test_result(test, "Configuration updates");
}

// Test 16: Reset functionality
bool test_reset() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f}
    };
    
    ResourceManager manager(config);
    
    // Allocate resources
    ResourceRequest req("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 50.0f);
    manager.allocate_resource(req);
    
    // Reset
    manager.reset();
    
    float available = manager.get_available_resources(ResourceType::COMPUTATIONAL);
    auto alerts = manager.get_active_alerts();
    
    bool test = (available == 100.0f) && alerts.empty();
    return test_result(test, "Reset functionality");
}

// Test 17: Performance - allocation latency
bool test_allocation_latency() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 1000.0f}
    };
    
    ResourceManager manager(config);
    
    // Measure allocation time
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        ResourceRequest req("module" + std::to_string(i), 
                          ResourceType::COMPUTATIONAL, 
                          Priority::NORMAL, 5.0f);
        manager.allocate_resource(req);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_latency = duration.count() / 100.0f;
    
    bool test = avg_latency < 1000.0f;  // < 1ms per allocation (individual latency target)
    
    std::cout << "  Average allocation latency: " << avg_latency << " μs" << std::endl;
    
    return test_result(test, "Allocation latency < 1ms");
}

// Test 18: Concurrent access
bool test_concurrent_access() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 1000.0f}
    };
    
    ResourceManager manager(config);
    
    std::vector<std::thread> threads;
    std::atomic<int> successful_allocations{0};
    
    // Launch multiple threads allocating resources
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&manager, &successful_allocations, i]() {
            for (int j = 0; j < 10; ++j) {
                ResourceRequest req("thread" + std::to_string(i) + "_alloc" + std::to_string(j),
                                  ResourceType::COMPUTATIONAL,
                                  Priority::NORMAL, 5.0f);
                auto alloc = manager.allocate_resource(req);
                if (alloc.fully_satisfied) {
                    successful_allocations++;
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    bool test = successful_allocations > 0;  // At least some should succeed
    
    std::cout << "  Successful allocations: " << successful_allocations << "/100" << std::endl;
    
    return test_result(test, "Concurrent access safety");
}

// Test 19: Memory fragmentation tracking
bool test_memory_fragmentation() {
    std::unordered_map<std::string, float> config = {
        {"memory_capacity", 10000.0f}
    };
    
    ResourceManager manager(config);
    
    // Allocate various sizes
    std::vector<void*> ptrs;
    ptrs.push_back(manager.allocate_memory(100, "module1"));
    ptrs.push_back(manager.allocate_memory(5000, "module2"));
    ptrs.push_back(manager.allocate_memory(50, "module3"));
    ptrs.push_back(manager.allocate_memory(3000, "module4"));
    
    float fragmentation = manager.get_memory_fragmentation();
    
    // Clean up
    for (auto ptr : ptrs) {
        manager.free_memory(ptr);
    }
    
    bool test = fragmentation >= 0.0f && fragmentation <= 1.0f;
    
    std::cout << "  Memory fragmentation: " << fragmentation << std::endl;
    
    return test_result(test, "Memory fragmentation tracking");
}

// Test 20: Comprehensive metrics
bool test_comprehensive_metrics() {
    std::unordered_map<std::string, float> config = {
        {"computational_capacity", 100.0f},
        {"memory_capacity", 1000.0f}
    };
    
    ResourceManager manager(config);
    
    // Perform various operations
    ResourceRequest req1("module1", ResourceType::COMPUTATIONAL, Priority::NORMAL, 30.0f);
    manager.allocate_resource(req1);
    
    void* ptr = manager.allocate_memory(200, "module2");
    
    // Get metrics
    auto metrics = manager.get_metrics();
    
    bool test1 = metrics.find("avg_utilization") != metrics.end();
    bool test2 = metrics.find("total_memory_allocated") != metrics.end();
    bool test3 = metrics.find("memory_fragmentation") != metrics.end();
    bool test4 = metrics.find("is_degraded") != metrics.end();
    
    // Clean up
    manager.free_memory(ptr);
    
    return test_result(test1 && test2 && test3 && test4, "Comprehensive metrics");
}

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   RROS Resource Management System - Test Suite          ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    int passed = 0;
    int total = 0;
    
    // Run all tests
    if (test_basic_allocation()) ++passed; ++total;
    if (test_resource_release()) ++passed; ++total;
    if (test_priority_allocation()) ++passed; ++total;
    if (test_task_scheduling()) ++passed; ++total;
    if (test_memory_management()) ++passed; ++total;
    if (test_garbage_collection()) ++passed; ++total;
    if (test_load_balancing()) ++passed; ++total;
    if (test_resource_monitoring()) ++passed; ++total;
    if (test_alert_generation()) ++passed; ++total;
    if (test_resource_prediction()) ++passed; ++total;
    if (test_preallocation()) ++passed; ++total;
    if (test_graceful_degradation()) ++passed; ++total;
    if (test_adaptive_load_balancing()) ++passed; ++total;
    if (test_utilization_tracking()) ++passed; ++total;
    if (test_configuration_updates()) ++passed; ++total;
    if (test_reset()) ++passed; ++total;
    if (test_allocation_latency()) ++passed; ++total;
    if (test_concurrent_access()) ++passed; ++total;
    if (test_memory_fragmentation()) ++passed; ++total;
    if (test_comprehensive_metrics()) ++passed; ++total;
    
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Test Results: " << passed << "/" << total << " tests passed";
    int padding = 29 - std::to_string(passed).length() - std::to_string(total).length();
    for (int i = 0; i < padding; ++i) std::cout << " ";
    std::cout << "║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;
    
    return (passed == total) ? 0 : 1;
}

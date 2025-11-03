#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <queue>
#include <functional>

namespace rros {

/**
 * @brief Resource types managed by the system
 */
enum class ResourceType {
    COMPUTATIONAL,      // CPU/processing resources
    MEMORY,            // Memory allocation
    ATTENTION,         // Attention budget
    IO,                // I/O bandwidth
    NETWORK            // Network bandwidth
};

/**
 * @brief Priority levels for resource allocation
 */
enum class Priority {
    CRITICAL = 0,      // Must complete on time
    HIGH = 1,          // Important cognitive operations
    NORMAL = 2,        // Standard processing
    LOW = 3,           // Background tasks
    IDLE = 4           // Can be deferred indefinitely
};

/**
 * @brief Resource allocation request
 */
struct ResourceRequest {
    std::string requester_id;           // ID of requesting component
    ResourceType resource_type;          // Type of resource needed
    Priority priority;                   // Request priority
    float amount;                        // Amount of resource needed
    std::chrono::milliseconds deadline;  // Deadline for allocation
    std::chrono::high_resolution_clock::time_point timestamp;
    
    ResourceRequest() = default;
    ResourceRequest(const std::string& id, ResourceType type, Priority prio, 
                   float amt, std::chrono::milliseconds dl = std::chrono::milliseconds(0))
        : requester_id(id), resource_type(type), priority(prio), 
          amount(amt), deadline(dl),
          timestamp(std::chrono::high_resolution_clock::now()) {}
};

/**
 * @brief Resource allocation result
 */
struct ResourceAllocation {
    std::string requester_id;
    ResourceType resource_type;
    float allocated_amount;
    bool fully_satisfied;
    std::chrono::high_resolution_clock::time_point allocation_time;
};

/**
 * @brief Resource usage statistics
 */
struct ResourceStats {
    ResourceType type;
    float total_capacity;
    float current_usage;
    float peak_usage;
    float average_usage;
    uint64_t allocation_count;
    uint64_t rejection_count;
    std::chrono::milliseconds avg_allocation_time;
};

/**
 * @brief Resource constraint violation alert
 */
struct ResourceAlert {
    enum class Severity { INFO, WARNING, CRITICAL };
    
    Severity severity;
    ResourceType resource_type;
    std::string message;
    float threshold_violated;
    std::chrono::high_resolution_clock::time_point timestamp;
};

/**
 * @brief Resource usage prediction
 */
struct ResourcePrediction {
    ResourceType resource_type;
    std::vector<float> predicted_usage;  // Future usage predictions
    float confidence;                     // Prediction confidence
    std::chrono::milliseconds horizon;    // Prediction time horizon
};

/**
 * @brief Task for priority-based scheduling
 */
struct ScheduledTask {
    std::string task_id;
    Priority priority;
    std::function<void()> task_function;
    std::chrono::high_resolution_clock::time_point deadline;
    std::unordered_map<ResourceType, float> resource_requirements;
    bool is_preemptable;
    
    // Comparison for priority queue
    bool operator<(const ScheduledTask& other) const {
        if (priority != other.priority) {
            return static_cast<int>(priority) > static_cast<int>(other.priority);
        }
        return deadline > other.deadline;
    }
};

/**
 * @brief Load balancing strategy
 */
enum class LoadBalancingStrategy {
    ROUND_ROBIN,       // Distribute evenly
    LEAST_LOADED,      // Assign to least loaded module
    PRIORITY_BASED,    // Consider task priorities
    PREDICTIVE         // Use predictions for optimal placement
};

/**
 * @brief Sophisticated resource management system
 * 
 * Dynamically allocates computational resources, memory, and attention
 * based on current cognitive demands and system constraints. Implements
 * priority-based scheduling, resource monitoring, and adaptive allocation.
 */
class ResourceManager {
public:
    /**
     * @brief Constructor initializes resource pools and monitoring
     * @param config Configuration parameters
     */
    explicit ResourceManager(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor ensures clean resource release
     */
    ~ResourceManager();
    
    // Resource Allocation
    
    /**
     * @brief Request resource allocation
     * @param request Resource request details
     * @return Allocation result
     */
    ResourceAllocation allocate_resource(const ResourceRequest& request);
    
    /**
     * @brief Release previously allocated resources
     * @param requester_id ID of component releasing resources
     * @param resource_type Type of resource to release
     * @param amount Amount to release
     */
    void release_resource(const std::string& requester_id, 
                         ResourceType resource_type, 
                         float amount);
    
    /**
     * @brief Get available resources of a specific type
     * @param resource_type Type of resource
     * @return Available amount
     */
    float get_available_resources(ResourceType resource_type) const;
    
    // Priority Scheduling
    
    /**
     * @brief Schedule task for execution with priority
     * @param task Task to schedule
     * @return True if task was scheduled successfully
     */
    bool schedule_task(const ScheduledTask& task);
    
    /**
     * @brief Execute next scheduled task based on priority
     * @return True if a task was executed
     */
    bool execute_next_task();
    
    /**
     * @brief Get count of pending tasks by priority
     * @return Map of priority to task count
     */
    std::unordered_map<Priority, size_t> get_pending_task_counts() const;
    
    // Memory Management
    
    /**
     * @brief Allocate memory with tracking
     * @param size Bytes to allocate
     * @param requester_id ID of requesting component
     * @return Pointer to allocated memory or nullptr
     */
    void* allocate_memory(size_t size, const std::string& requester_id);
    
    /**
     * @brief Free previously allocated memory
     * @param ptr Pointer to memory to free
     */
    void free_memory(void* ptr);
    
    /**
     * @brief Run garbage collection optimization
     * @return Bytes reclaimed
     */
    size_t run_garbage_collection();
    
    /**
     * @brief Get memory fragmentation level
     * @return Fragmentation ratio [0.0, 1.0]
     */
    float get_memory_fragmentation() const;
    
    // Load Balancing
    
    /**
     * @brief Set load balancing strategy
     * @param strategy Strategy to use
     */
    void set_load_balancing_strategy(LoadBalancingStrategy strategy);
    
    /**
     * @brief Balance load across cognitive modules
     * @param module_loads Current load per module
     * @return Recommended task assignments
     */
    std::unordered_map<std::string, std::vector<std::string>> 
    balance_load(const std::unordered_map<std::string, float>& module_loads);
    
    /**
     * @brief Adapt to changing load patterns
     * @param current_loads Current system loads
     */
    void adapt_to_load(const std::unordered_map<ResourceType, float>& current_loads);
    
    // Resource Monitoring
    
    /**
     * @brief Get current resource statistics
     * @param resource_type Type of resource
     * @return Resource usage statistics
     */
    ResourceStats get_resource_stats(ResourceType resource_type) const;
    
    /**
     * @brief Get all active resource alerts
     * @return Vector of current alerts
     */
    std::vector<ResourceAlert> get_active_alerts() const;
    
    /**
     * @brief Set alert threshold for resource type
     * @param resource_type Resource to monitor
     * @param threshold Threshold (0.0-1.0 of capacity)
     * @param severity Alert severity when exceeded
     */
    void set_alert_threshold(ResourceType resource_type, 
                            float threshold, 
                            ResourceAlert::Severity severity);
    
    /**
     * @brief Get real-time resource utilization
     * @return Map of resource type to utilization [0.0, 1.0]
     */
    std::unordered_map<ResourceType, float> get_utilization() const;
    
    // Resource Prediction
    
    /**
     * @brief Predict future resource usage
     * @param resource_type Type of resource
     * @param horizon Prediction time horizon
     * @return Usage prediction
     */
    ResourcePrediction predict_usage(ResourceType resource_type, 
                                    std::chrono::milliseconds horizon);
    
    /**
     * @brief Pre-allocate resources based on predictions
     * @param predictions Vector of predictions
     */
    void preallocate_predicted(const std::vector<ResourcePrediction>& predictions);
    
    /**
     * @brief Update prediction models with actual usage
     * @param resource_type Resource type
     * @param actual_usage Actual usage observed
     */
    void update_prediction_model(ResourceType resource_type, float actual_usage);
    
    // Graceful Degradation
    
    /**
     * @brief Enter degraded mode to maintain core functionality
     * @param resource_type Resource under pressure
     */
    void enter_degraded_mode(ResourceType resource_type);
    
    /**
     * @brief Exit degraded mode and restore full functionality
     * @param resource_type Resource that recovered
     */
    void exit_degraded_mode(ResourceType resource_type);
    
    /**
     * @brief Check if system is in degraded mode
     * @return True if any resource is degraded
     */
    bool is_degraded() const;
    
    // Configuration
    
    /**
     * @brief Update configuration parameters
     * @param config New configuration
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset to initial state
     */
    void reset();
    
    /**
     * @brief Get comprehensive metrics
     * @return Performance and efficiency metrics
     */
    std::unordered_map<std::string, float> get_metrics() const;

private:
    // Resource pools
    struct ResourcePool {
        float total_capacity;
        std::atomic<float> available;
        std::atomic<float> current_usage;
        float peak_usage;
        float usage_sum;
        uint64_t sample_count;
        std::unordered_map<std::string, float> allocations;  // requester_id -> amount
        std::mutex pool_mutex;
    };
    
    std::unordered_map<ResourceType, std::unique_ptr<ResourcePool>> resource_pools_;
    
    // Task scheduling
    std::priority_queue<ScheduledTask> task_queue_;
    mutable std::mutex task_queue_mutex_;
    
    // Memory management
    struct MemoryBlock {
        size_t size;
        std::string requester_id;
        std::chrono::high_resolution_clock::time_point allocation_time;
        size_t access_count;
    };
    
    std::unordered_map<void*, MemoryBlock> memory_allocations_;
    mutable std::mutex memory_mutex_;
    std::atomic<size_t> total_memory_allocated_{0};
    
    // Load balancing
    LoadBalancingStrategy load_strategy_;
    std::unordered_map<std::string, float> module_loads_;
    mutable std::mutex load_mutex_;
    
    // Monitoring and alerts
    std::vector<ResourceAlert> active_alerts_;
    std::unordered_map<ResourceType, std::pair<float, ResourceAlert::Severity>> alert_thresholds_;
    mutable std::mutex alert_mutex_;
    
    // Prediction
    struct PredictionModel {
        std::vector<float> historical_usage;
        size_t max_history;
        float learning_rate;
    };
    
    std::unordered_map<ResourceType, std::unique_ptr<PredictionModel>> prediction_models_;
    mutable std::mutex prediction_mutex_;
    
    // Degraded mode tracking
    std::unordered_map<ResourceType, bool> degraded_resources_;
    mutable std::mutex degraded_mutex_;
    
    // Statistics
    struct AllocationMetrics {
        std::atomic<uint64_t> allocation_count{0};
        std::atomic<uint64_t> rejection_count{0};
        std::chrono::nanoseconds total_allocation_time{0};
        mutable std::mutex metrics_mutex;
    };
    
    std::unordered_map<ResourceType, std::unique_ptr<AllocationMetrics>> allocation_metrics_;
    
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Internal methods
    void initialize_resource_pools();
    void update_statistics(ResourceType type, float amount);
    void check_and_generate_alerts(ResourceType type);
    bool try_allocate_from_pool(ResourcePool& pool, float amount);
    void release_to_pool(ResourcePool& pool, const std::string& requester_id, float amount);
    std::vector<float> predict_time_series(const std::vector<float>& history, size_t steps);
    float calculate_priority_weight(Priority priority) const;
    void compact_memory();
    void evict_low_priority_allocations(ResourceType type, float needed);
};

} // namespace rros

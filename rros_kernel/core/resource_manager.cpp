#include "resource_manager.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace rros {

ResourceManager::ResourceManager(const std::unordered_map<std::string, float>& config)
    : load_strategy_(LoadBalancingStrategy::PRIORITY_BASED), config_(config) {
    
    // Set default configuration
    if (config_.find("computational_capacity") == config_.end()) {
        config_["computational_capacity"] = 100.0f;
    }
    if (config_.find("memory_capacity") == config_.end()) {
        config_["memory_capacity"] = 1048576.0f;  // 1MB default
    }
    if (config_.find("attention_capacity") == config_.end()) {
        config_["attention_capacity"] = 1.0f;
    }
    if (config_.find("io_capacity") == config_.end()) {
        config_["io_capacity"] = 100.0f;
    }
    if (config_.find("network_capacity") == config_.end()) {
        config_["network_capacity"] = 100.0f;
    }
    if (config_.find("alert_threshold") == config_.end()) {
        config_["alert_threshold"] = 0.85f;
    }
    if (config_.find("gc_age_threshold") == config_.end()) {
        config_["gc_age_threshold"] = 300.0f;  // 5 minutes default
    }
    
    initialize_resource_pools();
}

ResourceManager::~ResourceManager() {
    // Clean up any remaining memory allocations
    for (auto& [ptr, block] : memory_allocations_) {
        delete[] static_cast<char*>(ptr);
    }
    memory_allocations_.clear();
}

void ResourceManager::initialize_resource_pools() {
    // Initialize resource pools with configured capacities
    for (int i = static_cast<int>(ResourceType::COMPUTATIONAL); 
         i <= static_cast<int>(ResourceType::NETWORK); ++i) {
        auto type = static_cast<ResourceType>(i);
        auto pool = std::make_unique<ResourcePool>();
        
        switch (type) {
            case ResourceType::COMPUTATIONAL:
                pool->total_capacity = config_["computational_capacity"];
                break;
            case ResourceType::MEMORY:
                pool->total_capacity = config_["memory_capacity"];
                break;
            case ResourceType::ATTENTION:
                pool->total_capacity = config_["attention_capacity"];
                break;
            case ResourceType::IO:
                pool->total_capacity = config_["io_capacity"];
                break;
            case ResourceType::NETWORK:
                pool->total_capacity = config_["network_capacity"];
                break;
        }
        
        pool->available = pool->total_capacity;
        pool->current_usage = 0.0f;
        pool->peak_usage = 0.0f;
        pool->usage_sum = 0.0f;
        pool->sample_count = 0;
        
        resource_pools_[type] = std::move(pool);
        
        // Initialize metrics
        allocation_metrics_[type] = std::make_unique<AllocationMetrics>();
        
        // Initialize prediction model
        auto model = std::make_unique<PredictionModel>();
        model->max_history = 1000;
        model->learning_rate = 0.1f;
        prediction_models_[type] = std::move(model);
        
        // Set default alert thresholds
        alert_thresholds_[type] = {config_["alert_threshold"], ResourceAlert::Severity::WARNING};
        
        // Initialize degraded state
        degraded_resources_[type] = false;
    }
}

ResourceAllocation ResourceManager::allocate_resource(const ResourceRequest& request) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ResourceAllocation result;
    result.requester_id = request.requester_id;
    result.resource_type = request.resource_type;
    result.allocated_amount = 0.0f;
    result.fully_satisfied = false;
    result.allocation_time = start_time;
    
    auto it = resource_pools_.find(request.resource_type);
    if (it == resource_pools_.end()) {
        allocation_metrics_[request.resource_type]->rejection_count++;
        return result;
    }
    
    auto& pool = it->second;
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    
    // Check if we can satisfy the request
    float available = pool->available.load();
    if (available >= request.amount) {
        // Full allocation
        pool->available.store(pool->available.load() - request.amount);
        pool->current_usage.store(pool->current_usage.load() + request.amount);
        pool->allocations[request.requester_id] += request.amount;
        
        result.allocated_amount = request.amount;
        result.fully_satisfied = true;
    } else if (request.priority == Priority::CRITICAL && available > 0) {
        // Partial allocation for critical requests
        pool->available.store(0.0f);
        pool->current_usage.store(pool->current_usage.load() + available);
        pool->allocations[request.requester_id] += available;
        
        result.allocated_amount = available;
        result.fully_satisfied = false;
    } else {
        // Cannot allocate
        allocation_metrics_[request.resource_type]->rejection_count++;
        return result;
    }
    
    // Update statistics
    update_statistics(request.resource_type, result.allocated_amount);
    allocation_metrics_[request.resource_type]->allocation_count++;
    
    // Check for alerts
    check_and_generate_alerts(request.resource_type);
    
    // Track allocation time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    {
        std::lock_guard<std::mutex> metrics_lock(
            allocation_metrics_[request.resource_type]->metrics_mutex);
        allocation_metrics_[request.resource_type]->total_allocation_time += duration;
    }
    
    return result;
}

void ResourceManager::release_resource(const std::string& requester_id,
                                       ResourceType resource_type,
                                       float amount) {
    auto it = resource_pools_.find(resource_type);
    if (it == resource_pools_.end()) {
        return;
    }
    
    release_to_pool(*it->second, requester_id, amount);
    check_and_generate_alerts(resource_type);
}

float ResourceManager::get_available_resources(ResourceType resource_type) const {
    auto it = resource_pools_.find(resource_type);
    if (it == resource_pools_.end()) {
        return 0.0f;
    }
    return it->second->available.load();
}

bool ResourceManager::schedule_task(const ScheduledTask& task) {
    std::lock_guard<std::mutex> lock(task_queue_mutex_);
    
    // Check if we have resources for the task
    bool can_schedule = true;
    for (const auto& [res_type, amount] : task.resource_requirements) {
        if (get_available_resources(res_type) < amount) {
            can_schedule = false;
            break;
        }
    }
    
    if (can_schedule || task.priority == Priority::CRITICAL) {
        task_queue_.push(task);
        return true;
    }
    
    return false;
}

bool ResourceManager::execute_next_task() {
    std::unique_lock<std::mutex> lock(task_queue_mutex_);
    
    if (task_queue_.empty()) {
        return false;
    }
    
    ScheduledTask task = task_queue_.top();
    task_queue_.pop();
    lock.unlock();
    
    // Allocate resources
    std::vector<ResourceAllocation> allocations;
    for (const auto& [res_type, amount] : task.resource_requirements) {
        ResourceRequest req(task.task_id, res_type, task.priority, amount);
        auto allocation = allocate_resource(req);
        allocations.push_back(allocation);
    }
    
    // Execute task if resources were allocated
    bool all_allocated = std::all_of(allocations.begin(), allocations.end(),
        [](const ResourceAllocation& alloc) { return alloc.fully_satisfied; });
    
    if (all_allocated || task.priority == Priority::CRITICAL) {
        try {
            task.task_function();
        } catch (...) {
            // Task execution failed, release resources
        }
    }
    
    // Release resources
    for (const auto& allocation : allocations) {
        if (allocation.allocated_amount > 0) {
            release_resource(task.task_id, allocation.resource_type, 
                           allocation.allocated_amount);
        }
    }
    
    return true;
}

std::unordered_map<Priority, size_t> ResourceManager::get_pending_task_counts() const {
    std::lock_guard<std::mutex> lock(task_queue_mutex_);
    
    std::unordered_map<Priority, size_t> counts;
    auto queue_copy = task_queue_;
    
    while (!queue_copy.empty()) {
        counts[queue_copy.top().priority]++;
        queue_copy.pop();
    }
    
    return counts;
}

void* ResourceManager::allocate_memory(size_t size, const std::string& requester_id) {
    // Request memory resource
    ResourceRequest req(requester_id, ResourceType::MEMORY, Priority::NORMAL, 
                       static_cast<float>(size));
    auto allocation = allocate_resource(req);
    
    if (!allocation.fully_satisfied) {
        return nullptr;
    }
    
    // Actually allocate the memory
    void* ptr = new char[size];
    
    std::lock_guard<std::mutex> lock(memory_mutex_);
    MemoryBlock block;
    block.size = size;
    block.requester_id = requester_id;
    block.allocation_time = std::chrono::high_resolution_clock::now();
    block.access_count = 0;
    
    memory_allocations_[ptr] = block;
    total_memory_allocated_ += size;
    
    return ptr;
}

void ResourceManager::free_memory(void* ptr) {
    if (!ptr) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = memory_allocations_.find(ptr);
    if (it != memory_allocations_.end()) {
        size_t size = it->second.size;
        std::string requester_id = it->second.requester_id;
        
        memory_allocations_.erase(it);
        total_memory_allocated_ -= size;
        
        delete[] static_cast<char*>(ptr);
        
        // Release memory resource
        release_resource(requester_id, ResourceType::MEMORY, static_cast<float>(size));
    }
}

size_t ResourceManager::run_garbage_collection() {
    size_t reclaimed = 0;
    auto now = std::chrono::high_resolution_clock::now();
    
    // Find unused allocations (not accessed recently)
    std::vector<void*> to_free;
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (auto& [ptr, block] : memory_allocations_) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(
                now - block.allocation_time).count();
            
            // Free if old and rarely accessed
            if (age > config_["gc_age_threshold"] && block.access_count < 5) {
                to_free.push_back(ptr);
                reclaimed += block.size;
            }
        }
    }
    
    // Free identified allocations (without holding the lock)
    for (void* ptr : to_free) {
        std::string requester_id;
        size_t size;
        {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            auto it = memory_allocations_.find(ptr);
            if (it != memory_allocations_.end()) {
                requester_id = it->second.requester_id;
                size = it->second.size;
                memory_allocations_.erase(it);
                total_memory_allocated_ -= size;
            } else {
                continue;
            }
        }
        
        delete[] static_cast<char*>(ptr);
        release_resource(requester_id, ResourceType::MEMORY, static_cast<float>(size));
    }
    
    // Compact memory if fragmented
    if (get_memory_fragmentation() > 0.5f) {
        compact_memory();
    }
    
    return reclaimed;
}

float ResourceManager::get_memory_fragmentation() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    if (memory_allocations_.empty()) {
        return 0.0f;
    }
    
    // Simple fragmentation metric: ratio of allocation count to total memory
    float avg_block_size = static_cast<float>(total_memory_allocated_) / 
                          memory_allocations_.size();
    float ideal_block_size = 4096.0f;  // Assume 4KB ideal
    
    return std::min(1.0f, std::abs(avg_block_size - ideal_block_size) / ideal_block_size);
}

void ResourceManager::set_load_balancing_strategy(LoadBalancingStrategy strategy) {
    std::lock_guard<std::mutex> lock(load_mutex_);
    load_strategy_ = strategy;
}

std::unordered_map<std::string, std::vector<std::string>> 
ResourceManager::balance_load(const std::unordered_map<std::string, float>& module_loads) {
    std::lock_guard<std::mutex> lock(load_mutex_);
    
    module_loads_ = module_loads;
    std::unordered_map<std::string, std::vector<std::string>> assignments;
    
    // Get pending tasks
    std::vector<ScheduledTask> tasks;
    {
        std::lock_guard<std::mutex> task_lock(task_queue_mutex_);
        auto queue_copy = task_queue_;
        while (!queue_copy.empty()) {
            tasks.push_back(queue_copy.top());
            queue_copy.pop();
        }
    }
    
    // Assign tasks based on strategy
    switch (load_strategy_) {
        case LoadBalancingStrategy::ROUND_ROBIN: {
            size_t idx = 0;
            for (const auto& task : tasks) {
                auto it = module_loads_.begin();
                std::advance(it, idx % module_loads_.size());
                assignments[it->first].push_back(task.task_id);
                idx++;
            }
            break;
        }
        
        case LoadBalancingStrategy::LEAST_LOADED: {
            for (const auto& task : tasks) {
                // Find module with least load
                auto min_it = std::min_element(module_loads_.begin(), module_loads_.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
                if (min_it != module_loads_.end()) {
                    assignments[min_it->first].push_back(task.task_id);
                    module_loads_[min_it->first] += 1.0f;  // Update load
                }
            }
            break;
        }
        
        case LoadBalancingStrategy::PRIORITY_BASED: {
            // Sort tasks by priority
            std::sort(tasks.begin(), tasks.end(),
                [](const ScheduledTask& a, const ScheduledTask& b) {
                    return a.priority < b.priority;
                });
            
            for (const auto& task : tasks) {
                // Assign high priority to least loaded, others round-robin
                if (task.priority <= Priority::HIGH) {
                    auto min_it = std::min_element(module_loads_.begin(), module_loads_.end(),
                        [](const auto& a, const auto& b) { return a.second < b.second; });
                    if (min_it != module_loads_.end()) {
                        assignments[min_it->first].push_back(task.task_id);
                        module_loads_[min_it->first] += calculate_priority_weight(task.priority);
                    }
                } else {
                    // Round robin for lower priority
                    static size_t rr_idx = 0;
                    auto it = module_loads_.begin();
                    std::advance(it, rr_idx % module_loads_.size());
                    assignments[it->first].push_back(task.task_id);
                    rr_idx++;
                }
            }
            break;
        }
        
        case LoadBalancingStrategy::PREDICTIVE: {
            // Use predictions to optimize placement
            for (const auto& task : tasks) {
                std::string best_module;
                float best_score = std::numeric_limits<float>::max();
                
                for (const auto& [module, load] : module_loads_) {
                    // Predict completion time based on current load
                    float predicted_time = load * calculate_priority_weight(task.priority);
                    if (predicted_time < best_score) {
                        best_score = predicted_time;
                        best_module = module;
                    }
                }
                
                if (!best_module.empty()) {
                    assignments[best_module].push_back(task.task_id);
                    module_loads_[best_module] += calculate_priority_weight(task.priority);
                }
            }
            break;
        }
    }
    
    return assignments;
}

void ResourceManager::adapt_to_load(const std::unordered_map<ResourceType, float>& current_loads) {
    for (const auto& [type, load] : current_loads) {
        auto it = resource_pools_.find(type);
        if (it == resource_pools_.end()) {
            continue;
        }
        
        float utilization = load / it->second->total_capacity;
        
        // Enter degraded mode if utilization is very high
        if (utilization > 0.95f && !degraded_resources_[type]) {
            enter_degraded_mode(type);
        }
        // Exit degraded mode if utilization drops
        else if (utilization < 0.7f && degraded_resources_[type]) {
            exit_degraded_mode(type);
        }
        
        // Update prediction models
        update_prediction_model(type, load);
    }
}

ResourceStats ResourceManager::get_resource_stats(ResourceType resource_type) const {
    ResourceStats stats;
    stats.type = resource_type;
    
    auto pool_it = resource_pools_.find(resource_type);
    if (pool_it != resource_pools_.end()) {
        const auto& pool = pool_it->second;
        std::lock_guard<std::mutex> lock(pool->pool_mutex);
        
        stats.total_capacity = pool->total_capacity;
        stats.current_usage = pool->current_usage.load();
        stats.peak_usage = pool->peak_usage;
        stats.average_usage = pool->sample_count > 0 ? 
            pool->usage_sum / pool->sample_count : 0.0f;
    }
    
    auto metrics_it = allocation_metrics_.find(resource_type);
    if (metrics_it != allocation_metrics_.end()) {
        const auto& metrics = metrics_it->second;
        stats.allocation_count = metrics->allocation_count.load();
        stats.rejection_count = metrics->rejection_count.load();
        
        std::lock_guard<std::mutex> lock(metrics->metrics_mutex);
        if (stats.allocation_count > 0) {
            stats.avg_allocation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                metrics->total_allocation_time / stats.allocation_count);
        }
    }
    
    return stats;
}

std::vector<ResourceAlert> ResourceManager::get_active_alerts() const {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    return active_alerts_;
}

void ResourceManager::set_alert_threshold(ResourceType resource_type,
                                          float threshold,
                                          ResourceAlert::Severity severity) {
    std::lock_guard<std::mutex> lock(alert_mutex_);
    alert_thresholds_[resource_type] = {threshold, severity};
}

std::unordered_map<ResourceType, float> ResourceManager::get_utilization() const {
    std::unordered_map<ResourceType, float> utilization;
    
    for (const auto& [type, pool] : resource_pools_) {
        std::lock_guard<std::mutex> lock(pool->pool_mutex);
        float usage = pool->current_usage.load();
        utilization[type] = usage / pool->total_capacity;
    }
    
    return utilization;
}

ResourcePrediction ResourceManager::predict_usage(ResourceType resource_type,
                                                  std::chrono::milliseconds horizon) {
    ResourcePrediction prediction;
    prediction.resource_type = resource_type;
    prediction.horizon = horizon;
    prediction.confidence = 0.5f;
    
    std::lock_guard<std::mutex> lock(prediction_mutex_);
    auto it = prediction_models_.find(resource_type);
    if (it != prediction_models_.end()) {
        const auto& model = it->second;
        
        if (!model->historical_usage.empty()) {
            // Simple time series prediction
            size_t steps = horizon.count() / 100;  // Assume 100ms per step
            prediction.predicted_usage = predict_time_series(model->historical_usage, steps);
            
            // Confidence based on history size
            prediction.confidence = std::min(1.0f, 
                static_cast<float>(model->historical_usage.size()) / model->max_history);
        }
    }
    
    return prediction;
}

void ResourceManager::preallocate_predicted(const std::vector<ResourcePrediction>& predictions) {
    for (const auto& pred : predictions) {
        if (pred.confidence < 0.5f || pred.predicted_usage.empty()) {
            continue;
        }
        
        // Pre-allocate based on predicted peak usage
        float peak_predicted = *std::max_element(pred.predicted_usage.begin(), 
                                                 pred.predicted_usage.end());
        
        ResourceRequest req("preallocation", pred.resource_type, 
                          Priority::LOW, peak_predicted * 0.5f);  // Reserve 50% of predicted peak
        allocate_resource(req);
    }
}

void ResourceManager::update_prediction_model(ResourceType resource_type, float actual_usage) {
    std::lock_guard<std::mutex> lock(prediction_mutex_);
    auto it = prediction_models_.find(resource_type);
    if (it != prediction_models_.end()) {
        auto& model = it->second;
        
        model->historical_usage.push_back(actual_usage);
        
        // Limit history size
        if (model->historical_usage.size() > model->max_history) {
            model->historical_usage.erase(model->historical_usage.begin());
        }
    }
}

void ResourceManager::enter_degraded_mode(ResourceType resource_type) {
    std::lock_guard<std::mutex> lock(degraded_mutex_);
    degraded_resources_[resource_type] = true;
    
    // Generate critical alert
    std::lock_guard<std::mutex> alert_lock(alert_mutex_);
    ResourceAlert alert;
    alert.severity = ResourceAlert::Severity::CRITICAL;
    alert.resource_type = resource_type;
    alert.message = "Entering degraded mode due to resource pressure";
    alert.threshold_violated = 0.95f;
    alert.timestamp = std::chrono::high_resolution_clock::now();
    active_alerts_.push_back(alert);
}

void ResourceManager::exit_degraded_mode(ResourceType resource_type) {
    std::lock_guard<std::mutex> lock(degraded_mutex_);
    degraded_resources_[resource_type] = false;
    
    // Generate info alert
    std::lock_guard<std::mutex> alert_lock(alert_mutex_);
    ResourceAlert alert;
    alert.severity = ResourceAlert::Severity::INFO;
    alert.resource_type = resource_type;
    alert.message = "Exiting degraded mode - resources recovered";
    alert.threshold_violated = 0.7f;
    alert.timestamp = std::chrono::high_resolution_clock::now();
    active_alerts_.push_back(alert);
}

bool ResourceManager::is_degraded() const {
    std::lock_guard<std::mutex> lock(degraded_mutex_);
    for (const auto& [type, degraded] : degraded_resources_) {
        if (degraded) {
            return true;
        }
    }
    return false;
}

void ResourceManager::update_config(const std::unordered_map<std::string, float>& config) {
    config_ = config;
    
    // Update resource pool capacities
    for (auto& [type, pool] : resource_pools_) {
        std::lock_guard<std::mutex> lock(pool->pool_mutex);
        
        float old_capacity = pool->total_capacity;
        float new_capacity = old_capacity;
        
        switch (type) {
            case ResourceType::COMPUTATIONAL:
                if (config.find("computational_capacity") != config.end()) {
                    new_capacity = config.at("computational_capacity");
                }
                break;
            case ResourceType::MEMORY:
                if (config.find("memory_capacity") != config.end()) {
                    new_capacity = config.at("memory_capacity");
                }
                break;
            case ResourceType::ATTENTION:
                if (config.find("attention_capacity") != config.end()) {
                    new_capacity = config.at("attention_capacity");
                }
                break;
            case ResourceType::IO:
                if (config.find("io_capacity") != config.end()) {
                    new_capacity = config.at("io_capacity");
                }
                break;
            case ResourceType::NETWORK:
                if (config.find("network_capacity") != config.end()) {
                    new_capacity = config.at("network_capacity");
                }
                break;
        }
        
        // Adjust available resources proportionally
        float delta = new_capacity - old_capacity;
        pool->total_capacity = new_capacity;
        pool->available = pool->available.load() + delta;
    }
}

void ResourceManager::reset() {
    // Clear all allocations
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        for (auto& [ptr, block] : memory_allocations_) {
            delete[] static_cast<char*>(ptr);
        }
        memory_allocations_.clear();
        total_memory_allocated_ = 0;
    }
    
    // Reset resource pools
    for (auto& [type, pool] : resource_pools_) {
        std::lock_guard<std::mutex> lock(pool->pool_mutex);
        pool->available = pool->total_capacity;
        pool->current_usage = 0.0f;
        pool->peak_usage = 0.0f;
        pool->usage_sum = 0.0f;
        pool->sample_count = 0;
        pool->allocations.clear();
    }
    
    // Clear task queue
    {
        std::lock_guard<std::mutex> lock(task_queue_mutex_);
        while (!task_queue_.empty()) {
            task_queue_.pop();
        }
    }
    
    // Clear alerts
    {
        std::lock_guard<std::mutex> lock(alert_mutex_);
        active_alerts_.clear();
    }
    
    // Reset degraded state
    {
        std::lock_guard<std::mutex> lock(degraded_mutex_);
        for (auto& [type, degraded] : degraded_resources_) {
            degraded = false;
        }
    }
}

std::unordered_map<std::string, float> ResourceManager::get_metrics() const {
    std::unordered_map<std::string, float> metrics;
    
    // Aggregate utilization
    float total_utilization = 0.0f;
    for (const auto& [type, pool] : resource_pools_) {
        std::lock_guard<std::mutex> lock(pool->pool_mutex);
        float util = pool->current_usage.load() / pool->total_capacity;
        total_utilization += util;
    }
    metrics["avg_utilization"] = total_utilization / resource_pools_.size();
    
    // Memory metrics
    metrics["total_memory_allocated"] = static_cast<float>(total_memory_allocated_.load());
    metrics["memory_fragmentation"] = get_memory_fragmentation();
    
    // Task metrics
    {
        std::lock_guard<std::mutex> lock(task_queue_mutex_);
        metrics["pending_tasks"] = static_cast<float>(task_queue_.size());
    }
    
    // Alert metrics
    {
        std::lock_guard<std::mutex> lock(alert_mutex_);
        metrics["active_alerts"] = static_cast<float>(active_alerts_.size());
    }
    
    // Degradation status
    metrics["is_degraded"] = is_degraded() ? 1.0f : 0.0f;
    
    return metrics;
}

// Private helper methods

void ResourceManager::update_statistics(ResourceType type, float amount) {
    auto it = resource_pools_.find(type);
    if (it != resource_pools_.end()) {
        auto& pool = it->second;
        float current = pool->current_usage.load();
        
        pool->usage_sum += current;
        pool->sample_count++;
        
        if (current > pool->peak_usage) {
            pool->peak_usage = current;
        }
    }
}

void ResourceManager::check_and_generate_alerts(ResourceType type) {
    auto pool_it = resource_pools_.find(type);
    auto threshold_it = alert_thresholds_.find(type);
    
    if (pool_it == resource_pools_.end() || threshold_it == alert_thresholds_.end()) {
        return;
    }
    
    const auto& pool = pool_it->second;
    float utilization = pool->current_usage.load() / pool->total_capacity;
    float threshold = threshold_it->second.first;
    
    if (utilization > threshold) {
        std::lock_guard<std::mutex> lock(alert_mutex_);
        
        // Check if we already have an alert for this
        bool has_alert = std::any_of(active_alerts_.begin(), active_alerts_.end(),
            [type](const ResourceAlert& alert) {
                return alert.resource_type == type;
            });
        
        if (!has_alert) {
            ResourceAlert alert;
            alert.severity = threshold_it->second.second;
            alert.resource_type = type;
            alert.message = "Resource utilization exceeded threshold";
            alert.threshold_violated = utilization;
            alert.timestamp = std::chrono::high_resolution_clock::now();
            active_alerts_.push_back(alert);
        }
    } else {
        // Clear any existing alerts if utilization is back to normal
        std::lock_guard<std::mutex> lock(alert_mutex_);
        active_alerts_.erase(
            std::remove_if(active_alerts_.begin(), active_alerts_.end(),
                [type](const ResourceAlert& alert) {
                    return alert.resource_type == type;
                }),
            active_alerts_.end()
        );
    }
}

bool ResourceManager::try_allocate_from_pool(ResourcePool& pool, float amount) {
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    float available = pool.available.load();
    
    if (available >= amount) {
        pool.available.store(pool.available.load() - amount);
        pool.current_usage.store(pool.current_usage.load() + amount);
        return true;
    }
    
    return false;
}

void ResourceManager::release_to_pool(ResourcePool& pool, 
                                      const std::string& requester_id, 
                                      float amount) {
    std::lock_guard<std::mutex> lock(pool.pool_mutex);
    
    auto it = pool.allocations.find(requester_id);
    if (it != pool.allocations.end()) {
        float to_release = std::min(amount, it->second);
        it->second -= to_release;
        
        if (it->second <= 0.0f) {
            pool.allocations.erase(it);
        }
        
        pool.available.store(pool.available.load() + to_release);
        pool.current_usage.store(pool.current_usage.load() - to_release);
    }
}

std::vector<float> ResourceManager::predict_time_series(const std::vector<float>& history, 
                                                        size_t steps) {
    std::vector<float> predictions;
    
    if (history.empty()) {
        return predictions;
    }
    
    // Simple moving average prediction
    size_t window = std::min(size_t(10), history.size());
    float sum = 0.0f;
    for (size_t i = history.size() - window; i < history.size(); ++i) {
        sum += history[i];
    }
    float avg = sum / window;
    
    // Calculate trend
    float trend = 0.0f;
    if (history.size() > 1) {
        trend = (history.back() - history[history.size() - window]) / window;
    }
    
    // Generate predictions
    for (size_t i = 0; i < steps; ++i) {
        float predicted = avg + trend * i;
        predictions.push_back(std::max(0.0f, predicted));
    }
    
    return predictions;
}

float ResourceManager::calculate_priority_weight(Priority priority) const {
    switch (priority) {
        case Priority::CRITICAL: return 10.0f;
        case Priority::HIGH: return 5.0f;
        case Priority::NORMAL: return 1.0f;
        case Priority::LOW: return 0.5f;
        case Priority::IDLE: return 0.1f;
        default: return 1.0f;
    }
}

void ResourceManager::compact_memory() {
    // Memory compaction would require moving allocations
    // For now, just a placeholder that could be implemented
    // with a custom allocator
}

void ResourceManager::evict_low_priority_allocations(ResourceType type, float needed) {
    auto it = resource_pools_.find(type);
    if (it == resource_pools_.end()) {
        return;
    }
    
    auto& pool = it->second;
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    
    // Find and release low priority allocations
    std::vector<std::pair<std::string, float>> to_evict;
    for (const auto& [requester_id, amount] : pool->allocations) {
        // Simple heuristic: evict if requester_id contains "low" or "idle"
        if (requester_id.find("low") != std::string::npos || 
            requester_id.find("idle") != std::string::npos) {
            to_evict.push_back({requester_id, amount});
        }
    }
    
    float evicted = 0.0f;
    for (const auto& [requester_id, amount] : to_evict) {
        pool->allocations.erase(requester_id);
        pool->available.store(pool->available.load() + amount);
        pool->current_usage.store(pool->current_usage.load() - amount);
        evicted += amount;
        
        if (evicted >= needed) {
            break;
        }
    }
}

} // namespace rros

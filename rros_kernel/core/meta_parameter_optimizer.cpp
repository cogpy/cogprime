#include "meta_parameter_optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {

MetaParameterOptimizer::MetaParameterOptimizer(
    const std::unordered_map<std::string, float>& config)
    : config_(config) {
    
    if (config_.find("learning_rate_scale") == config_.end()) {
        config_["learning_rate_scale"] = 0.1f;
    }
    if (config_.find("exploration_factor") == config_.end()) {
        config_["exploration_factor"] = 0.1f;
    }
}

MetaParameterOptimizer::~MetaParameterOptimizer() = default;

std::unordered_map<std::string, float> MetaParameterOptimizer::optimize(
    const std::unordered_map<std::string, float>& current_params,
    const std::vector<float>& performance_history) {
    
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    // Record current parameters and performance
    param_history_.push_back(current_params);
    if (!performance_history.empty()) {
        performance_scores_.push_back(performance_history.back());
    }
    
    // Limit history size
    if (param_history_.size() > 100) {
        param_history_.erase(param_history_.begin());
        if (!performance_scores_.empty()) {
            performance_scores_.erase(performance_scores_.begin());
        }
    }
    
    std::unordered_map<std::string, float> optimized_params = current_params;
    
    // Optimize each parameter
    for (const auto& [param_name, current_value] : current_params) {
        // Estimate performance gradient
        float gradient = estimate_performance_gradient(param_name, performance_history);
        
        // Suggest new value based on gradient
        float new_value = suggest_parameter_value(param_name, current_value, gradient);
        
        optimized_params[param_name] = new_value;
    }
    
    return optimized_params;
}

void MetaParameterOptimizer::update_config(
    const std::unordered_map<std::string, float>& config) {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
}

void MetaParameterOptimizer::reset() {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    param_history_.clear();
    performance_scores_.clear();
}

float MetaParameterOptimizer::estimate_performance_gradient(
    const std::string& param_name,
    const std::vector<float>& performance_history) const {
    
    if (performance_history.size() < 2 || param_history_.size() < 2) {
        return 0.0f;
    }
    
    // Compute gradient using recent history
    size_t window = std::min(size_t(5), performance_history.size());
    float gradient = 0.0f;
    
    for (size_t i = performance_history.size() - window; i < performance_history.size() - 1; ++i) {
        float perf_delta = performance_history[i + 1] - performance_history[i];
        
        // Check if we have corresponding parameter history
        if (i < param_history_.size() - 1) {
            auto it_curr = param_history_[i].find(param_name);
            auto it_next = param_history_[i + 1].find(param_name);
            
            if (it_curr != param_history_[i].end() && 
                it_next != param_history_[i + 1].end()) {
                
                float param_delta = it_next->second - it_curr->second;
                
                if (std::abs(param_delta) > 1e-6f) {
                    gradient += perf_delta / param_delta;
                }
            }
        }
    }
    
    return gradient / static_cast<float>(window);
}

float MetaParameterOptimizer::suggest_parameter_value(
    const std::string& param_name,
    float current_value,
    float gradient) const {
    
    // Apply gradient-based update
    float learning_rate = config_.at("learning_rate_scale");
    float exploration = config_.at("exploration_factor");
    
    // Update based on gradient
    float new_value = current_value + learning_rate * gradient;
    
    // Add exploration noise
    float noise = exploration * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f);
    new_value += noise;
    
    // Apply parameter-specific constraints
    if (param_name.find("learning_rate") != std::string::npos) {
        new_value = std::max(1e-6f, std::min(1.0f, new_value));
    } else if (param_name.find("threshold") != std::string::npos) {
        new_value = std::max(0.0f, std::min(1.0f, new_value));
    } else if (param_name.find("temperature") != std::string::npos) {
        new_value = std::max(0.1f, std::min(10.0f, new_value));
    } else {
        // Generic constraint
        new_value = std::max(0.0f, std::min(10.0f, new_value));
    }
    
    return new_value;
}

} // namespace rros

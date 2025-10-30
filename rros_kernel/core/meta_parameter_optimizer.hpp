#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>

namespace rros {

/**
 * @brief Meta-Parameter Optimizer for automatic hyperparameter tuning
 * 
 * Optimizes meta-parameters based on performance history using
 * Bayesian optimization and gradient-based methods.
 */
class MetaParameterOptimizer {
public:
    explicit MetaParameterOptimizer(const std::unordered_map<std::string, float>& config = {});
    ~MetaParameterOptimizer();
    
    /**
     * @brief Optimize meta-parameters
     * @param current_params Current parameter values
     * @param performance_history Historical performance
     * @return Optimized parameters
     */
    std::unordered_map<std::string, float> optimize(
        const std::unordered_map<std::string, float>& current_params,
        const std::vector<float>& performance_history
    );
    
    void update_config(const std::unordered_map<std::string, float>& config);
    void reset();

private:
    std::unordered_map<std::string, float> config_;
    std::vector<std::unordered_map<std::string, float>> param_history_;
    std::vector<float> performance_scores_;
    mutable std::mutex optimizer_mutex_;
    
    float estimate_performance_gradient(
        const std::string& param_name,
        const std::vector<float>& performance_history
    ) const;
    
    float suggest_parameter_value(
        const std::string& param_name,
        float current_value,
        float gradient
    ) const;
};

} // namespace rros

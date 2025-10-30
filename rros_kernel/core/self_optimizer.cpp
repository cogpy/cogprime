#include "self_optimizer.hpp"
#include "meta_cognitive_monitor.hpp"
#include "decision_quality_tracker.hpp"
#include <algorithm>
#include <cmath>

namespace rros {

// OptimizationStrategy implementation
OptimizationStrategy::OptimizationStrategy()
    : expected_improvement(0.0f)
    , confidence(0.5f)
{}

// ImprovementMetrics implementation
ImprovementMetrics::ImprovementMetrics()
    : baseline_performance(0.0f)
    , current_performance(0.0f)
    , improvement_rate(0.0f)
    , optimization_count(0)
    , last_optimization(std::chrono::high_resolution_clock::now())
{}

// SelfOptimizer implementation
SelfOptimizer::SelfOptimizer(const std::unordered_map<std::string, float>& config)
    : config_(config)
    , max_history_size_(static_cast<size_t>(
        config.count("max_history_size") ? config.at("max_history_size") : 100
      ))
    , autonomous_enabled_(false)
    , min_improvement_threshold_(config.count("min_improvement_threshold") ?
                                 config.at("min_improvement_threshold") : 0.01f)
    , max_parameter_change_(config.count("max_parameter_change") ?
                           config.at("max_parameter_change") : 0.2f)
    , total_optimizations_(0)
    , successful_optimizations_(0)
{
    strategy_history_.clear();
}

SelfOptimizer::~SelfOptimizer() {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    strategy_history_.clear();
}

std::vector<OptimizationStrategy> SelfOptimizer::generate_strategies(
    MetaCognitiveMonitor& monitor,
    DecisionQualityTracker& tracker
) {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    std::vector<OptimizationStrategy> strategies;
    
    // Get introspection data
    auto introspection = monitor.introspect_state(100);
    auto quality_metrics = tracker.get_quality_metrics("");
    
    // Analyze awareness and relevance stability
    float awareness = introspection.count("awareness_score") ? 
                     introspection.at("awareness_score") : 0.5f;
    float stability = introspection.count("relevance_stability") ?
                     introspection.at("relevance_stability") : 0.5f;
    
    // If awareness is low, generate strategy to improve monitoring
    if (awareness < 0.6f) {
        OptimizationStrategy strategy;
        strategy.target_process = "meta_cognitive_monitoring";
        strategy.strategy_type = "parameter_tuning";
        strategy.parameters["snapshot_frequency"] = 1.2f; // Increase frequency
        strategy.expected_improvement = 0.15f;
        strategy.confidence = 0.7f;
        strategies.push_back(strategy);
    }
    
    // If relevance is unstable, generate stabilization strategy
    if (stability < 0.7f) {
        OptimizationStrategy strategy;
        strategy.target_process = "relevance_realization";
        strategy.strategy_type = "parameter_tuning";
        strategy.parameters["smoothing_factor"] = 0.9f;
        strategy.expected_improvement = 0.1f;
        strategy.confidence = 0.65f;
        strategies.push_back(strategy);
    }
    
    // If prediction accuracy is low, generate improvement strategy
    if (quality_metrics.prediction_accuracy < 0.7f) {
        OptimizationStrategy strategy;
        strategy.target_process = "decision_making";
        strategy.strategy_type = "algorithm_switching";
        strategy.parameters["use_ensemble"] = 1.0f;
        strategy.expected_improvement = 0.2f;
        strategy.confidence = 0.75f;
        strategies.push_back(strategy);
    }
    
    return strategies;
}

std::unordered_map<std::string, float> SelfOptimizer::apply_strategy(
    const OptimizationStrategy& strategy,
    const std::unordered_map<std::string, float>& process_parameters
) {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    auto optimized_params = process_parameters;
    
    if (strategy.strategy_type == "parameter_tuning") {
        // Apply parameter adjustments
        for (const auto& param : strategy.parameters) {
            if (optimized_params.count(param.first)) {
                // Constrain parameter changes
                float old_value = optimized_params[param.first];
                float new_value = param.second;
                float max_change = std::abs(old_value) * max_parameter_change_;
                float change = std::max(-max_change, std::min(max_change, new_value - old_value));
                optimized_params[param.first] = old_value + change;
            } else {
                optimized_params[param.first] = param.second;
            }
        }
    }
    else if (strategy.strategy_type == "algorithm_switching") {
        // Switch algorithm flags
        for (const auto& param : strategy.parameters) {
            optimized_params[param.first] = param.second;
        }
    }
    
    total_optimizations_++;
    
    return optimized_params;
}

float SelfOptimizer::evaluate_effectiveness(
    const OptimizationStrategy& strategy,
    const ImprovementMetrics& before_metrics,
    const ImprovementMetrics& after_metrics
) {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    float actual_improvement = after_metrics.current_performance - 
                              before_metrics.current_performance;
    
    float expected_improvement = strategy.expected_improvement;
    
    // Effectiveness is how close we got to the expected improvement
    float effectiveness = 0.0f;
    
    if (expected_improvement > 0) {
        effectiveness = actual_improvement / expected_improvement;
    } else if (actual_improvement > 0) {
        effectiveness = 1.0f; // Unexpected improvement
    }
    
    // Clamp to [0, 2] (can exceed 1 if we did better than expected)
    effectiveness = std::max(0.0f, std::min(2.0f, effectiveness));
    
    // Record success if improvement was positive
    if (actual_improvement > min_improvement_threshold_) {
        successful_optimizations_++;
        record_optimization(strategy, true);
    } else {
        record_optimization(strategy, false);
    }
    
    return effectiveness;
}

std::unordered_map<std::string, float> SelfOptimizer::optimize_loop(
    MetaCognitiveMonitor& monitor,
    DecisionQualityTracker& tracker,
    const std::unordered_map<std::string, float>& process_params
) {
    if (!autonomous_enabled_) {
        return process_params;
    }
    
    // Generate optimization strategies
    auto strategies = generate_strategies(monitor, tracker);
    
    if (strategies.empty()) {
        return process_params;
    }
    
    // Apply the highest confidence strategy
    auto best_strategy = *std::max_element(
        strategies.begin(),
        strategies.end(),
        [](const OptimizationStrategy& a, const OptimizationStrategy& b) {
            return a.confidence < b.confidence;
        }
    );
    
    // Validate strategy before applying
    if (!validate_strategy(best_strategy)) {
        return process_params;
    }
    
    // Apply optimization
    return apply_strategy(best_strategy, process_params);
}

ImprovementMetrics SelfOptimizer::get_improvement_metrics(
    const std::string& process_name
) const {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    auto it = process_improvements_.find(process_name);
    if (it != process_improvements_.end()) {
        return it->second;
    }
    
    return ImprovementMetrics();
}

void SelfOptimizer::set_autonomous_mode(bool enabled) {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    autonomous_enabled_ = enabled;
}

void SelfOptimizer::reset() {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    process_improvements_.clear();
    strategy_history_.clear();
    total_optimizations_ = 0;
    successful_optimizations_ = 0;
}

std::unordered_map<std::string, float> SelfOptimizer::get_statistics() const {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    std::unordered_map<std::string, float> stats;
    stats["total_optimizations"] = static_cast<float>(total_optimizations_);
    stats["successful_optimizations"] = static_cast<float>(successful_optimizations_);
    stats["success_rate"] = total_optimizations_ > 0 ?
                           static_cast<float>(successful_optimizations_) / total_optimizations_ :
                           0.0f;
    stats["strategy_history_size"] = static_cast<float>(strategy_history_.size());
    stats["tracked_processes"] = static_cast<float>(process_improvements_.size());
    stats["autonomous_enabled"] = autonomous_enabled_ ? 1.0f : 0.0f;
    
    return stats;
}

// Private methods

OptimizationStrategy SelfOptimizer::generate_parameter_tuning_strategy(
    const std::string& process_name,
    float current_performance,
    float target_performance
) {
    OptimizationStrategy strategy;
    strategy.target_process = process_name;
    strategy.strategy_type = "parameter_tuning";
    
    float performance_gap = target_performance - current_performance;
    strategy.expected_improvement = performance_gap * 0.5f; // Conservative estimate
    strategy.confidence = 0.6f;
    
    // Generic parameter adjustment
    strategy.parameters["learning_rate"] = 0.01f * (1.0f + performance_gap);
    
    return strategy;
}

OptimizationStrategy SelfOptimizer::generate_algorithm_switching_strategy(
    const std::string& process_name,
    float current_performance
) {
    OptimizationStrategy strategy;
    strategy.target_process = process_name;
    strategy.strategy_type = "algorithm_switching";
    
    strategy.expected_improvement = 0.15f;
    strategy.confidence = 0.5f;
    
    // Suggest algorithm switch if performance is low
    if (current_performance < 0.5f) {
        strategy.parameters["use_alternative_algorithm"] = 1.0f;
        strategy.confidence = 0.7f;
    }
    
    return strategy;
}

bool SelfOptimizer::validate_strategy(const OptimizationStrategy& strategy) {
    // Check if strategy is reasonable
    if (strategy.expected_improvement < 0) {
        return false; // Don't apply strategies that expect to make things worse
    }
    
    if (strategy.confidence < 0.3f) {
        return false; // Don't apply low-confidence strategies
    }
    
    if (strategy.parameters.empty()) {
        return false; // Strategy must have parameters
    }
    
    return true;
}

void SelfOptimizer::record_optimization(
    const OptimizationStrategy& strategy,
    bool success
) {
    strategy_history_.push_back(strategy);
    
    // Update process improvement metrics
    if (process_improvements_.find(strategy.target_process) == process_improvements_.end()) {
        process_improvements_[strategy.target_process] = ImprovementMetrics();
    }
    
    auto& metrics = process_improvements_[strategy.target_process];
    metrics.optimization_count++;
    metrics.last_optimization = std::chrono::high_resolution_clock::now();
    
    if (success) {
        metrics.improvement_rate = (metrics.improvement_rate * (metrics.optimization_count - 1) + 
                                   strategy.expected_improvement) / metrics.optimization_count;
    }
    
    // Prune history if needed
    if (strategy_history_.size() > max_history_size_) {
        strategy_history_.erase(
            strategy_history_.begin(),
            strategy_history_.begin() + max_history_size_ / 5
        );
    }
}

} // namespace rros

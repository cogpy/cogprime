#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <functional>

namespace rros {

// Forward declarations
class MetaCognitiveMonitor;
class DecisionQualityTracker;
class ConfidenceEstimator;
class BiasDetector;

/**
 * @brief Optimization strategy for cognitive processes
 */
struct OptimizationStrategy {
    std::string target_process;
    std::string strategy_type;
    std::unordered_map<std::string, float> parameters;
    float expected_improvement;
    float confidence;
    
    OptimizationStrategy();
};

/**
 * @brief Performance improvement metrics
 */
struct ImprovementMetrics {
    float baseline_performance;
    float current_performance;
    float improvement_rate;
    uint64_t optimization_count;
    std::chrono::high_resolution_clock::time_point last_optimization;
    
    ImprovementMetrics();
};

/**
 * @brief Self-optimization and adaptive improvement system
 * 
 * Uses meta-cognitive feedback to autonomously optimize cognitive
 * processes, adjust parameters, and improve system performance over time.
 * Implements self-improvement loops from Episode 42 (Intelligence & Rationality)
 * and adaptive constraint satisfaction from Episode 30 (Relevance Realization).
 */
class SelfOptimizer {
public:
    /**
     * @brief Constructor initializes optimization system
     * @param config Configuration parameters
     */
    explicit SelfOptimizer(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~SelfOptimizer();
    
    /**
     * @brief Analyze performance and generate optimization strategies
     * @param monitor Meta-cognitive monitor for introspection
     * @param tracker Decision quality tracker for performance data
     * @return List of optimization strategies to apply
     */
    std::vector<OptimizationStrategy> generate_strategies(
        MetaCognitiveMonitor& monitor,
        DecisionQualityTracker& tracker
    );
    
    /**
     * @brief Apply optimization strategy to target process
     * @param strategy Strategy to apply
     * @param process_parameters Current process parameters
     * @return Optimized process parameters
     */
    std::unordered_map<std::string, float> apply_strategy(
        const OptimizationStrategy& strategy,
        const std::unordered_map<std::string, float>& process_parameters
    );
    
    /**
     * @brief Evaluate effectiveness of applied optimization
     * @param strategy Strategy that was applied
     * @param before_metrics Performance before optimization
     * @param after_metrics Performance after optimization
     * @return Effectiveness score [0.0, 1.0]
     */
    float evaluate_effectiveness(
        const OptimizationStrategy& strategy,
        const ImprovementMetrics& before_metrics,
        const ImprovementMetrics& after_metrics
    );
    
    /**
     * @brief Autonomous optimization loop
     * @param monitor Meta-cognitive monitor
     * @param tracker Decision quality tracker
     * @param process_params Current process parameters
     * @return Updated process parameters after optimization
     */
    std::unordered_map<std::string, float> optimize_loop(
        MetaCognitiveMonitor& monitor,
        DecisionQualityTracker& tracker,
        const std::unordered_map<std::string, float>& process_params
    );
    
    /**
     * @brief Get improvement metrics for a process
     * @param process_name Name of the process
     * @return Improvement metrics
     */
    ImprovementMetrics get_improvement_metrics(const std::string& process_name) const;
    
    /**
     * @brief Enable/disable autonomous optimization
     * @param enabled True to enable, false to disable
     */
    void set_autonomous_mode(bool enabled);
    
    /**
     * @brief Reset optimization state
     */
    void reset();
    
    /**
     * @brief Get optimization statistics
     * @return Current optimization statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Improvement tracking
    std::unordered_map<std::string, ImprovementMetrics> process_improvements_;
    
    // Strategy history
    std::vector<OptimizationStrategy> strategy_history_;
    size_t max_history_size_;
    
    // Autonomous mode flag
    bool autonomous_enabled_;
    
    // Optimization limits
    float min_improvement_threshold_;
    float max_parameter_change_;
    
    // Thread safety
    mutable std::mutex optimizer_mutex_;
    
    // Statistics
    uint64_t total_optimizations_;
    uint64_t successful_optimizations_;
    
    // Internal methods
    OptimizationStrategy generate_parameter_tuning_strategy(
        const std::string& process_name,
        float current_performance,
        float target_performance
    );
    
    OptimizationStrategy generate_algorithm_switching_strategy(
        const std::string& process_name,
        float current_performance
    );
    
    bool validate_strategy(const OptimizationStrategy& strategy);
    void record_optimization(const OptimizationStrategy& strategy, bool success);
};

} // namespace rros

#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <functional>

namespace rros {

/**
 * @brief Strategy evaluation criteria
 */
struct EvaluationCriteria {
    float accuracy_weight;          // Weight for accuracy in evaluation
    float speed_weight;             // Weight for execution speed
    float robustness_weight;        // Weight for robustness to variation
    float generalization_weight;    // Weight for cross-domain performance
    float efficiency_weight;        // Weight for resource efficiency
    
    EvaluationCriteria();
};

/**
 * @brief Strategy effectiveness results
 */
struct EffectivenessResults {
    float overall_effectiveness;    // Combined effectiveness score [0.0, 1.0]
    float accuracy_score;           // Accuracy component
    float speed_score;              // Speed component
    float robustness_score;         // Robustness component
    float generalization_score;     // Generalization component
    float efficiency_score;         // Efficiency component
    float confidence;               // Confidence in evaluation [0.0, 1.0]
    std::unordered_map<std::string, float> detailed_metrics;
    
    EffectivenessResults();
};

/**
 * @brief Performance prediction model
 */
struct PredictionModel {
    std::string model_type;
    std::vector<float> weights;
    std::vector<float> biases;
    float prediction_accuracy;
    
    PredictionModel();
};

/**
 * @brief Strategy Effectiveness Evaluator
 * 
 * Evaluates and predicts the effectiveness of cognitive strategies
 * using historical performance data, context analysis, and predictive modeling.
 * 
 * Integrates with GGML tensor operations for efficient evaluation.
 */
class StrategyEffectivenessEvaluator {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit StrategyEffectivenessEvaluator(
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Destructor
     */
    ~StrategyEffectivenessEvaluator();
    
    /**
     * @brief Evaluate strategy effectiveness
     * @param strategy_name Strategy to evaluate
     * @param context_data Context-specific data
     * @param performance_history Historical performance data
     * @param criteria Evaluation criteria
     * @return Effectiveness evaluation results
     */
    EffectivenessResults evaluate_effectiveness(
        const std::string& strategy_name,
        const std::vector<float>& context_data,
        const std::vector<float>& performance_history,
        const EvaluationCriteria& criteria = EvaluationCriteria()
    );
    
    /**
     * @brief Predict strategy performance in new context
     * @param strategy_name Strategy to predict
     * @param context_data New context data
     * @return Predicted performance [0.0, 1.0]
     */
    float predict_performance(
        const std::string& strategy_name,
        const std::vector<float>& context_data
    );
    
    /**
     * @brief Compare effectiveness of multiple strategies
     * @param strategies Strategies to compare
     * @param context_data Context for comparison
     * @return Ranked list of strategies with scores
     */
    std::vector<std::pair<std::string, float>> compare_strategies(
        const std::vector<std::string>& strategies,
        const std::vector<float>& context_data
    );
    
    /**
     * @brief Update prediction model with new data
     * @param strategy_name Strategy to update
     * @param context_data Context data
     * @param actual_performance Actual observed performance
     */
    void update_prediction_model(
        const std::string& strategy_name,
        const std::vector<float>& context_data,
        float actual_performance
    );
    
    /**
     * @brief Detect performance degradation
     * @param strategy_name Strategy to check
     * @param recent_performance Recent performance data
     * @return Degradation severity [0.0, 1.0]
     */
    float detect_degradation(
        const std::string& strategy_name,
        const std::vector<float>& recent_performance
    );
    
    /**
     * @brief Get confidence in effectiveness evaluation
     * @param strategy_name Strategy to check
     * @return Confidence level [0.0, 1.0]
     */
    float get_evaluation_confidence(const std::string& strategy_name) const;
    
    /**
     * @brief Get detailed evaluation metrics
     * @param strategy_name Strategy to analyze
     * @return Detailed metrics map
     */
    std::unordered_map<std::string, float> get_detailed_metrics(
        const std::string& strategy_name
    ) const;
    
    /**
     * @brief Reset evaluator state
     */
    void reset();

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Prediction models per strategy
    std::unordered_map<std::string, PredictionModel> prediction_models_;
    
    // Performance tracking
    struct PerformanceTracker {
        std::vector<float> history;
        std::vector<std::vector<float>> context_history;
        float running_mean;
        float running_variance;
        uint64_t sample_count;
        
        PerformanceTracker();
        void update(float value, const std::vector<float>& context);
    };
    
    std::unordered_map<std::string, PerformanceTracker> performance_trackers_;
    
    // Statistics
    uint64_t total_evaluations_;
    uint64_t total_predictions_;
    
    // Internal methods
    float compute_accuracy_score(
        const std::vector<float>& performance_history
    ) const;
    
    float compute_speed_score(
        const std::vector<float>& context_data
    ) const;
    
    float compute_robustness_score(
        const std::vector<float>& performance_history
    ) const;
    
    float compute_generalization_score(
        const std::string& strategy_name,
        const std::vector<float>& context_data
    ) const;
    
    float compute_efficiency_score(
        const std::vector<float>& context_data
    ) const;
    
    float compute_overall_effectiveness(
        const EffectivenessResults& results,
        const EvaluationCriteria& criteria
    ) const;
    
    float compute_prediction_confidence(
        const PredictionModel& model,
        const std::vector<float>& context_data
    ) const;
    
    void train_prediction_model(
        PredictionModel& model,
        const std::vector<std::vector<float>>& training_contexts,
        const std::vector<float>& training_targets
    );
    
    float apply_prediction_model(
        const PredictionModel& model,
        const std::vector<float>& context_data
    ) const;
};

} // namespace rros

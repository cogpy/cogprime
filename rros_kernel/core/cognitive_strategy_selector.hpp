#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>

namespace rros {

// Forward declaration
struct MetaCognitiveStrategy;

/**
 * @brief Strategy evaluation metrics
 */
struct StrategyEvaluation {
    float expected_effectiveness;   // Predicted effectiveness
    float confidence;               // Confidence in prediction
    float cost;                     // Computational/resource cost
    float risk;                     // Risk of failure
    float alignment;                // Alignment with goals
    
    StrategyEvaluation();
    
    // Overall utility score
    float utility_score() const;
};

/**
 * @brief Strategy selection context
 */
struct SelectionContext {
    std::unordered_map<std::string, float> cognitive_state;
    std::unordered_map<std::string, float> goals;
    std::unordered_map<std::string, float> constraints;
    std::vector<float> state_vector;
    
    SelectionContext();
};

/**
 * @brief Cognitive Strategy Selector
 * 
 * Selects and optimizes meta-cognitive strategies based on:
 * - Current cognitive context
 * - Historical performance
 * - Goal alignment
 * - Resource constraints
 * 
 * Implements adaptive strategy selection using reinforcement learning
 * principles and meta-cognitive optimization.
 */
class CognitiveStrategySelector {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit CognitiveStrategySelector(
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Destructor
     */
    ~CognitiveStrategySelector();
    
    /**
     * @brief Select optimal strategy for current context
     * @param context Selection context
     * @param available_strategies Available strategies
     * @return Selected strategy and confidence
     */
    std::pair<MetaCognitiveStrategy, float> select_strategy(
        const SelectionContext& context,
        const std::vector<MetaCognitiveStrategy>& available_strategies
    );
    
    /**
     * @brief Evaluate a strategy in given context
     * @param strategy Strategy to evaluate
     * @param context Evaluation context
     * @return Evaluation metrics
     */
    StrategyEvaluation evaluate_strategy(
        const MetaCognitiveStrategy& strategy,
        const SelectionContext& context
    );
    
    /**
     * @brief Optimize strategy parameters
     * @param strategy Strategy to optimize
     * @param context Optimization context
     * @param performance_history Historical performance data
     * @return Optimized strategy
     */
    MetaCognitiveStrategy optimize_strategy(
        const MetaCognitiveStrategy& strategy,
        const SelectionContext& context,
        const std::vector<float>& performance_history
    );
    
    /**
     * @brief Update strategy effectiveness based on outcome
     * @param strategy_name Name of executed strategy
     * @param outcome Outcome quality [0.0, 1.0]
     * @param context Context where strategy was applied
     */
    void update_effectiveness(
        const std::string& strategy_name,
        float outcome,
        const SelectionContext& context
    );
    
    /**
     * @brief Learn strategy preferences from experience
     * @param experiences Vector of (context, strategy, outcome) tuples
     */
    void learn_from_experiences(
        const std::vector<std::tuple<
            SelectionContext,
            MetaCognitiveStrategy,
            float
        >>& experiences
    );
    
    /**
     * @brief Get strategy effectiveness history
     * @param strategy_name Strategy identifier
     * @return Effectiveness metrics over time
     */
    std::vector<float> get_effectiveness_history(
        const std::string& strategy_name
    ) const;
    
    /**
     * @brief Recommend strategies for context
     * @param context Current context
     * @param num_recommendations Number of strategies to recommend
     * @return Ranked list of recommended strategies
     */
    std::vector<std::pair<std::string, float>> recommend_strategies(
        const SelectionContext& context,
        size_t num_recommendations = 3
    );
    
    /**
     * @brief Detect strategy conflicts
     * @param strategies Strategies to check for conflicts
     * @return Conflict severity [0.0, 1.0]
     */
    float detect_conflicts(
        const std::vector<MetaCognitiveStrategy>& strategies
    );
    
    /**
     * @brief Get selection statistics
     * @return Current statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Reset selector state
     */
    void reset();

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Strategy performance tracking
    struct StrategyPerformance {
        std::vector<float> outcomes;
        std::vector<SelectionContext> contexts;
        float average_effectiveness;
        float variance;
        uint64_t selection_count;
        
        StrategyPerformance();
        void update(float outcome, const SelectionContext& ctx);
    };
    
    std::unordered_map<std::string, StrategyPerformance> performance_tracking_;
    
    // Context-strategy mapping (learned preferences)
    struct ContextStrategyMapping {
        std::vector<float> context_features;
        std::string preferred_strategy;
        float preference_strength;
    };
    
    std::vector<ContextStrategyMapping> learned_mappings_;
    
    // Optimization state
    struct OptimizationState {
        float learning_rate;
        float exploration_rate;
        uint64_t total_selections;
    } optimization_state_;
    
    // Statistics
    uint64_t total_selections_;
    uint64_t successful_selections_;
    
    // Internal methods
    float compute_context_similarity(
        const SelectionContext& ctx1,
        const SelectionContext& ctx2
    ) const;
    
    float compute_expected_effectiveness(
        const MetaCognitiveStrategy& strategy,
        const SelectionContext& context
    ) const;
    
    float compute_alignment_score(
        const MetaCognitiveStrategy& strategy,
        const SelectionContext& context
    ) const;
    
    std::vector<float> extract_context_features(
        const SelectionContext& context
    ) const;
    
    void update_learned_mappings(
        const SelectionContext& context,
        const MetaCognitiveStrategy& strategy,
        float outcome
    );
    
    bool should_explore() const;
    
    MetaCognitiveStrategy apply_parameter_optimization(
        const MetaCognitiveStrategy& strategy,
        const std::vector<float>& performance_gradient
    ) const;
};

} // namespace rros

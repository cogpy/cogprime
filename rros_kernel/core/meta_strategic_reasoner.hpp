#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <chrono>
#include <mutex>

namespace rros {

// Forward declarations for optional components
class StrategyEffectivenessEvaluator;

// Placeholder for future components (not yet implemented)
class StrategyTransferEngine {};
class StrategicKnowledgeBase {};

/**
 * @brief Strategic plan representation
 */
struct StrategicPlan {
    std::string plan_id;
    std::vector<std::string> strategy_sequence;
    std::unordered_map<std::string, float> expected_outcomes;
    std::unordered_map<std::string, float> resource_requirements;
    float confidence;
    float priority;
    
    StrategicPlan();
};

/**
 * @brief Strategic reasoning context
 */
struct StrategicContext {
    std::unordered_map<std::string, float> current_state;
    std::unordered_map<std::string, float> goals;
    std::unordered_map<std::string, float> constraints;
    std::unordered_map<std::string, float> environmental_demands;
    std::vector<float> performance_history;
    
    StrategicContext();
};

/**
 * @brief Strategy performance metrics
 */
struct StrategyPerformanceMetrics {
    float effectiveness_score;          // Overall effectiveness [0.0, 1.0]
    float adaptation_rate;              // Speed of adaptation
    float generalization_ability;       // Cross-domain performance
    float robustness;                   // Performance under uncertainty
    float efficiency;                   // Resource utilization
    uint64_t execution_count;
    std::chrono::high_resolution_clock::time_point last_execution;
    
    StrategyPerformanceMetrics();
};

/**
 * @brief Meta-Strategic Reasoner
 * 
 * High-level strategic planning and reasoning system that optimizes
 * cognitive strategies, learning approaches, and problem-solving methods
 * based on performance feedback and environmental demands.
 * 
 * Integrates with GGML tensor operations for high-performance strategic
 * reasoning and planning.
 */
class MetaStrategicReasoner {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit MetaStrategicReasoner(
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Destructor
     */
    ~MetaStrategicReasoner();
    
    /**
     * @brief Generate strategic plan for given context
     * @param context Strategic context
     * @param planning_horizon Time horizon for planning
     * @return Generated strategic plan
     */
    StrategicPlan generate_strategic_plan(
        const StrategicContext& context,
        float planning_horizon
    );
    
    /**
     * @brief Evaluate effectiveness of a strategic plan
     * @param plan Strategic plan to evaluate
     * @param context Evaluation context
     * @return Effectiveness metrics
     */
    StrategyPerformanceMetrics evaluate_plan_effectiveness(
        const StrategicPlan& plan,
        const StrategicContext& context
    );
    
    /**
     * @brief Adapt strategic approach based on feedback
     * @param current_plan Current strategic plan
     * @param performance_feedback Recent performance data
     * @param context Current context
     * @return Adapted strategic plan
     */
    StrategicPlan adapt_strategic_approach(
        const StrategicPlan& current_plan,
        const std::unordered_map<std::string, float>& performance_feedback,
        const StrategicContext& context
    );
    
    /**
     * @brief Optimize strategy selection for context
     * @param available_strategies Available strategies
     * @param context Decision context
     * @return Optimized strategy sequence
     */
    std::vector<std::string> optimize_strategy_selection(
        const std::vector<std::string>& available_strategies,
        const StrategicContext& context
    );
    
    /**
     * @brief Transfer strategic knowledge to new domain
     * @param source_domain Source domain
     * @param target_domain Target domain
     * @param strategies Strategies to transfer
     * @return Transfer effectiveness [0.0, 1.0]
     */
    float transfer_strategic_knowledge(
        const std::string& source_domain,
        const std::string& target_domain,
        const std::vector<std::string>& strategies
    );
    
    /**
     * @brief Learn from strategic experience
     * @param plan Executed plan
     * @param actual_outcomes Actual outcomes achieved
     * @param context Context where plan was executed
     */
    void learn_from_experience(
        const StrategicPlan& plan,
        const std::unordered_map<std::string, float>& actual_outcomes,
        const StrategicContext& context
    );
    
    /**
     * @brief Predict strategy performance in context
     * @param strategy_name Strategy to predict
     * @param context Prediction context
     * @return Predicted performance [0.0, 1.0]
     */
    float predict_strategy_performance(
        const std::string& strategy_name,
        const StrategicContext& context
    );
    
    /**
     * @brief Get strategic insights from experience
     * @param domain Domain to analyze
     * @return Strategic insights and patterns
     */
    std::unordered_map<std::string, float> get_strategic_insights(
        const std::string& domain = ""
    );
    
    /**
     * @brief Get performance metrics for strategy
     * @param strategy_name Strategy identifier
     * @return Performance metrics
     */
    StrategyPerformanceMetrics get_strategy_metrics(
        const std::string& strategy_name
    ) const;
    
    /**
     * @brief Update configuration
     * @param config New configuration parameters
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Get system statistics
     * @return Current statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Reset reasoner state
     */
    void reset();

private:
    // Subsystems
    std::unique_ptr<StrategyEffectivenessEvaluator> effectiveness_evaluator_;
    std::unique_ptr<StrategyTransferEngine> transfer_engine_;
    std::unique_ptr<StrategicKnowledgeBase> knowledge_base_;
    
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Strategy performance tracking
    std::unordered_map<std::string, StrategyPerformanceMetrics> strategy_metrics_;
    
    // Strategic experience memory
    struct StrategicExperience {
        StrategicPlan plan;
        StrategicContext context;
        std::unordered_map<std::string, float> outcomes;
        float success_score;
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    std::vector<StrategicExperience> experience_history_;
    size_t max_experience_history_;
    
    // Planning state
    struct PlanningState {
        float exploration_rate;
        float exploitation_rate;
        uint64_t total_plans_generated;
        uint64_t successful_plans;
    } planning_state_;
    
    // Statistics
    uint64_t total_strategic_decisions_;
    uint64_t successful_adaptations_;
    
    // Thread safety
    mutable std::mutex reasoner_mutex_;
    
    // Internal methods
    void initialize_subsystems();
    
    std::vector<std::string> generate_strategy_sequence(
        const StrategicContext& context,
        size_t max_length
    );
    
    float compute_plan_alignment(
        const StrategicPlan& plan,
        const StrategicContext& context
    ) const;
    
    float compute_context_similarity(
        const StrategicContext& ctx1,
        const StrategicContext& ctx2
    ) const;
    
    void update_strategy_metrics(
        const std::string& strategy_name,
        float performance,
        const StrategicContext& context
    );
    
    std::vector<StrategicExperience> find_similar_experiences(
        const StrategicContext& context,
        size_t top_k
    ) const;
    
    float estimate_transfer_effectiveness(
        const std::string& source_domain,
        const std::string& target_domain
    ) const;
    
    bool should_explore() const;
    
    void prune_experience_history();
};

} // namespace rros

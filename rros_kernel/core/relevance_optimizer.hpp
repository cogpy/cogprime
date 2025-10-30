#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <deque>

namespace rros {

/**
 * @brief Relevance scoring details
 */
struct RelevanceScore {
    float base_score;           // Base relevance [0.0, 1.0]
    float contextual_weight;    // Context-adjusted weight [0.0, 2.0]
    float goal_alignment;       // Goal relevance [0.0, 1.0]
    float salience;            // Environmental salience [0.0, 1.0]
    float temporal_relevance;  // Time-based relevance [0.0, 1.0]
    float final_score;         // Combined final score [0.0, 1.0]
    float confidence;          // Confidence in score [0.0, 1.0]
};

/**
 * @brief Goal specification for relevance alignment
 */
struct Goal {
    std::string id;
    std::vector<float> target_features;
    float priority;
    float urgency;
    std::unordered_map<std::string, float> constraints;
};

/**
 * @brief Environmental context for salience detection
 */
struct EnvironmentalContext {
    std::vector<float> current_state;
    std::vector<float> recent_changes;
    std::unordered_map<std::string, float> features;
    float novelty_score;
    float threat_level;
    float opportunity_score;
};

/**
 * @brief Relevance feedback for learning
 */
struct RelevanceFeedback {
    float predicted_relevance;
    float actual_outcome;
    float error;
    std::vector<float> context_snapshot;
    uint64_t timestamp;
};

/**
 * @brief Advanced Relevance Optimization System
 * 
 * Implements comprehensive relevance optimization mechanisms that dynamically
 * prioritize cognitive resources based on contextual importance, goal relevance,
 * and environmental demands following Vervaeke's relevance realization framework.
 */
class RelevanceOptimizer {
public:
    explicit RelevanceOptimizer(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Assess relevance with comprehensive scoring
     * @param input Input data to assess
     * @param context Current contextual information
     * @param goals Active goals for alignment
     * @param env_context Environmental context
     * @return Detailed relevance score
     */
    RelevanceScore assess_relevance(
        const std::vector<float>& input,
        const std::unordered_map<std::string, float>& context,
        const std::vector<Goal>& goals,
        const EnvironmentalContext& env_context
    );
    
    /**
     * @brief Dynamically allocate attention based on relevance
     * @param items Items to allocate attention across
     * @param scores Relevance scores for each item
     * @param budget Total attention budget available
     * @return Attention allocation weights
     */
    std::vector<float> allocate_attention_dynamic(
        const std::vector<std::vector<float>>& items,
        const std::vector<RelevanceScore>& scores,
        float budget = 1.0f
    );
    
    /**
     * @brief Compute contextual importance weights
     * @param input Input data
     * @param context Current context
     * @param history Historical context
     * @return Contextual weight multipliers
     */
    std::vector<float> compute_contextual_weights(
        const std::vector<float>& input,
        const std::unordered_map<std::string, float>& context,
        const std::vector<std::unordered_map<std::string, float>>& history
    );
    
    /**
     * @brief Align relevance with active goals
     * @param input Input to evaluate
     * @param goals Active goals to align with
     * @return Goal-alignment scores
     */
    std::vector<float> align_with_goals(
        const std::vector<float>& input,
        const std::vector<Goal>& goals
    );
    
    /**
     * @brief Detect environmental salience
     * @param env_context Current environmental context
     * @return Salience map and critical features
     */
    std::pair<std::vector<float>, std::vector<size_t>> detect_salience(
        const EnvironmentalContext& env_context
    );
    
    /**
     * @brief Retrieve memory based on relevance
     * @param query Query for retrieval
     * @param memory_pool Available memories
     * @param top_k Number of items to retrieve
     * @return Indices of most relevant memories
     */
    std::vector<size_t> retrieve_relevant_memories(
        const std::vector<float>& query,
        const std::vector<std::vector<float>>& memory_pool,
        size_t top_k = 5
    );
    
    /**
     * @brief Apply adaptive filtering based on relevance thresholds
     * @param items Items to filter
     * @param scores Relevance scores
     * @param adaptive_threshold Use adaptive or fixed threshold
     * @return Filtered items and indices
     */
    std::pair<std::vector<std::vector<float>>, std::vector<size_t>> 
    adaptive_filter(
        const std::vector<std::vector<float>>& items,
        const std::vector<RelevanceScore>& scores,
        bool adaptive_threshold = true
    );
    
    /**
     * @brief Provide feedback for learning
     * @param feedback Outcome feedback
     */
    void provide_feedback(const RelevanceFeedback& feedback);
    
    /**
     * @brief Learn from accumulated feedback
     * @return Learning metrics (error reduction, accuracy improvement)
     */
    std::unordered_map<std::string, float> learn_from_feedback();
    
    /**
     * @brief Get current relevance threshold (adaptive or fixed)
     * @return Current threshold value
     */
    float get_current_threshold() const;
    
    /**
     * @brief Get optimization metrics
     * @return Performance and efficiency metrics
     */
    std::unordered_map<std::string, float> get_metrics() const;
    
    /**
     * @brief Update configuration
     * @param config New configuration parameters
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset optimizer to initial state
     */
    void reset();

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Thresholds and parameters
    float base_threshold_;
    float adaptive_threshold_;
    float goal_weight_;
    float context_weight_;
    float salience_weight_;
    float temporal_weight_;
    
    // Learning components
    std::deque<RelevanceFeedback> feedback_history_;
    size_t max_feedback_history_;
    std::vector<float> learned_weights_;
    
    // Performance tracking
    uint64_t assessment_count_;
    float total_prediction_error_;
    float average_confidence_;
    
    // Internal methods
    float compute_base_relevance(const std::vector<float>& input);
    
    float compute_contextual_boost(
        const std::vector<float>& input,
        const std::unordered_map<std::string, float>& context
    );
    
    float compute_goal_alignment(
        const std::vector<float>& input,
        const std::vector<Goal>& goals
    );
    
    float compute_salience_score(
        const std::vector<float>& input,
        const EnvironmentalContext& env_context
    );
    
    float compute_temporal_relevance(
        const std::vector<float>& input,
        const EnvironmentalContext& env_context
    );
    
    void update_adaptive_threshold();
    
    float compute_similarity(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) const;
    
    void update_learned_weights();
};

} // namespace rros

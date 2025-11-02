#pragma once

/**
 * @file relevance_realization.hpp
 * @brief Comprehensive Vervaeke Relevance Realization Framework
 * 
 * Implements Vervaeke's relevance realization as a core cognitive mechanism
 * operating across multiple time scales and cognitive levels.
 * 
 * Based on: "Awakening from the Meaning Crisis" Episodes 27-32, 42
 * Key concepts: Salience landscaping, opponent processing, participatory knowing
 */

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <chrono>

namespace rros {

/**
 * @brief Time scale for relevance assessment
 */
enum class TimeScale {
    IMMEDIATE,      // < 100ms - sensory processing
    SHORT_TERM,     // 100ms - 5s - working memory
    MEDIUM_TERM,    // 5s - 5min - task context
    LONG_TERM,      // > 5min - goals and knowledge
    HISTORICAL      // Days/weeks - learning patterns
};

/**
 * @brief Cognitive level for relevance processing
 */
enum class CognitiveLevel {
    SENSORY,        // Raw sensory input
    PERCEPTUAL,     // Object recognition
    CONCEPTUAL,     // Abstract concepts
    GOAL_ORIENTED,  // Task goals
    WISDOM          // Meta-cognitive, meaning-making
};

/**
 * @brief Multi-scale relevance assessment result
 */
struct MultiScaleRelevance {
    std::unordered_map<TimeScale, float> time_scale_scores;
    std::unordered_map<CognitiveLevel, float> cognitive_scores;
    float integrated_relevance;      // Overall relevance score
    float confidence;                // Assessment confidence
    std::vector<size_t> critical_features;  // Most relevant feature indices
    std::chrono::microseconds processing_time;
};

/**
 * @brief Adaptive threshold configuration
 */
struct AdaptiveThreshold {
    float base_threshold;            // Base relevance threshold
    float current_threshold;         // Dynamically adjusted threshold
    float sensitivity;               // How quickly threshold adapts
    float min_threshold;             // Lower bound
    float max_threshold;             // Upper bound
    size_t adaptation_window;        // History window for adaptation
};

/**
 * @brief Relevance-guided attention directive
 */
struct AttentionDirective {
    std::vector<float> focus_weights;     // Attention weights per feature
    std::vector<size_t> priority_indices;  // Ordered by priority
    float total_allocation;                // Total attention budget used
    float focus_sharpness;                 // How concentrated the focus is
};

/**
 * @brief Memory retrieval guidance based on relevance
 */
struct MemoryGuidance {
    std::vector<float> retrieval_cues;     // Cue strengths for memory search
    std::vector<size_t> relevant_episodes; // Episode indices to retrieve
    float temporal_bias;                    // Recency vs relevance balance
    float associative_strength;             // Strength of associations
};

/**
 * @brief Knowledge integration priority
 */
struct IntegrationPriority {
    std::vector<float> knowledge_relevance;  // Relevance of each knowledge item
    std::vector<size_t> integration_order;   // Order to integrate knowledge
    float coherence_requirement;             // Required coherence for integration
    float novelty_weight;                    // Weight for novel information
};

/**
 * @brief Relevance learning feedback
 */
struct RelevanceLearningFeedback {
    float predicted_relevance;       // What was predicted
    float actual_relevance;          // What was actually relevant
    float error;                     // Prediction error
    std::vector<float> context_state; // Context when prediction made
    TimeScale time_scale;            // Time scale of feedback
    std::chrono::steady_clock::time_point timestamp;
};

/**
 * @brief Relevance propagation signal
 */
struct RelevancePropagation {
    std::string source_module;       // Where relevance originated
    std::string target_module;       // Where to propagate
    float relevance_value;           // Relevance to propagate
    float propagation_strength;      // How strongly to propagate
    std::unordered_map<std::string, float> context; // Propagation context
};

/**
 * @brief Action coupling recommendation
 */
struct ActionCoupling {
    std::vector<float> action_relevance;  // Relevance of each action
    size_t recommended_action;            // Highest relevance action
    float urgency;                        // How urgently to act
    float expected_impact;                // Expected relevance change
    float confidence;                     // Confidence in recommendation
};

/**
 * @brief Comprehensive Relevance Realization System
 * 
 * Implements Vervaeke's framework for dynamically determining relevance
 * across multiple time scales and cognitive levels, with adaptive resource
 * allocation and learning capabilities.
 */
class RelevanceRealizationSystem {
public:
    /**
     * @brief Initialize relevance realization system
     * @param config Configuration parameters
     */
    explicit RelevanceRealizationSystem(
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Perform multi-scale relevance assessment
     * @param input Input data to assess
     * @param context Current context
     * @param goals Active goals
     * @return Multi-scale relevance assessment
     */
    MultiScaleRelevance assess_multi_scale_relevance(
        const std::vector<float>& input,
        const std::unordered_map<std::string, float>& context,
        const std::vector<std::vector<float>>& goals
    );
    
    /**
     * @brief Get current adaptive threshold for time scale
     * @param scale Time scale to query
     * @return Current adaptive threshold
     */
    AdaptiveThreshold get_adaptive_threshold(TimeScale scale) const;
    
    /**
     * @brief Update adaptive thresholds based on performance
     * @param scale Time scale to update
     * @param performance Recent performance metrics
     */
    void update_adaptive_threshold(
        TimeScale scale,
        const std::vector<float>& performance
    );
    
    /**
     * @brief Generate relevance-guided attention directive
     * @param relevance Multi-scale relevance assessment
     * @param attention_budget Available attention budget
     * @return Attention allocation directive
     */
    AttentionDirective guide_attention(
        const MultiScaleRelevance& relevance,
        float attention_budget = 1.0f
    );
    
    /**
     * @brief Generate memory retrieval guidance
     * @param query Query for memory retrieval
     * @param relevance Current relevance state
     * @return Memory guidance
     */
    MemoryGuidance guide_memory_retrieval(
        const std::vector<float>& query,
        const MultiScaleRelevance& relevance
    );
    
    /**
     * @brief Determine knowledge integration priorities
     * @param knowledge_items Knowledge items to potentially integrate
     * @param current_knowledge Current knowledge state
     * @param relevance Current relevance assessment
     * @return Integration priorities
     */
    IntegrationPriority prioritize_knowledge_integration(
        const std::vector<std::vector<float>>& knowledge_items,
        const std::vector<float>& current_knowledge,
        const MultiScaleRelevance& relevance
    );
    
    /**
     * @brief Provide learning feedback for relevance system
     * @param feedback Learning feedback
     */
    void provide_learning_feedback(const RelevanceLearningFeedback& feedback);
    
    /**
     * @brief Learn from accumulated feedback
     * @return Learning metrics (accuracy, improvement, etc.)
     */
    std::unordered_map<std::string, float> learn_from_feedback();
    
    /**
     * @brief Propagate relevance to target module
     * @param propagation Propagation specification
     */
    void propagate_relevance(const RelevancePropagation& propagation);
    
    /**
     * @brief Get propagated relevance from source
     * @param source_module Source module name
     * @return Propagated relevance value
     */
    float get_propagated_relevance(const std::string& source_module) const;
    
    /**
     * @brief Generate action coupling recommendations
     * @param state Current state
     * @param available_actions Available actions
     * @param relevance Current relevance
     * @return Action coupling recommendation
     */
    ActionCoupling couple_to_action(
        const std::vector<float>& state,
        const std::vector<std::vector<float>>& available_actions,
        const MultiScaleRelevance& relevance
    );
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Get system metrics
     * @return Current metrics
     */
    std::unordered_map<std::string, float> get_metrics() const;
    
    /**
     * @brief Reset system to initial state
     */
    void reset();

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Adaptive thresholds per time scale
    std::unordered_map<TimeScale, AdaptiveThreshold> adaptive_thresholds_;
    
    // Learning feedback history
    std::vector<RelevanceLearningFeedback> feedback_history_;
    size_t max_feedback_history_;
    
    // Relevance propagation state
    std::unordered_map<std::string, float> propagated_relevance_;
    
    // Performance tracking
    size_t total_assessments_;
    float cumulative_accuracy_;
    std::chrono::steady_clock::time_point last_update_;
    
    // Internal processing methods
    float compute_immediate_relevance(
        const std::vector<float>& input,
        const std::unordered_map<std::string, float>& context
    );
    
    float compute_short_term_relevance(
        const std::vector<float>& input,
        const std::unordered_map<std::string, float>& context
    );
    
    float compute_medium_term_relevance(
        const std::vector<float>& input,
        const std::vector<std::vector<float>>& goals
    );
    
    float compute_long_term_relevance(
        const std::vector<float>& input,
        const std::vector<std::vector<float>>& goals
    );
    
    float compute_historical_relevance(
        const std::vector<float>& input
    );
    
    float compute_sensory_relevance(const std::vector<float>& input);
    float compute_perceptual_relevance(const std::vector<float>& input);
    float compute_conceptual_relevance(const std::vector<float>& input);
    float compute_goal_relevance(
        const std::vector<float>& input,
        const std::vector<std::vector<float>>& goals
    );
    float compute_wisdom_relevance(const std::vector<float>& input);
    
    void identify_critical_features(
        const std::vector<float>& input,
        std::vector<size_t>& critical_features
    );
    
    void adapt_threshold(
        AdaptiveThreshold& threshold,
        const std::vector<float>& performance
    );
    
    float integrate_multi_scale_scores(
        const std::unordered_map<TimeScale, float>& time_scores,
        const std::unordered_map<CognitiveLevel, float>& cognitive_scores
    );
    
    void update_learning_from_feedback();
};

} // namespace rros

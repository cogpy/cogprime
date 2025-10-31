#pragma once

#include "temporal_knowledge.hpp"
#include "episodic_memory.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace rros {

/**
 * @brief Detected historical pattern
 */
struct HistoricalPattern {
    std::string pattern_type;              // Pattern classification
    std::vector<size_t> event_sequence;    // Event IDs forming pattern
    float confidence;                      // Detection confidence [0.0, 1.0]
    float support;                         // Pattern support (frequency)
    uint64_t first_occurrence;             // First time pattern observed
    uint64_t last_occurrence;              // Most recent occurrence
    size_t occurrence_count;               // Total occurrences
    std::vector<float> pattern_signature;  // Pattern feature signature
};

/**
 * @brief Temporal inference rule
 */
struct TemporalRule {
    std::vector<size_t> antecedent_events; // Condition events
    std::vector<size_t> consequent_events; // Result events
    float confidence;                      // Rule confidence [0.0, 1.0]
    uint64_t typical_delay;                // Typical time delay
    size_t support_count;                  // Number of times rule held
};

/**
 * @brief Historical Context Integration System
 * 
 * Comprehensive system for integrating historical context into decision-making:
 * - Pattern recognition across temporal sequences
 * - Temporal reasoning and inference
 * - Causal relationship detection
 * - Historical context-aware predictions
 * - Decision quality improvement through historical learning
 * 
 * Target: 25% decision quality improvement through historical context
 */
class HistoricalContextSystem {
public:
    /**
     * @brief Construct historical context system
     * @param temporal_knowledge Temporal knowledge base
     * @param episodic_memory Episodic memory system
     */
    explicit HistoricalContextSystem(
        std::shared_ptr<TemporalKnowledge> temporal_knowledge,
        std::shared_ptr<EpisodicMemorySystem> episodic_memory
    );
    
    /**
     * @brief Detect patterns in event sequence
     * @param event_ids Sequence of event IDs to analyze
     * @param min_support Minimum support threshold
     * @param min_confidence Minimum confidence threshold
     * @return Detected patterns
     */
    std::vector<HistoricalPattern> detect_patterns(
        const std::vector<size_t>& event_ids,
        float min_support = 0.1f,
        float min_confidence = 0.7f
    );
    
    /**
     * @brief Learn temporal inference rules
     * @param event_window Number of recent events to analyze
     * @param min_support Minimum rule support
     * @return Number of rules learned
     */
    size_t learn_temporal_rules(
        size_t event_window = 1000,
        float min_support = 0.05f
    );
    
    /**
     * @brief Detect causal relationships in recent history
     * @param lookback_time Time window to analyze (microseconds)
     * @param min_correlation Minimum correlation threshold
     * @return Vector of detected causal relations
     */
    std::vector<CausalRelation> detect_causal_relations(
        uint64_t lookback_time,
        float min_correlation = 0.6f
    );
    
    /**
     * @brief Predict future events based on historical patterns
     * @param current_context Current context vector
     * @param horizon_time Prediction horizon (microseconds)
     * @return Predicted event vectors with confidence scores
     */
    std::vector<std::pair<std::vector<float>, float>> predict_future(
        const std::vector<float>& current_context,
        uint64_t horizon_time
    );
    
    /**
     * @brief Make decision with historical context awareness
     * @param decision_context Current decision context
     * @param alternatives Available decision alternatives
     * @return Decision scores for each alternative
     */
    std::vector<float> make_historical_decision(
        const std::vector<float>& decision_context,
        const std::vector<std::vector<float>>& alternatives
    );
    
    /**
     * @brief Validate historical knowledge consistency
     * @return Consistency score [0.0, 1.0]
     */
    float validate_consistency();
    
    /**
     * @brief Get historical pattern matching a query
     * @param query Query pattern
     * @param max_results Maximum number of results
     * @return Matching patterns
     */
    std::vector<HistoricalPattern> query_patterns(
        const std::vector<float>& query,
        size_t max_results = 5
    );
    
    /**
     * @brief Update system with new observation
     * @param observation New observation vector
     * @param outcome Observed outcome
     * @param actual_quality Actual decision quality
     */
    void update_from_observation(
        const std::vector<float>& observation,
        const std::vector<float>& outcome,
        float actual_quality
    );
    
    /**
     * @brief Get system statistics
     * @return Map of statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Clear all historical context data
     */
    void clear();

private:
    std::shared_ptr<TemporalKnowledge> temporal_knowledge_;
    std::shared_ptr<EpisodicMemorySystem> episodic_memory_;
    
    std::vector<HistoricalPattern> learned_patterns_;
    std::vector<TemporalRule> inference_rules_;
    
    // Decision quality tracking
    std::vector<float> decision_quality_history_;
    float baseline_decision_quality_;
    
    // Helper methods
    float compute_pattern_similarity(
        const HistoricalPattern& p1,
        const HistoricalPattern& p2
    ) const;
    
    float compute_sequence_coherence(
        const std::vector<size_t>& event_ids
    ) const;
    
    float estimate_causal_strength(
        size_t cause_event,
        size_t effect_event,
        uint64_t time_window
    ) const;
    
    std::vector<float> extract_pattern_features(
        const std::vector<size_t>& event_ids
    ) const;
    
    float compute_rule_confidence(
        const TemporalRule& rule,
        const std::vector<size_t>& event_sequence
    ) const;
};

} // namespace rros

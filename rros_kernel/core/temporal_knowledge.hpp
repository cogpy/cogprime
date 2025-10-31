#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <chrono>
#include <cstdint>

namespace rros {

/**
 * @brief Temporal event structure for timeline representation
 * 
 * Represents discrete events in time with associated context and relationships.
 */
struct TemporalEvent {
    uint64_t timestamp;                    // Microseconds since epoch
    std::vector<float> state_vector;       // Event state representation
    float importance;                      // Event significance [0.0, 1.0]
    std::vector<size_t> causal_predecessors; // Events that caused this event
    std::vector<size_t> causal_successors;   // Events caused by this event
    std::unordered_map<std::string, float> metadata; // Additional event metadata
};

/**
 * @brief Temporal pattern structure
 * 
 * Represents recurring patterns detected across temporal sequences.
 */
struct TemporalPattern {
    std::vector<std::vector<float>> sequence_template; // Pattern template
    float frequency;                       // Occurrence frequency
    float predictive_power;               // How well it predicts future states
    uint64_t first_observed;              // First occurrence timestamp
    uint64_t last_observed;               // Most recent occurrence timestamp
    size_t occurrence_count;              // Total occurrences
};

/**
 * @brief Causal relationship between events
 */
struct CausalRelation {
    size_t cause_event;                   // Index of causal event
    size_t effect_event;                  // Index of effect event
    float strength;                       // Causal strength [0.0, 1.0]
    uint64_t time_lag;                    // Time delay in microseconds
    float confidence;                     // Confidence in relationship [0.0, 1.0]
};

/**
 * @brief Temporal abstraction representing patterns across time scales
 */
struct TemporalAbstraction {
    std::string name;                     // Abstraction identifier
    uint64_t time_scale;                  // Time scale in microseconds
    std::vector<float> abstract_features; // High-level features
    std::vector<size_t> component_events; // Concrete events at this abstraction
    float coherence;                      // Internal coherence [0.0, 1.0]
};

/**
 * @brief Temporal Knowledge Representation Framework
 * 
 * Provides comprehensive temporal knowledge representation capabilities including:
 * - Event timeline management
 * - Temporal pattern storage
 * - Causal relationship tracking
 * - Multi-scale temporal abstractions
 * 
 * Performance target: ≤1µs event insertion, ≤5µs pattern matching
 */
class TemporalKnowledge {
public:
    /**
     * @brief Construct temporal knowledge system
     * @param event_capacity Maximum number of events to store
     * @param pattern_capacity Maximum number of patterns to store
     */
    explicit TemporalKnowledge(
        size_t event_capacity = 10000,
        size_t pattern_capacity = 1000
    );
    
    /**
     * @brief Add new temporal event
     * @param state_vector Event state representation
     * @param importance Event importance score
     * @return Event ID
     */
    size_t add_event(
        const std::vector<float>& state_vector,
        float importance
    );
    
    /**
     * @brief Add causal relationship between events
     * @param cause_id Causal event ID
     * @param effect_id Effect event ID
     * @param strength Causal strength
     * @param confidence Confidence in relationship
     */
    void add_causal_relation(
        size_t cause_id,
        size_t effect_id,
        float strength,
        float confidence
    );
    
    /**
     * @brief Register temporal pattern
     * @param pattern Pattern structure to register
     * @return Pattern ID
     */
    size_t register_pattern(const TemporalPattern& pattern);
    
    /**
     * @brief Retrieve events in time range
     * @param start_time Start timestamp
     * @param end_time End timestamp
     * @return Vector of event IDs in range
     */
    std::vector<size_t> get_events_in_range(
        uint64_t start_time,
        uint64_t end_time
    ) const;
    
    /**
     * @brief Get event by ID
     * @param event_id Event identifier
     * @return Const reference to event
     */
    const TemporalEvent& get_event(size_t event_id) const;
    
    /**
     * @brief Get causal predecessors of event
     * @param event_id Event identifier
     * @return Vector of causal relationships
     */
    std::vector<CausalRelation> get_causal_predecessors(size_t event_id) const;
    
    /**
     * @brief Get causal successors of event
     * @param event_id Event identifier
     * @return Vector of causal relationships
     */
    std::vector<CausalRelation> get_causal_successors(size_t event_id) const;
    
    /**
     * @brief Create temporal abstraction from events
     * @param event_ids Events to abstract
     * @param time_scale Time scale for abstraction
     * @return Abstraction ID
     */
    size_t create_abstraction(
        const std::vector<size_t>& event_ids,
        uint64_t time_scale
    );
    
    /**
     * @brief Get temporal statistics
     * @return Map of statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Clear all temporal knowledge
     */
    void clear();

private:
    size_t event_capacity_;
    size_t pattern_capacity_;
    
    std::vector<TemporalEvent> events_;
    std::vector<TemporalPattern> patterns_;
    std::vector<CausalRelation> causal_relations_;
    std::vector<TemporalAbstraction> abstractions_;
    
    uint64_t next_event_id_;
    uint64_t next_pattern_id_;
    
    // Helper methods
    uint64_t get_current_time_us() const;
    float compute_sequence_similarity(
        const std::vector<std::vector<float>>& seq1,
        const std::vector<std::vector<float>>& seq2
    ) const;
};

} // namespace rros

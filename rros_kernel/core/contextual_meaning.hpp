#pragma once

#include "semantic_representation.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace rros {
namespace meaning {

/**
 * @brief Context type for meaning construction
 */
enum class ContextType {
    SPATIAL,             // Physical/spatial context
    TEMPORAL,            // Time-based context
    SOCIAL,              // Social interaction context
    TASK,                // Goal/task context
    EMOTIONAL,           // Affective context
    CULTURAL,            // Cultural background
    CONVERSATIONAL,      // Dialogue context
    HISTORICAL,          // Past experience context
    SITUATIONAL          // General situation
};

/**
 * @brief Context frame representing a specific contextual situation
 */
struct ContextFrame {
    std::string frame_id;
    ContextType type;
    
    // Context parameters
    std::unordered_map<std::string, float> parameters;
    std::unordered_map<std::string, std::string> symbolic_features;
    
    // Active concepts in this context
    std::vector<std::string> salient_concepts;
    
    // Context strength and validity
    float activation;
    float relevance;
    std::chrono::system_clock::time_point timestamp;
    
    ContextFrame() : type(ContextType::SITUATIONAL), 
                    activation(1.0f), relevance(1.0f) {}
};

/**
 * @brief Situated meaning - meaning within a specific context
 */
struct SituatedMeaning {
    std::string concept_id;
    std::string context_id;
    
    // Context-specific interpretation
    std::vector<float> contextualized_embedding;
    std::unordered_map<std::string, float> situated_features;
    
    // Meaning strength in this context
    float applicability;     // How well meaning applies to context
    float specificity;       // How context-specific this meaning is
    float confidence;        // Confidence in this interpretation
    
    SituatedMeaning() : applicability(0.0f), specificity(0.0f), confidence(0.0f) {}
};

/**
 * @brief Contextual meaning construction system
 * 
 * Adapts semantic meaning based on contextual factors,
 * enabling situated understanding and interpretation
 */
class ContextualMeaning {
public:
    explicit ContextualMeaning(
        std::shared_ptr<SemanticRepresentation> semantic_system,
        const std::unordered_map<std::string, float>& config = {}
    );
    ~ContextualMeaning();
    
    // Context management
    void add_context_frame(const ContextFrame& frame);
    ContextFrame* get_context_frame(const std::string& frame_id);
    void update_context(const std::string& frame_id, 
                       const std::unordered_map<std::string, float>& updates);
    void activate_context(const std::string& frame_id, float activation = 1.0f);
    void deactivate_context(const std::string& frame_id);
    
    // Contextual interpretation
    SituatedMeaning interpret_in_context(
        const std::string& concept_id,
        const std::string& context_id
    );
    
    SituatedMeaning interpret_in_multi_context(
        const std::string& concept_id,
        const std::vector<std::string>& context_ids
    );
    
    std::vector<SituatedMeaning> generate_interpretations(
        const std::string& concept_id,
        size_t max_interpretations = 5
    );
    
    // Context-based meaning construction
    std::vector<float> contextualize_embedding(
        const std::vector<float>& base_embedding,
        const std::vector<std::string>& context_ids
    );
    
    float compute_contextual_similarity(
        const std::string& concept1,
        const std::string& concept2,
        const std::string& context_id
    );
    
    // Multi-modal context integration
    void integrate_sensory_context(
        const std::string& context_id,
        const std::vector<float>& sensory_input,
        const std::string& modality
    );
    
    void integrate_motor_context(
        const std::string& context_id,
        const std::vector<float>& motor_state
    );
    
    void integrate_emotional_context(
        const std::string& context_id,
        float valence,
        float arousal
    );
    
    // Context switching and blending
    void switch_context(
        const std::string& from_context,
        const std::string& to_context,
        float transition_speed = 0.5f
    );
    
    ContextFrame blend_contexts(
        const std::vector<std::string>& context_ids,
        const std::vector<float>& weights = {}
    );
    
    // Dynamic context tracking
    void update_context_relevance(
        const std::vector<float>& current_state
    );
    
    std::vector<std::string> get_active_contexts(
        float min_activation = 0.3f
    ) const;
    
    std::string get_dominant_context() const;
    
    // Context prediction and anticipation
    std::vector<std::string> predict_next_contexts(
        const std::string& current_context,
        size_t top_k = 3
    );
    
    float estimate_context_transition_probability(
        const std::string& from_context,
        const std::string& to_context
    );
    
    // Adaptation mechanisms
    void adapt_to_context(
        const std::string& context_id,
        const std::vector<float>& feedback
    );
    
    void learn_context_associations(
        const std::string& context_id,
        const std::vector<std::string>& associated_concepts
    );
    
    // Analysis and introspection
    float compute_context_coherence(
        const std::vector<std::string>& context_ids
    );
    
    std::unordered_map<std::string, float> analyze_context_influence(
        const std::string& concept_id
    );
    
    std::unordered_map<std::string, float> get_statistics() const;
    void clear();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::shared_ptr<SemanticRepresentation> semantic_system_;
    mutable std::mutex mutex_;
    
    // Helper methods
    float compute_context_compatibility(
        const ContextFrame& c1,
        const ContextFrame& c2
    ) const;
    
    std::vector<float> apply_context_modulation(
        const std::vector<float>& embedding,
        const ContextFrame& context
    ) const;
    
    void update_transition_probabilities(
        const std::string& from_context,
        const std::string& to_context
    );
};

} // namespace meaning
} // namespace rros

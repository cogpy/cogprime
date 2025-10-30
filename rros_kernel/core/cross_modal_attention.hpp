#pragma once

#include "unified_representation.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>

namespace rros {

/**
 * @brief Attention weights for cross-modal integration
 */
struct CrossModalAttention {
    std::unordered_map<Modality, float> modality_weights;
    std::vector<float> attention_scores;
    float integration_strength;
    
    CrossModalAttention();
};

/**
 * @brief Cross-Modal Attention and Integration
 * 
 * Implements attention mechanisms for integrating information
 * across different sensory modalities and domains, enabling
 * the system to focus on relevant cross-modal patterns.
 */
class CrossModalAttentionSystem {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit CrossModalAttentionSystem(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor
     */
    ~CrossModalAttentionSystem();
    
    /**
     * @brief Compute cross-modal attention weights
     * @param concepts Concepts from different modalities
     * @return Attention weights for each modality
     */
    CrossModalAttention compute_attention(
        const std::vector<UnifiedConcept>& concepts
    );
    
    /**
     * @brief Integrate information across modalities
     * @param concepts Multi-modal concepts
     * @param attention Attention weights
     * @return Integrated representation
     */
    UnifiedConcept integrate_modalities(
        const std::vector<UnifiedConcept>& concepts,
        const CrossModalAttention& attention
    );
    
    /**
     * @brief Focus attention on specific modality
     * @param modality Target modality
     * @param strength Focus strength [0.0, 1.0]
     */
    void focus_modality(Modality modality, float strength);
    
    /**
     * @brief Compute cross-modal correlation
     * @param concept1 First concept
     * @param concept2 Second concept
     * @return Correlation score [0.0, 1.0]
     */
    float compute_cross_modal_correlation(
        const UnifiedConcept& concept1,
        const UnifiedConcept& concept2
    );
    
    /**
     * @brief Detect cross-modal patterns
     * @param concepts Multi-modal concepts
     * @return Pattern strength score
     */
    float detect_cross_modal_patterns(
        const std::vector<UnifiedConcept>& concepts
    );
    
    /**
     * @brief Apply selective attention to concepts
     * @param concepts Input concepts
     * @param query Query vector for attention
     * @return Attended concepts
     */
    std::vector<UnifiedConcept> apply_selective_attention(
        const std::vector<UnifiedConcept>& concepts,
        const std::vector<float>& query
    );
    
    /**
     * @brief Get metrics
     */
    std::unordered_map<std::string, float> get_metrics() const;
    
    /**
     * @brief Update configuration
     */
    void update_config(const std::unordered_map<std::string, float>& new_config);
    
    /**
     * @brief Reset to initial state
     */
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mutex_;
    
    // Helper functions
    float compute_attention_score(const UnifiedConcept& concept, const std::vector<float>& query);
    std::vector<float> compute_modality_importance(const std::vector<UnifiedConcept>& concepts);
};

} // namespace rros

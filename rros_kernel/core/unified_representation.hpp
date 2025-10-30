#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>

namespace rros {

/**
 * @brief Modality types for multi-modal representation
 */
enum class Modality {
    VISUAL,
    AUDITORY,
    TEXTUAL,
    SENSORY,
    ABSTRACT,
    SYMBOLIC,
    NUMERICAL
};

/**
 * @brief Unified concept representation across domains
 * 
 * Provides a domain-agnostic representation that enables seamless
 * cross-domain reasoning and knowledge transfer.
 */
struct UnifiedConcept {
    std::string concept_id;
    std::vector<float> embedding;           // Dense semantic embedding
    std::vector<float> abstract_features;   // Domain-invariant abstract features
    std::unordered_map<std::string, float> attributes;  // Named attributes
    std::vector<Modality> modalities;       // Associated modalities
    float abstraction_level;                // 0.0 (concrete) to 1.0 (abstract)
    float confidence;                       // Representation confidence
    
    UnifiedConcept();
    UnifiedConcept(const std::string& id, const std::vector<float>& emb);
};

/**
 * @brief Unified Representation Framework
 * 
 * Manages unified representations for cross-domain concepts,
 * enabling seamless reasoning across different modalities and domains.
 */
class UnifiedRepresentation {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit UnifiedRepresentation(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor
     */
    ~UnifiedRepresentation();
    
    /**
     * @brief Create unified representation from domain-specific data
     * @param domain_data Data in domain-specific format
     * @param modality Source modality
     * @return Unified concept representation
     */
    UnifiedConcept create_representation(
        const std::vector<float>& domain_data,
        Modality modality
    );
    
    /**
     * @brief Merge representations from multiple modalities
     * @param concepts Concepts from different modalities
     * @return Merged unified representation
     */
    UnifiedConcept merge_representations(
        const std::vector<UnifiedConcept>& concepts
    );
    
    /**
     * @brief Abstract concept to higher level
     * @param concept Input concept
     * @param target_level Target abstraction level [0.0, 1.0]
     * @return Abstracted concept
     */
    UnifiedConcept abstract_concept(
        const UnifiedConcept& concept,
        float target_level
    );
    
    /**
     * @brief Ground abstract concept to specific domain
     * @param concept Abstract concept
     * @param modality Target modality
     * @return Domain-specific representation
     */
    std::vector<float> ground_to_domain(
        const UnifiedConcept& concept,
        Modality modality
    );
    
    /**
     * @brief Compute semantic similarity between concepts
     * @param c1 First concept
     * @param c2 Second concept
     * @return Similarity score [0.0, 1.0]
     */
    float compute_similarity(
        const UnifiedConcept& c1,
        const UnifiedConcept& c2
    );
    
    /**
     * @brief Align concept representations across domains
     * @param concepts Concepts to align
     * @return Aligned concepts in common space
     */
    std::vector<UnifiedConcept> align_representations(
        const std::vector<UnifiedConcept>& concepts
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
    std::vector<float> extract_abstract_features(const std::vector<float>& data);
    float compute_abstraction_level(const std::vector<float>& features);
    std::vector<float> modality_transform(const std::vector<float>& data, Modality from, Modality to);
};

} // namespace rros

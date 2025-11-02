#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <chrono>

namespace rros {
namespace meaning {

/**
 * @brief Semantic feature representing a dimension of meaning
 * 
 * Features capture distributed properties of concepts in a continuous space
 */
struct SemanticFeature {
    std::string name;
    float value;              // Feature activation strength
    float salience;           // Relevance of this feature
    
    SemanticFeature() : value(0.0f), salience(0.0f) {}
    SemanticFeature(const std::string& n, float v, float s = 1.0f) 
        : name(n), value(v), salience(s) {}
};

/**
 * @brief Relation types between semantic concepts
 */
enum class RelationType {
    IS_A,                    // Taxonomic hierarchy
    PART_OF,                 // Mereological relation
    SIMILAR_TO,              // Similarity/analogy
    OPPOSITE_OF,             // Antonymy
    CAUSES,                  // Causal relation
    REQUIRES,                // Dependency
    ASSOCIATED_WITH,         // General association
    INSTANTIATES,            // Instance relation
    PROPERTY_OF,             // Attribute relation
    ENABLES,                 // Enablement relation
    COMPOSED_OF,             // Compositional relation
    METAPHORICAL             // Metaphorical mapping
};

/**
 * @brief Semantic relation between concepts
 */
struct SemanticRelation {
    std::string source_concept;
    std::string target_concept;
    RelationType type;
    float strength;          // Relation strength [0, 1]
    float confidence;        // Confidence in relation
    
    SemanticRelation() : type(RelationType::ASSOCIATED_WITH), 
                        strength(0.0f), confidence(0.0f) {}
};

/**
 * @brief Semantic concept node with distributed representation
 * 
 * Represents a concept with both symbolic (name, relations) and
 * subsymbolic (feature vectors) properties
 */
struct ConceptNode {
    std::string concept_id;
    std::string label;
    
    // Subsymbolic representation
    std::vector<float> embedding;         // Dense vector representation
    std::vector<SemanticFeature> features; // Interpretable features
    
    // Symbolic properties
    std::unordered_set<std::string> categories;
    std::unordered_map<std::string, float> attributes;
    
    // Relations to other concepts
    std::vector<SemanticRelation> relations;
    
    // Metadata
    float activation;        // Current activation level
    float coherence;         // Internal coherence score
    size_t usage_count;      // Number of times accessed
    std::chrono::system_clock::time_point last_accessed;
    
    ConceptNode() : activation(0.0f), coherence(1.0f), usage_count(0) {}
};

/**
 * @brief Semantic representation system for meaning construction
 * 
 * Provides a hybrid symbolic-subsymbolic framework for representing
 * and manipulating semantic meaning
 */
class SemanticRepresentation {
public:
    explicit SemanticRepresentation(
        const std::unordered_map<std::string, float>& config = {}
    );
    ~SemanticRepresentation();
    
    // Concept management
    void add_concept(const ConceptNode& concept);
    ConceptNode* get_concept(const std::string& concept_id);
    bool has_concept(const std::string& concept_id) const;
    void remove_concept(const std::string& concept_id);
    
    // Relation management
    void add_relation(const SemanticRelation& relation);
    std::vector<SemanticRelation> get_relations(
        const std::string& concept_id,
        RelationType type = RelationType::ASSOCIATED_WITH
    );
    
    // Semantic operations
    float compute_similarity(
        const std::string& concept1,
        const std::string& concept2
    );
    
    std::vector<std::string> find_similar_concepts(
        const std::string& concept_id,
        size_t top_k = 10,
        float min_similarity = 0.5f
    );
    
    std::vector<std::string> get_path(
        const std::string& from_concept,
        const std::string& to_concept,
        size_t max_depth = 5
    );
    
    // Feature extraction and analysis
    std::vector<SemanticFeature> extract_features(
        const std::vector<float>& input_data,
        const std::unordered_map<std::string, float>& context = {}
    );
    
    std::vector<float> compose_embedding(
        const std::vector<std::string>& concept_ids,
        const std::string& composition_method = "average"
    );
    
    // Activation spreading
    void spread_activation(
        const std::string& source_concept,
        float initial_activation,
        size_t max_hops = 3,
        float decay = 0.8f
    );
    
    void decay_activations(float decay_rate = 0.95f);
    
    std::vector<std::string> get_activated_concepts(
        float min_activation = 0.3f
    );
    
    // Graph analysis
    float compute_centrality(const std::string& concept_id);
    
    std::vector<std::string> find_community(
        const std::string& concept_id,
        size_t max_size = 20
    );
    
    float compute_coherence(
        const std::vector<std::string>& concept_ids
    );
    
    // Semantic space operations
    std::vector<float> project_to_semantic_space(
        const std::vector<float>& input_vector
    );
    
    std::string find_nearest_concept(
        const std::vector<float>& vector
    );
    
    // Statistics and introspection
    std::unordered_map<std::string, float> get_statistics() const;
    void clear();
    void prune_unused(size_t min_usage_count = 1);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mutex_;
    
    // Helper methods
    float cosine_similarity(
        const std::vector<float>& v1,
        const std::vector<float>& v2
    ) const;
    
    std::vector<float> normalize_vector(
        const std::vector<float>& vec
    ) const;
    
    void update_coherence(ConceptNode& concept);
};

} // namespace meaning
} // namespace rros

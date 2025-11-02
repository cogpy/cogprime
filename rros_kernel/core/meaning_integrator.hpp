#pragma once

#include "semantic_representation.hpp"
#include "contextual_meaning.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

namespace rros {
namespace meaning {

/**
 * @brief Levels of meaning representation
 */
enum class MeaningLevel {
    SENSORY,          // Raw sensory/perceptual meaning
    CONCEPTUAL,       // Basic concept-level meaning
    PROPOSITIONAL,    // Relational/propositional meaning
    NARRATIVE,        // Story/narrative meaning
    ABSTRACT,         // Abstract/metaphorical meaning
    META              // Meta-level meaning about meaning
};

/**
 * @brief Hierarchical meaning structure
 */
struct MeaningHierarchy {
    std::string hierarchy_id;
    
    // Meaning at each level
    std::unordered_map<MeaningLevel, std::vector<std::string>> level_concepts;
    std::unordered_map<MeaningLevel, std::vector<float>> level_embeddings;
    
    // Inter-level connections
    std::unordered_map<std::string, std::vector<std::string>> bottom_up_links;
    std::unordered_map<std::string, std::vector<std::string>> top_down_links;
    
    // Coherence measures
    std::unordered_map<MeaningLevel, float> level_coherence;
    float global_coherence;
    
    MeaningHierarchy() : global_coherence(0.0f) {}
};

/**
 * @brief Multi-level meaning integrator
 * 
 * Integrates meaning across hierarchical levels from sensory to abstract,
 * maintaining coherence and enabling bidirectional meaning flow
 */
class MeaningIntegrator {
public:
    explicit MeaningIntegrator(
        std::shared_ptr<SemanticRepresentation> semantic_system,
        std::shared_ptr<ContextualMeaning> context_system,
        const std::unordered_map<std::string, float>& config = {}
    );
    ~MeaningIntegrator();
    
    // Hierarchy construction
    MeaningHierarchy build_hierarchy(
        const std::vector<float>& input_data,
        const std::string& context_id = ""
    );
    
    void add_to_hierarchy(
        MeaningHierarchy& hierarchy,
        MeaningLevel level,
        const std::vector<std::string>& concepts
    );
    
    // Bottom-up integration
    std::vector<std::string> integrate_bottom_up(
        const MeaningHierarchy& hierarchy,
        MeaningLevel from_level,
        MeaningLevel to_level
    );
    
    std::vector<float> compose_higher_level_meaning(
        const std::vector<std::string>& lower_concepts,
        MeaningLevel target_level
    );
    
    // Top-down modulation
    void modulate_top_down(
        MeaningHierarchy& hierarchy,
        MeaningLevel from_level,
        MeaningLevel to_level,
        float modulation_strength = 0.5f
    );
    
    std::vector<float> refine_lower_level_meaning(
        const std::vector<float>& base_embedding,
        const std::vector<std::string>& higher_concepts
    );
    
    // Coherence management
    float compute_hierarchy_coherence(
        const MeaningHierarchy& hierarchy
    );
    
    float compute_level_coherence(
        const MeaningHierarchy& hierarchy,
        MeaningLevel level
    );
    
    float compute_inter_level_coherence(
        const MeaningHierarchy& hierarchy,
        MeaningLevel level1,
        MeaningLevel level2
    );
    
    void enforce_coherence(
        MeaningHierarchy& hierarchy,
        float min_coherence = 0.7f
    );
    
    // Meaning composition and decomposition
    std::vector<std::string> compose_meaning(
        const std::vector<std::vector<std::string>>& concept_groups,
        const std::string& composition_method = "hierarchical"
    );
    
    std::vector<std::vector<std::string>> decompose_meaning(
        const std::string& complex_concept,
        size_t max_levels = 3
    );
    
    // Cross-level queries
    std::vector<std::string> query_level(
        const MeaningHierarchy& hierarchy,
        MeaningLevel level,
        const std::string& query_concept
    );
    
    std::vector<std::string> trace_to_abstract(
        const MeaningHierarchy& hierarchy,
        const std::string& concrete_concept
    );
    
    std::vector<std::string> ground_to_concrete(
        const MeaningHierarchy& hierarchy,
        const std::string& abstract_concept
    );
    
    // Integration operations
    MeaningHierarchy merge_hierarchies(
        const std::vector<MeaningHierarchy>& hierarchies,
        const std::string& merge_strategy = "union"
    );
    
    void align_hierarchies(
        MeaningHierarchy& h1,
        MeaningHierarchy& h2
    );
    
    // Adaptive operations
    void learn_level_mapping(
        MeaningLevel from_level,
        MeaningLevel to_level,
        const std::vector<std::pair<std::string, std::string>>& examples
    );
    
    void optimize_hierarchy(
        MeaningHierarchy& hierarchy,
        const std::vector<float>& feedback
    );
    
    // Statistics and introspection
    std::unordered_map<std::string, float> analyze_hierarchy(
        const MeaningHierarchy& hierarchy
    );
    
    std::unordered_map<std::string, float> get_statistics() const;
    void clear();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::shared_ptr<SemanticRepresentation> semantic_system_;
    std::shared_ptr<ContextualMeaning> context_system_;
    mutable std::mutex mutex_;
    
    // Helper methods
    std::vector<std::string> extract_concepts_at_level(
        const std::vector<float>& input,
        MeaningLevel level
    );
    
    float compute_link_strength(
        const std::string& lower_concept,
        const std::string& higher_concept
    );
};

} // namespace meaning
} // namespace rros

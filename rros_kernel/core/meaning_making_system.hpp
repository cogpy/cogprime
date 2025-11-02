#pragma once

#include "semantic_representation.hpp"
#include "contextual_meaning.hpp"
#include "meaning_integrator.hpp"
#include "symbolic_subsymbolic_bridge.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

namespace rros {
namespace meaning {

/**
 * @brief Unified meaning-making system
 * 
 * Integrates all meaning-making components into a cohesive system
 * for comprehensive semantic understanding and construction
 */
class MeaningMakingSystem {
public:
    explicit MeaningMakingSystem(
        const std::unordered_map<std::string, float>& config = {}
    );
    ~MeaningMakingSystem();
    
    // High-level meaning construction
    std::vector<std::string> construct_meaning(
        const std::vector<float>& input_data,
        const std::string& context_id = "",
        const std::unordered_map<std::string, float>& context_params = {}
    );
    
    // Multi-level understanding
    MeaningHierarchy understand_input(
        const std::vector<float>& sensory_input,
        const std::vector<std::string>& context_ids = {}
    );
    
    // Contextual interpretation
    SituatedMeaning interpret(
        const std::string& concept_id,
        const std::string& context_id
    );
    
    // Symbol grounding and translation
    void ground_concept(
        const std::string& concept_id,
        const std::vector<float>& grounding_experience
    );
    
    std::vector<std::string> translate_to_symbols(
        const std::vector<float>& distributed_input,
        size_t max_symbols = 5
    );
    
    // Coherence validation
    bool validate_meaning_coherence(
        const std::vector<std::string>& concepts,
        float min_coherence = 0.7f
    );
    
    float assess_meaning_quality(
        const MeaningHierarchy& hierarchy
    );
    
    // Adaptive refinement
    void refine_meaning(
        const std::string& concept_id,
        const std::vector<float>& feedback
    );
    
    void learn_from_feedback(
        const std::vector<std::string>& concepts,
        float quality_score
    );
    
    // System introspection
    std::unordered_map<std::string, float> get_system_statistics() const;
    std::unordered_map<std::string, float> analyze_meaning_state() const;
    
    // Component access
    std::shared_ptr<SemanticRepresentation> semantic_system() { return semantic_system_; }
    std::shared_ptr<ContextualMeaning> context_system() { return context_system_; }
    std::shared_ptr<MeaningIntegrator> integrator() { return integrator_; }
    std::shared_ptr<SymbolicSubsymbolicBridge> bridge() { return bridge_; }
    
private:
    std::shared_ptr<SemanticRepresentation> semantic_system_;
    std::shared_ptr<ContextualMeaning> context_system_;
    std::shared_ptr<MeaningIntegrator> integrator_;
    std::shared_ptr<SymbolicSubsymbolicBridge> bridge_;
    
    std::unordered_map<std::string, float> config_;
    mutable std::mutex mutex_;
};

} // namespace meaning
} // namespace rros

#include "meaning_making_system.hpp"
#include <algorithm>
#include <numeric>

namespace rros {
namespace meaning {

MeaningMakingSystem::MeaningMakingSystem(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    // Initialize all subsystems
    semantic_system_ = std::make_shared<SemanticRepresentation>(config);
    context_system_ = std::make_shared<ContextualMeaning>(semantic_system_, config);
    integrator_ = std::make_shared<MeaningIntegrator>(semantic_system_, context_system_, config);
    bridge_ = std::make_shared<SymbolicSubsymbolicBridge>(semantic_system_, config);
}

MeaningMakingSystem::~MeaningMakingSystem() = default;

std::vector<std::string> MeaningMakingSystem::construct_meaning(
    const std::vector<float>& input_data,
    const std::string& context_id,
    const std::unordered_map<std::string, float>& context_params
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 1. Extract semantic features
    auto features = semantic_system_->extract_features(input_data, context_params);
    
    // 2. Project to semantic space
    auto semantic_vector = semantic_system_->project_to_semantic_space(input_data);
    
    // 3. Translate to symbols via bridge
    auto symbols = bridge_->vector_to_symbols(semantic_vector, 5, 0.3f);
    
    // 4. If context provided, contextualize
    if (!context_id.empty() && context_system_->get_context_frame(context_id)) {
        auto contextualized = context_system_->contextualize_embedding(
            semantic_vector, {context_id}
        );
        auto context_symbols = bridge_->vector_to_symbols(contextualized, 3, 0.4f);
        symbols.insert(symbols.end(), context_symbols.begin(), context_symbols.end());
    }
    
    // 5. Remove duplicates
    std::sort(symbols.begin(), symbols.end());
    symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());
    
    return symbols;
}

MeaningHierarchy MeaningMakingSystem::understand_input(
    const std::vector<float>& sensory_input,
    const std::vector<std::string>& context_ids
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Build multi-level meaning hierarchy
    std::string primary_context = context_ids.empty() ? "" : context_ids[0];
    auto hierarchy = integrator_->build_hierarchy(sensory_input, primary_context);
    
    // Enhance with contextual information
    if (!context_ids.empty()) {
        auto contextualized = context_system_->contextualize_embedding(
            sensory_input, context_ids
        );
        
        // Add contextualized concepts to conceptual level
        auto symbols = bridge_->vector_to_symbols(contextualized, 5);
        integrator_->add_to_hierarchy(hierarchy, MeaningLevel::CONCEPTUAL, symbols);
    }
    
    // Ensure coherence across levels
    integrator_->enforce_coherence(hierarchy);
    
    return hierarchy;
}

SituatedMeaning MeaningMakingSystem::interpret(
    const std::string& concept_id,
    const std::string& context_id
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return context_system_->interpret_in_context(concept_id, context_id);
}

void MeaningMakingSystem::ground_concept(
    const std::string& concept_id,
    const std::vector<float>& grounding_experience
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Ground symbol in bridge
    bridge_->ground_symbol(concept_id, grounding_experience);
    
    // Update semantic representation
    ConceptNode concept;
    concept.concept_id = concept_id;
    concept.label = concept_id;
    concept.embedding = grounding_experience;
    concept.activation = 1.0f;
    
    semantic_system_->add_concept(concept);
}

std::vector<std::string> MeaningMakingSystem::translate_to_symbols(
    const std::vector<float>& distributed_input,
    size_t max_symbols
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return bridge_->distributed_to_symbolic(distributed_input, max_symbols);
}

bool MeaningMakingSystem::validate_meaning_coherence(
    const std::vector<std::string>& concepts,
    float min_coherence
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    float coherence = semantic_system_->compute_coherence(concepts);
    return coherence >= min_coherence;
}

float MeaningMakingSystem::assess_meaning_quality(
    const MeaningHierarchy& hierarchy
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Assess based on coherence and completeness
    float coherence = integrator_->compute_hierarchy_coherence(hierarchy);
    
    // Count filled levels
    size_t filled_levels = 0;
    for (const auto& [level, concepts] : hierarchy.level_concepts) {
        if (!concepts.empty()) filled_levels++;
    }
    
    float completeness = static_cast<float>(filled_levels) / 6.0f; // 6 levels total
    
    return 0.7f * coherence + 0.3f * completeness;
}

void MeaningMakingSystem::refine_meaning(
    const std::string& concept_id,
    const std::vector<float>& feedback
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update grounding with feedback
    bridge_->update_grounding(concept_id, feedback, 0.1f);
    
    // Update concept embedding
    auto* concept = semantic_system_->get_concept(concept_id);
    if (concept) {
        for (size_t i = 0; i < std::min(concept->embedding.size(), feedback.size()); ++i) {
            concept->embedding[i] = concept->embedding[i] * 0.9f + feedback[i] * 0.1f;
        }
    }
}

void MeaningMakingSystem::learn_from_feedback(
    const std::vector<std::string>& concepts,
    float quality_score
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // If quality is high, strengthen relations between concepts
    if (quality_score > 0.7f) {
        for (size_t i = 0; i < concepts.size(); ++i) {
            for (size_t j = i + 1; j < concepts.size(); ++j) {
                SemanticRelation rel;
                rel.source_concept = concepts[i];
                rel.target_concept = concepts[j];
                rel.type = RelationType::ASSOCIATED_WITH;
                rel.strength = quality_score * 0.5f;
                rel.confidence = quality_score;
                
                semantic_system_->add_relation(rel);
            }
        }
    }
}

std::unordered_map<std::string, float> MeaningMakingSystem::get_system_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto semantic_stats = semantic_system_->get_statistics();
    auto context_stats = context_system_->get_statistics();
    auto integrator_stats = integrator_->get_statistics();
    auto bridge_stats = bridge_->get_statistics();
    
    std::unordered_map<std::string, float> combined;
    
    // Merge all statistics
    for (const auto& [key, value] : semantic_stats) {
        combined["semantic_" + key] = value;
    }
    for (const auto& [key, value] : context_stats) {
        combined["context_" + key] = value;
    }
    for (const auto& [key, value] : integrator_stats) {
        combined["integrator_" + key] = value;
    }
    for (const auto& [key, value] : bridge_stats) {
        combined["bridge_" + key] = value;
    }
    
    return combined;
}

std::unordered_map<std::string, float> MeaningMakingSystem::analyze_meaning_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto stats = get_system_statistics();
    
    // Compute derived metrics
    float total_concepts = stats.count("semantic_total_concepts") ? 
        stats["semantic_total_concepts"] : 0.0f;
    float total_groundings = stats.count("bridge_total_groundings") ?
        stats["bridge_total_groundings"] : 0.0f;
    float total_contexts = stats.count("context_total_contexts") ?
        stats["context_total_contexts"] : 0.0f;
    
    stats["grounding_ratio"] = (total_concepts > 0) ? 
        (total_groundings / total_concepts) : 0.0f;
    stats["context_coverage"] = (total_concepts > 0) ?
        (total_contexts / total_concepts) : 0.0f;
    
    return stats;
}

} // namespace meaning
} // namespace rros

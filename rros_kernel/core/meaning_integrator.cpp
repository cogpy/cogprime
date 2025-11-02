#include "meaning_integrator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {
namespace meaning {

struct MeaningIntegrator::Impl {
    std::unordered_map<std::string, float> config;
    
    // Learned level mappings
    std::unordered_map<MeaningLevel, 
        std::unordered_map<MeaningLevel, 
            std::vector<std::pair<std::string, std::string>>>> level_mappings;
    
    // Statistics
    size_t hierarchies_built;
    size_t bottom_up_integrations;
    size_t top_down_modulations;
    size_t coherence_enforcements;
    
    Impl() : hierarchies_built(0), bottom_up_integrations(0),
            top_down_modulations(0), coherence_enforcements(0) {
        config["min_coherence"] = 0.7f;
        config["composition_weight"] = 0.6f;
        config["modulation_strength"] = 0.5f;
    }
};

MeaningIntegrator::MeaningIntegrator(
    std::shared_ptr<SemanticRepresentation> semantic_system,
    std::shared_ptr<ContextualMeaning> context_system,
    const std::unordered_map<std::string, float>& config
) : impl_(std::make_unique<Impl>()),
    semantic_system_(semantic_system),
    context_system_(context_system) {
    for (const auto& [key, value] : config) {
        impl_->config[key] = value;
    }
}

MeaningIntegrator::~MeaningIntegrator() = default;

MeaningHierarchy MeaningIntegrator::build_hierarchy(
    const std::vector<float>& input_data,
    const std::string& context_id
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    MeaningHierarchy hierarchy;
    hierarchy.hierarchy_id = "hierarchy_" + std::to_string(impl_->hierarchies_built);
    
    // Extract concepts at each level
    for (int i = static_cast<int>(MeaningLevel::SENSORY); 
         i <= static_cast<int>(MeaningLevel::META); ++i) {
        MeaningLevel level = static_cast<MeaningLevel>(i);
        auto concepts = extract_concepts_at_level(input_data, level);
        hierarchy.level_concepts[level] = concepts;
        
        // Compute level embedding
        if (!concepts.empty()) {
            auto embedding = semantic_system_->compose_embedding(concepts);
            hierarchy.level_embeddings[level] = embedding;
        }
    }
    
    impl_->hierarchies_built++;
    return hierarchy;
}

void MeaningIntegrator::add_to_hierarchy(
    MeaningHierarchy& hierarchy,
    MeaningLevel level,
    const std::vector<std::string>& concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    hierarchy.level_concepts[level].insert(
        hierarchy.level_concepts[level].end(),
        concepts.begin(), concepts.end()
    );
    
    // Recompute embedding
    if (!hierarchy.level_concepts[level].empty()) {
        hierarchy.level_embeddings[level] = 
            semantic_system_->compose_embedding(hierarchy.level_concepts[level]);
    }
}

std::vector<std::string> MeaningIntegrator::integrate_bottom_up(
    const MeaningHierarchy& hierarchy,
    MeaningLevel from_level,
    MeaningLevel to_level
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> integrated;
    
    auto from_it = hierarchy.level_concepts.find(from_level);
    if (from_it == hierarchy.level_concepts.end()) {
        return integrated;
    }
    
    // Group lower-level concepts into higher-level ones
    const auto& lower_concepts = from_it->second;
    
    // Simple clustering: group by similarity
    std::vector<bool> used(lower_concepts.size(), false);
    
    for (size_t i = 0; i < lower_concepts.size(); ++i) {
        if (used[i]) continue;
        
        std::vector<std::string> cluster;
        cluster.push_back(lower_concepts[i]);
        used[i] = true;
        
        // Find similar concepts
        for (size_t j = i + 1; j < lower_concepts.size(); ++j) {
            if (!used[j]) {
                float sim = semantic_system_->compute_similarity(
                    lower_concepts[i], lower_concepts[j]
                );
                if (sim > 0.7f) {
                    cluster.push_back(lower_concepts[j]);
                    used[j] = true;
                }
            }
        }
        
        // Create higher-level concept from cluster
        if (!cluster.empty()) {
            std::string higher_concept = "integrated_" + 
                std::to_string(integrated.size());
            integrated.push_back(higher_concept);
        }
    }
    
    const_cast<Impl*>(impl_.get())->bottom_up_integrations++;
    return integrated;
}

std::vector<float> MeaningIntegrator::compose_higher_level_meaning(
    const std::vector<std::string>& lower_concepts,
    MeaningLevel target_level
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Compose embedding with level-specific weighting
    auto base_embedding = semantic_system_->compose_embedding(lower_concepts);
    
    // Apply level-specific transformation
    float level_factor = 1.0f + 0.1f * static_cast<int>(target_level);
    for (float& v : base_embedding) {
        v *= level_factor;
    }
    
    return base_embedding;
}

void MeaningIntegrator::modulate_top_down(
    MeaningHierarchy& hierarchy,
    MeaningLevel from_level,
    MeaningLevel to_level,
    float modulation_strength
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto from_it = hierarchy.level_embeddings.find(from_level);
    auto to_it = hierarchy.level_embeddings.find(to_level);
    
    if (from_it != hierarchy.level_embeddings.end() &&
        to_it != hierarchy.level_embeddings.end()) {
        
        const auto& higher_emb = from_it->second;
        auto& lower_emb = to_it->second;
        
        // Apply modulation
        for (size_t i = 0; i < std::min(higher_emb.size(), lower_emb.size()); ++i) {
            lower_emb[i] = lower_emb[i] * (1.0f - modulation_strength) +
                          higher_emb[i] * modulation_strength;
        }
    }
    
    impl_->top_down_modulations++;
}

float MeaningIntegrator::compute_hierarchy_coherence(
    const MeaningHierarchy& hierarchy
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    float total_coherence = 0.0f;
    size_t count = 0;
    
    // Compute coherence at each level
    for (const auto& [level, concepts] : hierarchy.level_concepts) {
        if (!concepts.empty()) {
            float level_coh = semantic_system_->compute_coherence(concepts);
            total_coherence += level_coh;
            count++;
        }
    }
    
    return count > 0 ? total_coherence / count : 0.0f;
}

float MeaningIntegrator::compute_level_coherence(
    const MeaningHierarchy& hierarchy,
    MeaningLevel level
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = hierarchy.level_concepts.find(level);
    if (it != hierarchy.level_concepts.end() && !it->second.empty()) {
        return semantic_system_->compute_coherence(it->second);
    }
    return 0.0f;
}

void MeaningIntegrator::enforce_coherence(
    MeaningHierarchy& hierarchy,
    float min_coherence
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check and fix coherence at each level
    for (auto& [level, concepts] : hierarchy.level_concepts) {
        if (concepts.size() > 1) {
            float coherence = semantic_system_->compute_coherence(concepts);
            
            if (coherence < min_coherence) {
                // Remove least coherent concepts
                std::vector<std::pair<float, std::string>> concept_scores;
                
                for (const auto& concept : concepts) {
                    float avg_sim = 0.0f;
                    for (const auto& other : concepts) {
                        if (concept != other) {
                            avg_sim += semantic_system_->compute_similarity(concept, other);
                        }
                    }
                    avg_sim /= (concepts.size() - 1);
                    concept_scores.push_back({avg_sim, concept});
                }
                
                // Sort by score
                std::sort(concept_scores.begin(), concept_scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                
                // Keep top concepts
                concepts.clear();
                size_t keep_count = std::max(size_t(1), concept_scores.size() * 2 / 3);
                for (size_t i = 0; i < keep_count; ++i) {
                    concepts.push_back(concept_scores[i].second);
                }
            }
        }
    }
    
    impl_->coherence_enforcements++;
}

std::vector<std::string> MeaningIntegrator::compose_meaning(
    const std::vector<std::vector<std::string>>& concept_groups,
    const std::string& composition_method
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> composed;
    
    if (composition_method == "hierarchical") {
        // Build hierarchy from groups
        for (size_t i = 0; i < concept_groups.size(); ++i) {
            if (!concept_groups[i].empty()) {
                std::string composite = "composite_" + std::to_string(composed.size());
                composed.push_back(composite);
            }
        }
    } else {
        // Flat composition
        for (const auto& group : concept_groups) {
            composed.insert(composed.end(), group.begin(), group.end());
        }
    }
    
    return composed;
}

std::vector<std::vector<std::string>> MeaningIntegrator::decompose_meaning(
    const std::string& complex_concept,
    size_t max_levels
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::vector<std::string>> decomposed(max_levels);
    
    // Get relations to decompose
    auto relations = semantic_system_->get_relations(complex_concept);
    
    size_t current_level = 0;
    std::vector<std::string> to_process = {complex_concept};
    
    while (current_level < max_levels && !to_process.empty()) {
        std::vector<std::string> next_level;
        
        for (const auto& concept : to_process) {
            auto rels = semantic_system_->get_relations(concept, RelationType::PART_OF);
            for (const auto& rel : rels) {
                decomposed[current_level].push_back(rel.target_concept);
                next_level.push_back(rel.target_concept);
            }
        }
        
        to_process = next_level;
        current_level++;
    }
    
    return decomposed;
}

std::unordered_map<std::string, float> MeaningIntegrator::analyze_hierarchy(
    const MeaningHierarchy& hierarchy
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, float> analysis;
    
    analysis["total_levels"] = static_cast<float>(hierarchy.level_concepts.size());
    analysis["global_coherence"] = compute_hierarchy_coherence(hierarchy);
    
    size_t total_concepts = 0;
    for (const auto& [level, concepts] : hierarchy.level_concepts) {
        total_concepts += concepts.size();
    }
    analysis["total_concepts"] = static_cast<float>(total_concepts);
    analysis["avg_concepts_per_level"] = total_concepts / 
        std::max(size_t(1), hierarchy.level_concepts.size());
    
    return analysis;
}

std::unordered_map<std::string, float> MeaningIntegrator::get_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return {
        {"hierarchies_built", static_cast<float>(impl_->hierarchies_built)},
        {"bottom_up_integrations", static_cast<float>(impl_->bottom_up_integrations)},
        {"top_down_modulations", static_cast<float>(impl_->top_down_modulations)},
        {"coherence_enforcements", static_cast<float>(impl_->coherence_enforcements)}
    };
}

void MeaningIntegrator::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->level_mappings.clear();
}

std::vector<std::string> MeaningIntegrator::extract_concepts_at_level(
    const std::vector<float>& input,
    MeaningLevel level
) {
    // Extract features appropriate for this level
    std::vector<std::string> concepts;
    
    // Simple level-based concept generation
    size_t num_concepts = 1 + static_cast<size_t>(level);
    for (size_t i = 0; i < num_concepts; ++i) {
        std::string concept = "concept_L" + std::to_string(static_cast<int>(level)) + 
                             "_" + std::to_string(i);
        concepts.push_back(concept);
    }
    
    return concepts;
}

float MeaningIntegrator::compute_link_strength(
    const std::string& lower_concept,
    const std::string& higher_concept
) {
    return semantic_system_->compute_similarity(lower_concept, higher_concept);
}

std::vector<float> MeaningIntegrator::refine_lower_level_meaning(
    const std::vector<float>& base_embedding,
    const std::vector<std::string>& higher_concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto higher_embedding = semantic_system_->compose_embedding(higher_concepts);
    std::vector<float> refined = base_embedding;
    
    float blend_factor = impl_->config["modulation_strength"];
    for (size_t i = 0; i < std::min(refined.size(), higher_embedding.size()); ++i) {
        refined[i] = refined[i] * (1.0f - blend_factor) + 
                    higher_embedding[i] * blend_factor;
    }
    
    return refined;
}

std::vector<std::string> MeaningIntegrator::query_level(
    const MeaningHierarchy& hierarchy,
    MeaningLevel level,
    const std::string& query_concept
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = hierarchy.level_concepts.find(level);
    if (it != hierarchy.level_concepts.end()) {
        return it->second;
    }
    return {};
}

std::vector<std::string> MeaningIntegrator::trace_to_abstract(
    const MeaningHierarchy& hierarchy,
    const std::string& concrete_concept
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> trace;
    trace.push_back(concrete_concept);
    
    // Follow bottom_up_links
    std::string current = concrete_concept;
    while (hierarchy.bottom_up_links.count(current)) {
        const auto& targets = hierarchy.bottom_up_links.at(current);
        if (!targets.empty()) {
            current = targets[0];
            trace.push_back(current);
        } else {
            break;
        }
    }
    
    return trace;
}

MeaningHierarchy MeaningIntegrator::merge_hierarchies(
    const std::vector<MeaningHierarchy>& hierarchies,
    const std::string& merge_strategy
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    MeaningHierarchy merged;
    merged.hierarchy_id = "merged_" + std::to_string(impl_->hierarchies_built);
    
    if (hierarchies.empty()) return merged;
    
    // Merge concepts at each level
    for (const auto& h : hierarchies) {
        for (const auto& [level, concepts] : h.level_concepts) {
            merged.level_concepts[level].insert(
                merged.level_concepts[level].end(),
                concepts.begin(), concepts.end()
            );
        }
    }
    
    // Recompute embeddings
    for (auto& [level, concepts] : merged.level_concepts) {
        if (!concepts.empty()) {
            merged.level_embeddings[level] = 
                semantic_system_->compose_embedding(concepts);
        }
    }
    
    return merged;
}

void MeaningIntegrator::learn_level_mapping(
    MeaningLevel from_level,
    MeaningLevel to_level,
    const std::vector<std::pair<std::string, std::string>>& examples
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    impl_->level_mappings[from_level][to_level] = examples;
}

void MeaningIntegrator::optimize_hierarchy(
    MeaningHierarchy& hierarchy,
    const std::vector<float>& feedback
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Simple optimization: adjust based on feedback
    if (feedback.empty()) return;
    
    float avg_feedback = std::accumulate(feedback.begin(), feedback.end(), 0.0f) / 
                        feedback.size();
    
    if (avg_feedback < 0.5f) {
        // Poor feedback: try to improve coherence
        enforce_coherence(hierarchy, impl_->config["min_coherence"]);
    }
}

float MeaningIntegrator::compute_inter_level_coherence(
    const MeaningHierarchy& hierarchy,
    MeaningLevel level1,
    MeaningLevel level2
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto emb1_it = hierarchy.level_embeddings.find(level1);
    auto emb2_it = hierarchy.level_embeddings.find(level2);
    
    if (emb1_it != hierarchy.level_embeddings.end() &&
        emb2_it != hierarchy.level_embeddings.end()) {
        
        // Compute embedding similarity
        const auto& e1 = emb1_it->second;
        const auto& e2 = emb2_it->second;
        
        if (e1.size() != e2.size()) return 0.0f;
        
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (size_t i = 0; i < e1.size(); ++i) {
            dot += e1[i] * e2[i];
            norm1 += e1[i] * e1[i];
            norm2 += e2[i] * e2[i];
        }
        
        float denom = std::sqrt(norm1) * std::sqrt(norm2);
        return (denom > 1e-8f) ? (dot / denom) : 0.0f;
    }
    
    return 0.0f;
}

std::vector<std::string> MeaningIntegrator::ground_to_concrete(
    const MeaningHierarchy& hierarchy,
    const std::string& abstract_concept
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> grounding;
    
    // Follow top_down_links
    if (hierarchy.top_down_links.count(abstract_concept)) {
        const auto& targets = hierarchy.top_down_links.at(abstract_concept);
        grounding.insert(grounding.end(), targets.begin(), targets.end());
    }
    
    return grounding;
}

void MeaningIntegrator::align_hierarchies(
    MeaningHierarchy& h1,
    MeaningHierarchy& h2
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Align concepts at each level
    for (auto& [level, concepts1] : h1.level_concepts) {
        if (h2.level_concepts.count(level)) {
            auto& concepts2 = h2.level_concepts[level];
            
            // Find mappings between concepts
            for (const auto& c1 : concepts1) {
                for (const auto& c2 : concepts2) {
                    float sim = semantic_system_->compute_similarity(c1, c2);
                    if (sim > 0.8f) {
                        // Create alignment link
                        h1.bottom_up_links[c1].push_back(c2);
                        h2.bottom_up_links[c2].push_back(c1);
                    }
                }
            }
        }
    }
}

} // namespace meaning
} // namespace rros

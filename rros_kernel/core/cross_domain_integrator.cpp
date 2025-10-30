#include "cross_domain_integrator.hpp"
#include "meta_learning_engine.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>

namespace rros {

// Implementation struct
struct CrossDomainIntegrator::Impl {
    std::unordered_map<std::string, float> config;
    std::unordered_map<std::string, Domain> domains;
    std::unordered_map<std::string, KnowledgeNode> knowledge_graph;
    std::vector<AbstractMapping> mappings;
    
    // Metrics
    size_t domains_registered;
    size_t mappings_created;
    size_t reasonings_performed;
    size_t integrations_performed;
    size_t adaptations_performed;
    size_t validations_performed;
    size_t graph_queries;
    
    Impl() : domains_registered(0), mappings_created(0), reasonings_performed(0),
             integrations_performed(0), adaptations_performed(0), 
             validations_performed(0), graph_queries(0) {
        // Default configuration
        config["adaptation_threshold"] = 0.9f;
        config["consistency_threshold"] = 0.85f;
        config["mapping_confidence_threshold"] = 0.7f;
        config["reasoning_depth"] = 3.0f;
        config["integration_weight"] = 0.8f;
        config["graph_search_depth"] = 5.0f;
    }
};

// AbstractMapping implementation
AbstractMapping::AbstractMapping() : mapping_confidence(0.0f) {}

// KnowledgeNode implementation
KnowledgeNode::KnowledgeNode() {}

// CrossDomainIntegrator implementation
CrossDomainIntegrator::CrossDomainIntegrator(const std::unordered_map<std::string, float>& config)
    : impl_(std::make_unique<Impl>()),
      unified_repr_(std::make_unique<UnifiedRepresentation>()),
      attention_system_(std::make_unique<CrossModalAttentionSystem>()) {
    if (!config.empty()) {
        for (const auto& [key, value] : config) {
            impl_->config[key] = value;
        }
    }
}

CrossDomainIntegrator::~CrossDomainIntegrator() = default;

void CrossDomainIntegrator::register_domain(const Domain& domain) {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->domains[domain.name] = domain;
    impl_->domains_registered++;
}

AbstractMapping CrossDomainIntegrator::create_abstract_mapping(
    const std::string& source_domain,
    const std::string& target_domain
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    AbstractMapping mapping;
    mapping.source_domain = source_domain;
    mapping.target_domain = target_domain;
    
    // Check if domains are registered
    if (impl_->domains.find(source_domain) == impl_->domains.end() ||
        impl_->domains.find(target_domain) == impl_->domains.end()) {
        mapping.mapping_confidence = 0.0f;
        return mapping;
    }
    
    const auto& source = impl_->domains[source_domain];
    const auto& target = impl_->domains[target_domain];
    
    // Compute domain similarity for mapping confidence
    float similarity = compute_domain_similarity(source_domain, target_domain);
    mapping.mapping_confidence = similarity;
    
    // Create concept mappings based on feature similarity
    // In a real implementation, this would use more sophisticated alignment
    size_t common_features = std::min(source.feature_space.size(), target.feature_space.size());
    
    for (size_t i = 0; i < common_features; ++i) {
        std::string source_concept = "concept_" + std::to_string(i) + "_" + source_domain;
        std::string target_concept = "concept_" + std::to_string(i) + "_" + target_domain;
        
        // Map if features are similar enough
        if (std::abs(source.feature_space[i] - target.feature_space[i]) < 0.3f) {
            mapping.concept_mappings[source_concept] = target_concept;
        }
    }
    
    impl_->mappings.push_back(mapping);
    impl_->mappings_created++;
    
    return mapping;
}

UnifiedConcept CrossDomainIntegrator::cross_domain_reasoning(
    const std::vector<UnifiedConcept>& concepts,
    const std::vector<float>& query
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concepts.empty()) {
        return UnifiedConcept();
    }
    
    // Apply cross-modal attention
    auto attention = attention_system_->compute_attention(concepts);
    
    // Select relevant concepts using attention
    auto relevant_concepts = attention_system_->apply_selective_attention(concepts, query);
    
    // Integrate across domains
    UnifiedConcept reasoning_result = attention_system_->integrate_modalities(
        relevant_concepts, attention);
    
    // Enhance with abstract reasoning
    float target_abstraction = std::min(1.0f, reasoning_result.abstraction_level + 0.2f);
    reasoning_result = unified_repr_->abstract_concept(reasoning_result, target_abstraction);
    
    impl_->reasonings_performed++;
    
    return reasoning_result;
}

UnifiedConcept CrossDomainIntegrator::integrate_cross_domain_knowledge(
    const std::unordered_map<std::string, std::vector<UnifiedConcept>>& domain_concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (domain_concepts.empty()) {
        return UnifiedConcept();
    }
    
    // Collect all concepts
    std::vector<UnifiedConcept> all_concepts;
    for (const auto& [domain, concepts] : domain_concepts) {
        for (const auto& concept : concepts) {
            all_concepts.push_back(concept);
        }
    }
    
    // Align representations across domains
    auto aligned_concepts = unified_repr_->align_representations(all_concepts);
    
    // Merge aligned concepts
    UnifiedConcept integrated = unified_repr_->merge_representations(aligned_concepts);
    
    // Set higher abstraction level due to cross-domain integration
    integrated.abstraction_level = std::min(1.0f, integrated.abstraction_level + 0.15f);
    
    impl_->integrations_performed++;
    
    return integrated;
}

UnifiedConcept CrossDomainIntegrator::translate_to_general(
    const std::vector<float>& domain_knowledge,
    const std::string& source_domain
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create unified representation from domain-specific knowledge
    Modality modality = Modality::ABSTRACT;  // Default modality
    
    // Determine modality based on domain name
    if (source_domain.find("visual") != std::string::npos) {
        modality = Modality::VISUAL;
    } else if (source_domain.find("audio") != std::string::npos) {
        modality = Modality::AUDITORY;
    } else if (source_domain.find("text") != std::string::npos) {
        modality = Modality::TEXTUAL;
    }
    
    UnifiedConcept general = unified_repr_->create_representation(domain_knowledge, modality);
    
    // Abstract to general level
    general = unified_repr_->abstract_concept(general, 0.8f);
    
    return general;
}

std::vector<float> CrossDomainIntegrator::translate_to_domain(
    const UnifiedConcept& general_knowledge,
    const std::string& target_domain
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Determine target modality
    Modality target_modality = Modality::ABSTRACT;
    
    if (target_domain.find("visual") != std::string::npos) {
        target_modality = Modality::VISUAL;
    } else if (target_domain.find("audio") != std::string::npos) {
        target_modality = Modality::AUDITORY;
    } else if (target_domain.find("text") != std::string::npos) {
        target_modality = Modality::TEXTUAL;
    }
    
    // Ground to domain
    std::vector<float> domain_specific = unified_repr_->ground_to_domain(
        general_knowledge, target_modality);
    
    return domain_specific;
}

void CrossDomainIntegrator::add_knowledge_node(const KnowledgeNode& node) {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->knowledge_graph[node.node_id] = node;
}

void CrossDomainIntegrator::connect_nodes(
    const std::string& node1_id,
    const std::string& node2_id,
    float weight
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (impl_->knowledge_graph.find(node1_id) != impl_->knowledge_graph.end() &&
        impl_->knowledge_graph.find(node2_id) != impl_->knowledge_graph.end()) {
        
        auto& node1 = impl_->knowledge_graph[node1_id];
        auto& node2 = impl_->knowledge_graph[node2_id];
        
        node1.edges.push_back(node2_id);
        node1.edge_weights[node2_id] = weight;
        
        node2.edges.push_back(node1_id);
        node2.edge_weights[node1_id] = weight;
    }
}

std::vector<KnowledgeNode> CrossDomainIntegrator::query_knowledge_graph(
    const std::vector<float>& query,
    size_t max_results
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::pair<float, KnowledgeNode>> scored_nodes;
    
    // Score all nodes
    for (const auto& [id, node] : impl_->knowledge_graph) {
        // Compute similarity between query and node concept
        float score = 0.0f;
        size_t min_size = std::min(query.size(), node.concept.embedding.size());
        
        for (size_t i = 0; i < min_size; ++i) {
            score += query[i] * node.concept.embedding[i];
        }
        
        score *= node.concept.confidence;
        scored_nodes.push_back({score, node});
    }
    
    // Sort by score (descending)
    std::sort(scored_nodes.begin(), scored_nodes.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return top results
    std::vector<KnowledgeNode> results;
    for (size_t i = 0; i < std::min(max_results, scored_nodes.size()); ++i) {
        results.push_back(scored_nodes[i].second);
    }
    
    impl_->graph_queries++;
    
    return results;
}

float CrossDomainIntegrator::validate_consistency(
    const std::vector<UnifiedConcept>& concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concepts.size() < 2) {
        return 1.0f;  // Single concept is always consistent
    }
    
    float total_consistency = 0.0f;
    int pairs = 0;
    
    // Check pairwise consistency
    for (size_t i = 0; i < concepts.size(); ++i) {
        for (size_t j = i + 1; j < concepts.size(); ++j) {
            bool consistent = check_logical_consistency(concepts[i], concepts[j]);
            
            if (consistent) {
                // Compute semantic similarity as consistency measure
                float similarity = unified_repr_->compute_similarity(concepts[i], concepts[j]);
                total_consistency += similarity;
            }
            
            pairs++;
        }
    }
    
    float consistency_score = pairs > 0 ? total_consistency / pairs : 1.0f;
    
    impl_->validations_performed++;
    
    return consistency_score;
}

float CrossDomainIntegrator::adapt_domain(
    const std::string& source_domain,
    const std::vector<float>& target_context
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (impl_->domains.find(source_domain) == impl_->domains.end()) {
        return 0.0f;
    }
    
    auto& domain = impl_->domains[source_domain];
    
    // Compute adaptation by aligning domain features to target context
    float adaptation_score = 0.0f;
    size_t min_size = std::min(domain.feature_space.size(), target_context.size());
    
    if (min_size > 0) {
        // Compute how well domain can adapt to target context
        float norm_domain = 0.0f;
        float norm_target = 0.0f;
        float dot_product = 0.0f;
        
        for (size_t i = 0; i < min_size; ++i) {
            dot_product += domain.feature_space[i] * target_context[i];
            norm_domain += domain.feature_space[i] * domain.feature_space[i];
            norm_target += target_context[i] * target_context[i];
        }
        
        norm_domain = std::sqrt(norm_domain);
        norm_target = std::sqrt(norm_target);
        
        if (norm_domain > 1e-6f && norm_target > 1e-6f) {
            adaptation_score = dot_product / (norm_domain * norm_target);
            
            // Apply adaptation threshold
            if (adaptation_score >= impl_->config["adaptation_threshold"]) {
                // Update domain features towards target (learning)
                float adaptation_rate = 0.1f;
                for (size_t i = 0; i < min_size; ++i) {
                    domain.feature_space[i] += (target_context[i] - domain.feature_space[i]) * adaptation_rate;
                }
            }
        }
    }
    
    impl_->adaptations_performed++;
    
    return std::max(0.0f, adaptation_score);
}

std::unordered_map<std::string, float> CrossDomainIntegrator::get_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto metrics = impl_->config;  // Include config as metrics
    metrics["domains_registered"] = static_cast<float>(impl_->domains_registered);
    metrics["mappings_created"] = static_cast<float>(impl_->mappings_created);
    metrics["reasonings_performed"] = static_cast<float>(impl_->reasonings_performed);
    metrics["integrations_performed"] = static_cast<float>(impl_->integrations_performed);
    metrics["adaptations_performed"] = static_cast<float>(impl_->adaptations_performed);
    metrics["validations_performed"] = static_cast<float>(impl_->validations_performed);
    metrics["graph_queries"] = static_cast<float>(impl_->graph_queries);
    metrics["knowledge_nodes"] = static_cast<float>(impl_->knowledge_graph.size());
    
    return metrics;
}

std::unordered_map<std::string, float> CrossDomainIntegrator::get_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Build stats manually to avoid calling get_metrics() which would deadlock
    std::unordered_map<std::string, float> stats = impl_->config;  
    stats["domains_registered"] = static_cast<float>(impl_->domains_registered);
    stats["mappings_created"] = static_cast<float>(impl_->mappings_created);
    stats["reasonings_performed"] = static_cast<float>(impl_->reasonings_performed);
    stats["integrations_performed"] = static_cast<float>(impl_->integrations_performed);
    stats["adaptations_performed"] = static_cast<float>(impl_->adaptations_performed);
    stats["validations_performed"] = static_cast<float>(impl_->validations_performed);
    stats["graph_queries"] = static_cast<float>(impl_->graph_queries);
    stats["knowledge_nodes"] = static_cast<float>(impl_->knowledge_graph.size());
    
    // Get subsystem metrics
    auto unified_metrics = unified_repr_->get_metrics();
    auto attention_metrics = attention_system_->get_metrics();
    
    for (const auto& [key, value] : unified_metrics) {
        stats["unified_" + key] = value;
    }
    
    for (const auto& [key, value] : attention_metrics) {
        stats["attention_" + key] = value;
    }
    
    return stats;
}

void CrossDomainIntegrator::update_config(const std::unordered_map<std::string, float>& new_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [key, value] : new_config) {
        impl_->config[key] = value;
    }
    
    // Update subsystems
    unified_repr_->update_config(new_config);
    attention_system_->update_config(new_config);
}

void CrossDomainIntegrator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->domains.clear();
    impl_->knowledge_graph.clear();
    impl_->mappings.clear();
    impl_->domains_registered = 0;
    impl_->mappings_created = 0;
    impl_->reasonings_performed = 0;
    impl_->integrations_performed = 0;
    impl_->adaptations_performed = 0;
    impl_->validations_performed = 0;
    impl_->graph_queries = 0;
    
    unified_repr_->reset();
    attention_system_->reset();
}

// Private helper functions
float CrossDomainIntegrator::compute_domain_similarity(
    const std::string& domain1,
    const std::string& domain2
) {
    if (impl_->domains.find(domain1) == impl_->domains.end() ||
        impl_->domains.find(domain2) == impl_->domains.end()) {
        return 0.0f;
    }
    
    const auto& d1 = impl_->domains[domain1];
    const auto& d2 = impl_->domains[domain2];
    
    // Compute cosine similarity on domain features
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    size_t min_size = std::min(d1.feature_space.size(), d2.feature_space.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += d1.feature_space[i] * d2.feature_space[i];
        norm1 += d1.feature_space[i] * d1.feature_space[i];
        norm2 += d2.feature_space[i] * d2.feature_space[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 < 1e-6f || norm2 < 1e-6f) {
        return 0.0f;
    }
    
    return dot_product / (norm1 * norm2);
}

std::vector<UnifiedConcept> CrossDomainIntegrator::align_domain_concepts(
    const std::vector<UnifiedConcept>& concepts,
    const std::string& target_domain
) {
    // Delegate to unified representation system
    return unified_repr_->align_representations(concepts);
}

bool CrossDomainIntegrator::check_logical_consistency(
    const UnifiedConcept& c1,
    const UnifiedConcept& c2
) {
    // Check if concepts are logically consistent
    // In a real implementation, this would use formal logic
    
    // Simple heuristic: concepts are consistent if they're not contradictory
    // (i.e., not negatively correlated in abstract feature space)
    
    if (c1.abstract_features.empty() || c2.abstract_features.empty()) {
        return true;  // Can't determine, assume consistent
    }
    
    float correlation = 0.0f;
    size_t min_size = std::min(c1.abstract_features.size(), c2.abstract_features.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        correlation += c1.abstract_features[i] * c2.abstract_features[i];
    }
    
    // Normalize
    float norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < min_size; ++i) {
        norm1 += c1.abstract_features[i] * c1.abstract_features[i];
        norm2 += c2.abstract_features[i] * c2.abstract_features[i];
    }
    
    if (norm1 > 1e-6f && norm2 > 1e-6f) {
        correlation /= std::sqrt(norm1 * norm2);
    }
    
    // Concepts are inconsistent if highly negatively correlated
    return correlation > -0.7f;
}

} // namespace rros

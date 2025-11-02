#include "semantic_representation.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <set>

namespace rros {
namespace meaning {

namespace {
    // Helper: Compute cosine similarity between vectors
    float vector_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
        
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return (denom > 1e-8f) ? (dot / denom) : 0.0f;
    }
    
    // Helper: Normalize vector to unit length
    std::vector<float> normalize(const std::vector<float>& vec) {
        std::vector<float> result = vec;
        float norm = std::sqrt(std::accumulate(vec.begin(), vec.end(), 0.0f,
            [](float sum, float v) { return sum + v * v; }));
        
        if (norm > 1e-8f) {
            for (float& v : result) v /= norm;
        }
        return result;
    }
}

// Implementation struct
struct SemanticRepresentation::Impl {
    std::unordered_map<std::string, ConceptNode> concepts;
    std::unordered_map<std::string, float> config;
    
    // Statistics
    size_t total_concepts;
    size_t total_relations;
    size_t total_activations;
    
    Impl() : total_concepts(0), total_relations(0), total_activations(0) {
        config["embedding_dim"] = 128.0f;
        config["activation_threshold"] = 0.3f;
        config["similarity_threshold"] = 0.5f;
        config["max_relations_per_concept"] = 100.0f;
    }
};

SemanticRepresentation::SemanticRepresentation(
    const std::unordered_map<std::string, float>& config
) : impl_(std::make_unique<Impl>()) {
    for (const auto& [key, value] : config) {
        impl_->config[key] = value;
    }
}

SemanticRepresentation::~SemanticRepresentation() = default;

void SemanticRepresentation::add_concept(const ConceptNode& concept) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concept.concept_id.empty()) return;
    
    impl_->concepts[concept.concept_id] = concept;
    impl_->concepts[concept.concept_id].last_accessed = 
        std::chrono::system_clock::now();
    impl_->total_concepts++;
}

ConceptNode* SemanticRepresentation::get_concept(const std::string& concept_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->concepts.find(concept_id);
    if (it != impl_->concepts.end()) {
        it->second.usage_count++;
        it->second.last_accessed = std::chrono::system_clock::now();
        return &(it->second);
    }
    return nullptr;
}

bool SemanticRepresentation::has_concept(const std::string& concept_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->concepts.find(concept_id) != impl_->concepts.end();
}

void SemanticRepresentation::remove_concept(const std::string& concept_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->concepts.find(concept_id);
    if (it != impl_->concepts.end()) {
        // Remove relations pointing to this concept
        for (auto& [id, concept] : impl_->concepts) {
            concept.relations.erase(
                std::remove_if(concept.relations.begin(), concept.relations.end(),
                    [&concept_id](const SemanticRelation& rel) {
                        return rel.target_concept == concept_id || 
                               rel.source_concept == concept_id;
                    }),
                concept.relations.end()
            );
        }
        
        impl_->concepts.erase(it);
    }
}

void SemanticRepresentation::add_relation(const SemanticRelation& relation) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->concepts.find(relation.source_concept);
    if (it != impl_->concepts.end()) {
        // Check if relation already exists
        bool exists = false;
        for (auto& rel : it->second.relations) {
            if (rel.target_concept == relation.target_concept && 
                rel.type == relation.type) {
                // Update existing relation
                rel.strength = std::max(rel.strength, relation.strength);
                rel.confidence = std::max(rel.confidence, relation.confidence);
                exists = true;
                break;
            }
        }
        
        if (!exists) {
            it->second.relations.push_back(relation);
            impl_->total_relations++;
        }
    }
}

std::vector<SemanticRelation> SemanticRepresentation::get_relations(
    const std::string& concept_id,
    RelationType type
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<SemanticRelation> result;
    auto it = impl_->concepts.find(concept_id);
    if (it != impl_->concepts.end()) {
        for (const auto& rel : it->second.relations) {
            if (type == RelationType::ASSOCIATED_WITH || rel.type == type) {
                result.push_back(rel);
            }
        }
    }
    return result;
}

float SemanticRepresentation::compute_similarity(
    const std::string& concept1,
    const std::string& concept2
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it1 = impl_->concepts.find(concept1);
    auto it2 = impl_->concepts.find(concept2);
    
    if (it1 == impl_->concepts.end() || it2 == impl_->concepts.end()) {
        return 0.0f;
    }
    
    // Compute embedding similarity
    float embedding_sim = vector_cosine_similarity(
        it1->second.embedding, it2->second.embedding
    );
    
    // Compute relation-based similarity
    float relation_sim = 0.0f;
    size_t shared_targets = 0;
    std::unordered_set<std::string> targets1;
    for (const auto& rel : it1->second.relations) {
        targets1.insert(rel.target_concept);
    }
    for (const auto& rel : it2->second.relations) {
        if (targets1.find(rel.target_concept) != targets1.end()) {
            shared_targets++;
        }
    }
    
    size_t total_targets = it1->second.relations.size() + it2->second.relations.size();
    if (total_targets > 0) {
        relation_sim = 2.0f * shared_targets / total_targets;
    }
    
    // Weighted combination
    return 0.7f * embedding_sim + 0.3f * relation_sim;
}

std::vector<std::string> SemanticRepresentation::find_similar_concepts(
    const std::string& concept_id,
    size_t top_k,
    float min_similarity
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::pair<float, std::string>> similarities;
    
    auto target_it = impl_->concepts.find(concept_id);
    if (target_it == impl_->concepts.end()) {
        return {};
    }
    
    for (const auto& [id, concept] : impl_->concepts) {
        if (id == concept_id) continue;
        
        float sim = vector_cosine_similarity(
            target_it->second.embedding, concept.embedding
        );
        
        if (sim >= min_similarity) {
            similarities.push_back({sim, id});
        }
    }
    
    // Sort by similarity (descending)
    std::sort(similarities.begin(), similarities.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return top-k
    std::vector<std::string> result;
    for (size_t i = 0; i < std::min(top_k, similarities.size()); ++i) {
        result.push_back(similarities[i].second);
    }
    
    return result;
}

std::vector<std::string> SemanticRepresentation::get_path(
    const std::string& from_concept,
    const std::string& to_concept,
    size_t max_depth
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!impl_->concepts.count(from_concept) || !impl_->concepts.count(to_concept)) {
        return {};
    }
    
    // BFS to find shortest path
    std::queue<std::string> queue;
    std::unordered_map<std::string, std::string> parent;
    std::unordered_set<std::string> visited;
    
    queue.push(from_concept);
    visited.insert(from_concept);
    parent[from_concept] = "";
    
    size_t depth = 0;
    while (!queue.empty() && depth < max_depth) {
        size_t level_size = queue.size();
        
        for (size_t i = 0; i < level_size; ++i) {
            std::string current = queue.front();
            queue.pop();
            
            if (current == to_concept) {
                // Reconstruct path
                std::vector<std::string> path;
                std::string node = to_concept;
                while (!node.empty()) {
                    path.push_back(node);
                    node = parent[node];
                }
                std::reverse(path.begin(), path.end());
                return path;
            }
            
            // Explore neighbors
            auto it = impl_->concepts.find(current);
            if (it != impl_->concepts.end()) {
                for (const auto& rel : it->second.relations) {
                    if (visited.find(rel.target_concept) == visited.end()) {
                        visited.insert(rel.target_concept);
                        parent[rel.target_concept] = current;
                        queue.push(rel.target_concept);
                    }
                }
            }
        }
        depth++;
    }
    
    return {}; // No path found
}

std::vector<SemanticFeature> SemanticRepresentation::extract_features(
    const std::vector<float>& input_data,
    const std::unordered_map<std::string, float>& context
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<SemanticFeature> features;
    
    if (input_data.empty()) return features;
    
    // Extract statistical features
    float mean = std::accumulate(input_data.begin(), input_data.end(), 0.0f) / input_data.size();
    float variance = 0.0f;
    for (float v : input_data) {
        variance += (v - mean) * (v - mean);
    }
    variance /= input_data.size();
    float std_dev = std::sqrt(variance);
    
    features.emplace_back("magnitude", mean, 1.0f);
    features.emplace_back("variability", std_dev, 0.8f);
    features.emplace_back("complexity", std_dev / (mean + 1e-6f), 0.7f);
    
    // Add context-based features
    for (const auto& [key, value] : context) {
        features.emplace_back("context_" + key, value, 0.6f);
    }
    
    return features;
}

std::vector<float> SemanticRepresentation::compose_embedding(
    const std::vector<std::string>& concept_ids,
    const std::string& composition_method
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concept_ids.empty()) return {};
    
    size_t embedding_dim = static_cast<size_t>(impl_->config["embedding_dim"]);
    std::vector<float> result(embedding_dim, 0.0f);
    
    size_t valid_concepts = 0;
    for (const auto& id : concept_ids) {
        auto it = impl_->concepts.find(id);
        if (it != impl_->concepts.end() && !it->second.embedding.empty()) {
            for (size_t i = 0; i < std::min(embedding_dim, it->second.embedding.size()); ++i) {
                if (composition_method == "sum" || composition_method == "average") {
                    result[i] += it->second.embedding[i];
                } else if (composition_method == "max") {
                    result[i] = std::max(result[i], it->second.embedding[i]);
                }
            }
            valid_concepts++;
        }
    }
    
    // Average for "average" method
    if (composition_method == "average" && valid_concepts > 0) {
        for (float& v : result) {
            v /= valid_concepts;
        }
    }
    
    return result;
}

void SemanticRepresentation::spread_activation(
    const std::string& source_concept,
    float initial_activation,
    size_t max_hops,
    float decay
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->concepts.find(source_concept);
    if (it == impl_->concepts.end()) return;
    
    // Reset activations
    for (auto& [id, concept] : impl_->concepts) {
        concept.activation = 0.0f;
    }
    
    // Initialize source
    it->second.activation = initial_activation;
    
    // Spread activation using BFS
    std::queue<std::pair<std::string, size_t>> queue;
    queue.push({source_concept, 0});
    
    while (!queue.empty()) {
        auto [current_id, depth] = queue.front();
        queue.pop();
        
        if (depth >= max_hops) continue;
        
        auto current_it = impl_->concepts.find(current_id);
        if (current_it == impl_->concepts.end()) continue;
        
        float current_activation = current_it->second.activation;
        
        // Spread to neighbors
        for (const auto& rel : current_it->second.relations) {
            auto neighbor_it = impl_->concepts.find(rel.target_concept);
            if (neighbor_it != impl_->concepts.end()) {
                float spread_amount = current_activation * decay * rel.strength;
                neighbor_it->second.activation = std::max(
                    neighbor_it->second.activation, spread_amount
                );
                
                if (spread_amount > impl_->config["activation_threshold"]) {
                    queue.push({rel.target_concept, depth + 1});
                }
            }
        }
    }
    
    impl_->total_activations++;
}

void SemanticRepresentation::decay_activations(float decay_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [id, concept] : impl_->concepts) {
        concept.activation *= decay_rate;
    }
}

std::vector<std::string> SemanticRepresentation::get_activated_concepts(
    float min_activation
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> result;
    for (const auto& [id, concept] : impl_->concepts) {
        if (concept.activation >= min_activation) {
            result.push_back(id);
        }
    }
    
    // Sort by activation (descending)
    std::sort(result.begin(), result.end(), [this](const auto& a, const auto& b) {
        return impl_->concepts[a].activation > impl_->concepts[b].activation;
    });
    
    return result;
}

float SemanticRepresentation::compute_centrality(const std::string& concept_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->concepts.find(concept_id);
    if (it == impl_->concepts.end()) return 0.0f;
    
    // Compute degree centrality (normalized by total concepts)
    size_t outgoing = it->second.relations.size();
    
    // Count incoming relations
    size_t incoming = 0;
    for (const auto& [id, concept] : impl_->concepts) {
        for (const auto& rel : concept.relations) {
            if (rel.target_concept == concept_id) {
                incoming++;
            }
        }
    }
    
    size_t total_edges = outgoing + incoming;
    size_t max_possible = (impl_->concepts.size() - 1) * 2; // Max in + out
    
    return max_possible > 0 ? static_cast<float>(total_edges) / max_possible : 0.0f;
}

std::vector<std::string> SemanticRepresentation::find_community(
    const std::string& concept_id,
    size_t max_size
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> community;
    if (!impl_->concepts.count(concept_id)) return community;
    
    // Simple community detection: connected components via strong relations
    std::unordered_set<std::string> visited;
    std::queue<std::string> queue;
    
    queue.push(concept_id);
    visited.insert(concept_id);
    
    while (!queue.empty() && community.size() < max_size) {
        std::string current = queue.front();
        queue.pop();
        community.push_back(current);
        
        auto it = impl_->concepts.find(current);
        if (it != impl_->concepts.end()) {
            for (const auto& rel : it->second.relations) {
                if (rel.strength >= 0.7f && 
                    visited.find(rel.target_concept) == visited.end()) {
                    visited.insert(rel.target_concept);
                    queue.push(rel.target_concept);
                }
            }
        }
    }
    
    return community;
}

float SemanticRepresentation::compute_coherence(
    const std::vector<std::string>& concept_ids
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concept_ids.size() < 2) return 1.0f;
    
    float total_similarity = 0.0f;
    size_t count = 0;
    
    for (size_t i = 0; i < concept_ids.size(); ++i) {
        for (size_t j = i + 1; j < concept_ids.size(); ++j) {
            auto it1 = impl_->concepts.find(concept_ids[i]);
            auto it2 = impl_->concepts.find(concept_ids[j]);
            
            if (it1 != impl_->concepts.end() && it2 != impl_->concepts.end()) {
                total_similarity += vector_cosine_similarity(
                    it1->second.embedding, it2->second.embedding
                );
                count++;
            }
        }
    }
    
    return count > 0 ? total_similarity / count : 0.0f;
}

std::vector<float> SemanticRepresentation::project_to_semantic_space(
    const std::vector<float>& input_vector
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Simple projection: normalize and pad/truncate to embedding dimension
    size_t embedding_dim = static_cast<size_t>(impl_->config["embedding_dim"]);
    std::vector<float> projected(embedding_dim, 0.0f);
    
    auto normalized = normalize(input_vector);
    for (size_t i = 0; i < std::min(embedding_dim, normalized.size()); ++i) {
        projected[i] = normalized[i];
    }
    
    return projected;
}

std::string SemanticRepresentation::find_nearest_concept(
    const std::vector<float>& vector
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string nearest_id;
    float max_similarity = -1.0f;
    
    for (const auto& [id, concept] : impl_->concepts) {
        if (!concept.embedding.empty()) {
            float sim = vector_cosine_similarity(vector, concept.embedding);
            if (sim > max_similarity) {
                max_similarity = sim;
                nearest_id = id;
            }
        }
    }
    
    return nearest_id;
}

std::unordered_map<std::string, float> SemanticRepresentation::get_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return {
        {"total_concepts", static_cast<float>(impl_->concepts.size())},
        {"total_relations", static_cast<float>(impl_->total_relations)},
        {"total_activations", static_cast<float>(impl_->total_activations)},
        {"avg_relations_per_concept", impl_->concepts.empty() ? 0.0f : 
            static_cast<float>(impl_->total_relations) / impl_->concepts.size()}
    };
}

void SemanticRepresentation::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->concepts.clear();
    impl_->total_concepts = 0;
    impl_->total_relations = 0;
    impl_->total_activations = 0;
}

void SemanticRepresentation::prune_unused(size_t min_usage_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> to_remove;
    for (const auto& [id, concept] : impl_->concepts) {
        if (concept.usage_count < min_usage_count) {
            to_remove.push_back(id);
        }
    }
    
    for (const auto& id : to_remove) {
        impl_->concepts.erase(id);
    }
}

float SemanticRepresentation::cosine_similarity(
    const std::vector<float>& v1,
    const std::vector<float>& v2
) const {
    return vector_cosine_similarity(v1, v2);
}

std::vector<float> SemanticRepresentation::normalize_vector(
    const std::vector<float>& vec
) const {
    return normalize(vec);
}

void SemanticRepresentation::update_coherence(ConceptNode& concept) {
    if (concept.embedding.empty()) {
        concept.coherence = 1.0f;
        return;
    }
    
    // Compute self-consistency of features and embedding
    float variance = 0.0f;
    for (float v : concept.embedding) {
        variance += v * v;
    }
    variance /= concept.embedding.size();
    
    // High variance can indicate rich representation, normalize appropriately
    concept.coherence = 1.0f / (1.0f + std::abs(variance - 0.5f));
}

} // namespace meaning
} // namespace rros

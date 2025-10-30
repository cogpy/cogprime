#include "cross_modal_attention.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace rros {

// Implementation struct
struct CrossModalAttentionSystem::Impl {
    std::unordered_map<std::string, float> config;
    std::unordered_map<Modality, float> modality_focus;
    
    // Metrics
    size_t attention_computations;
    size_t integrations_performed;
    size_t patterns_detected;
    
    Impl() : attention_computations(0), integrations_performed(0), patterns_detected(0) {
        // Default configuration
        config["attention_temperature"] = 2.0f;
        config["integration_threshold"] = 0.5f;
        config["pattern_threshold"] = 0.7f;
        config["attention_dropout"] = 0.1f;
        config["focus_strength"] = 1.0f;
    }
};

// CrossModalAttention implementation
CrossModalAttention::CrossModalAttention() : integration_strength(1.0f) {}

// CrossModalAttentionSystem implementation
CrossModalAttentionSystem::CrossModalAttentionSystem(const std::unordered_map<std::string, float>& config)
    : impl_(std::make_unique<Impl>()) {
    if (!config.empty()) {
        for (const auto& [key, value] : config) {
            impl_->config[key] = value;
        }
    }
}

CrossModalAttentionSystem::~CrossModalAttentionSystem() = default;

CrossModalAttention CrossModalAttentionSystem::compute_attention(
    const std::vector<UnifiedConcept>& concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    CrossModalAttention attention;
    
    if (concepts.empty()) {
        return attention;
    }
    
    // Compute modality importance
    std::unordered_map<Modality, float> modality_scores;
    std::unordered_map<Modality, int> modality_counts;
    
    for (const auto& concept : concepts) {
        for (Modality m : concept.modalities) {
            modality_scores[m] += concept.confidence;
            modality_counts[m]++;
        }
    }
    
    // Normalize and apply focus
    float total_score = 0.0f;
    for (auto& [modality, score] : modality_scores) {
        score /= modality_counts[modality];
        
        // Apply modality focus if set
        if (impl_->modality_focus.count(modality)) {
            score *= impl_->modality_focus[modality];
        }
        
        total_score += score;
    }
    
    // Normalize to attention weights
    if (total_score > 0.0f) {
        for (auto& [modality, score] : modality_scores) {
            attention.modality_weights[modality] = score / total_score;
        }
    }
    
    // Compute per-concept attention scores
    attention.attention_scores.resize(concepts.size());
    for (size_t i = 0; i < concepts.size(); ++i) {
        float score = concepts[i].confidence;
        
        // Boost score for focused modalities
        for (Modality m : concepts[i].modalities) {
            if (attention.modality_weights.count(m)) {
                score *= (1.0f + attention.modality_weights[m]);
            }
        }
        
        attention.attention_scores[i] = score;
    }
    
    // Apply softmax with temperature
    float temperature = impl_->config["attention_temperature"];
    float max_score = *std::max_element(attention.attention_scores.begin(), attention.attention_scores.end());
    float sum_exp = 0.0f;
    
    for (auto& score : attention.attention_scores) {
        score = std::exp((score - max_score) / temperature);
        sum_exp += score;
    }
    
    for (auto& score : attention.attention_scores) {
        score /= sum_exp;
    }
    
    // Compute integration strength
    attention.integration_strength = std::min(1.0f, 
        static_cast<float>(modality_scores.size()) / 5.0f);
    
    impl_->attention_computations++;
    
    return attention;
}

UnifiedConcept CrossModalAttentionSystem::integrate_modalities(
    const std::vector<UnifiedConcept>& concepts,
    const CrossModalAttention& attention
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concepts.empty()) {
        return UnifiedConcept();
    }
    
    UnifiedConcept integrated;
    integrated.concept_id = "integrated_" + std::to_string(impl_->integrations_performed);
    
    // Determine output dimensionality
    size_t max_dim = 0;
    for (const auto& c : concepts) {
        max_dim = std::max(max_dim, c.embedding.size());
    }
    
    integrated.embedding.resize(max_dim, 0.0f);
    integrated.abstract_features.resize(max_dim / 2, 0.0f);
    
    // Weighted integration using attention scores
    float total_weight = 0.0f;
    
    for (size_t i = 0; i < concepts.size(); ++i) {
        float weight = (i < attention.attention_scores.size()) ? 
                      attention.attention_scores[i] : (1.0f / concepts.size());
        
        for (size_t j = 0; j < concepts[i].embedding.size(); ++j) {
            integrated.embedding[j] += concepts[i].embedding[j] * weight;
        }
        
        for (size_t j = 0; j < concepts[i].abstract_features.size() && 
                           j < integrated.abstract_features.size(); ++j) {
            integrated.abstract_features[j] += concepts[i].abstract_features[j] * weight;
        }
        
        total_weight += weight;
        
        // Collect all modalities
        for (Modality m : concepts[i].modalities) {
            if (std::find(integrated.modalities.begin(), integrated.modalities.end(), m) 
                == integrated.modalities.end()) {
                integrated.modalities.push_back(m);
            }
        }
    }
    
    // Normalize
    if (total_weight > 0.0f) {
        for (auto& val : integrated.embedding) {
            val /= total_weight;
        }
        for (auto& val : integrated.abstract_features) {
            val /= total_weight;
        }
    }
    
    // Set abstraction level (higher due to integration)
    integrated.abstraction_level = 0.0f;
    for (const auto& c : concepts) {
        integrated.abstraction_level += c.abstraction_level;
    }
    integrated.abstraction_level = std::min(1.0f, 
        integrated.abstraction_level / concepts.size() + 0.1f);
    
    // Set confidence based on integration strength
    integrated.confidence = attention.integration_strength * 
                           (total_weight / concepts.size());
    
    impl_->integrations_performed++;
    
    return integrated;
}

void CrossModalAttentionSystem::focus_modality(Modality modality, float strength) {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->modality_focus[modality] = std::max(0.0f, std::min(2.0f, strength));
}

float CrossModalAttentionSystem::compute_cross_modal_correlation(
    const UnifiedConcept& concept1,
    const UnifiedConcept& concept2
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if concepts have different modalities
    bool different_modalities = false;
    for (Modality m1 : concept1.modalities) {
        bool found = false;
        for (Modality m2 : concept2.modalities) {
            if (m1 == m2) {
                found = true;
                break;
            }
        }
        if (!found) {
            different_modalities = true;
            break;
        }
    }
    
    if (!different_modalities) {
        return 0.0f;  // Same modalities, not cross-modal
    }
    
    // Compute correlation on abstract features
    float correlation = 0.0f;
    size_t min_size = std::min(concept1.abstract_features.size(), 
                               concept2.abstract_features.size());
    
    if (min_size > 0) {
        float mean1 = std::accumulate(concept1.abstract_features.begin(), 
                                     concept1.abstract_features.begin() + min_size, 0.0f) / min_size;
        float mean2 = std::accumulate(concept2.abstract_features.begin(), 
                                     concept2.abstract_features.begin() + min_size, 0.0f) / min_size;
        
        float cov = 0.0f;
        float var1 = 0.0f;
        float var2 = 0.0f;
        
        for (size_t i = 0; i < min_size; ++i) {
            float diff1 = concept1.abstract_features[i] - mean1;
            float diff2 = concept2.abstract_features[i] - mean2;
            cov += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }
        
        if (var1 > 1e-6f && var2 > 1e-6f) {
            correlation = std::abs(cov / std::sqrt(var1 * var2));
        }
    }
    
    return correlation;
}

float CrossModalAttentionSystem::detect_cross_modal_patterns(
    const std::vector<UnifiedConcept>& concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concepts.size() < 2) {
        return 0.0f;
    }
    
    // Detect patterns by computing pairwise correlations WITHOUT recursive locks
    float total_correlation = 0.0f;
    int cross_modal_pairs = 0;
    
    for (size_t i = 0; i < concepts.size(); ++i) {
        for (size_t j = i + 1; j < concepts.size(); ++j) {
            // Inline correlation computation to avoid recursive mutex lock
            const auto& concept1 = concepts[i];
            const auto& concept2 = concepts[j];
            
            // Check if concepts have different modalities
            bool different_modalities = false;
            for (Modality m1 : concept1.modalities) {
                bool found = false;
                for (Modality m2 : concept2.modalities) {
                    if (m1 == m2) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    different_modalities = true;
                    break;
                }
            }
            
            if (!different_modalities) {
                continue;  // Same modalities, skip
            }
            
            // Compute correlation
            float correlation = 0.0f;
            size_t min_size = std::min(concept1.abstract_features.size(), 
                                       concept2.abstract_features.size());
            
            if (min_size > 0) {
                float mean1 = std::accumulate(concept1.abstract_features.begin(), 
                                             concept1.abstract_features.begin() + min_size, 0.0f) / min_size;
                float mean2 = std::accumulate(concept2.abstract_features.begin(), 
                                             concept2.abstract_features.begin() + min_size, 0.0f) / min_size;
                
                float cov = 0.0f;
                float var1 = 0.0f;
                float var2 = 0.0f;
                
                for (size_t k = 0; k < min_size; ++k) {
                    float diff1 = concept1.abstract_features[k] - mean1;
                    float diff2 = concept2.abstract_features[k] - mean2;
                    cov += diff1 * diff2;
                    var1 += diff1 * diff1;
                    var2 += diff2 * diff2;
                }
                
                if (var1 > 1e-6f && var2 > 1e-6f) {
                    correlation = std::abs(cov / std::sqrt(var1 * var2));
                }
            }
            
            if (correlation > impl_->config["pattern_threshold"]) {
                total_correlation += correlation;
                cross_modal_pairs++;
            }
        }
    }
    
    float pattern_strength = cross_modal_pairs > 0 ? 
        total_correlation / cross_modal_pairs : 0.0f;
    
    if (pattern_strength > impl_->config["pattern_threshold"]) {
        impl_->patterns_detected++;
    }
    
    return pattern_strength;
}

std::vector<UnifiedConcept> CrossModalAttentionSystem::apply_selective_attention(
    const std::vector<UnifiedConcept>& concepts,
    const std::vector<float>& query
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<UnifiedConcept> attended;
    std::vector<std::pair<float, size_t>> scores;
    
    // Compute attention scores
    for (size_t i = 0; i < concepts.size(); ++i) {
        float score = compute_attention_score(concepts[i], query);
        scores.push_back({score, i});
    }
    
    // Sort by score (descending)
    std::sort(scores.begin(), scores.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top concepts above threshold
    float threshold = impl_->config["integration_threshold"];
    for (const auto& [score, idx] : scores) {
        if (score >= threshold) {
            attended.push_back(concepts[idx]);
        }
    }
    
    return attended;
}

std::unordered_map<std::string, float> CrossModalAttentionSystem::get_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return {
        {"attention_computations", static_cast<float>(impl_->attention_computations)},
        {"integrations_performed", static_cast<float>(impl_->integrations_performed)},
        {"patterns_detected", static_cast<float>(impl_->patterns_detected)}
    };
}

void CrossModalAttentionSystem::update_config(const std::unordered_map<std::string, float>& new_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [key, value] : new_config) {
        impl_->config[key] = value;
    }
}

void CrossModalAttentionSystem::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->modality_focus.clear();
    impl_->attention_computations = 0;
    impl_->integrations_performed = 0;
    impl_->patterns_detected = 0;
}

// Private helper functions
float CrossModalAttentionSystem::compute_attention_score(
    const UnifiedConcept& concept,
    const std::vector<float>& query
) {
    // Compute dot product between concept embedding and query
    float score = 0.0f;
    size_t min_size = std::min(concept.embedding.size(), query.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        score += concept.embedding[i] * query[i];
    }
    
    // Normalize by concept confidence
    score *= concept.confidence;
    
    return score;
}

std::vector<float> CrossModalAttentionSystem::compute_modality_importance(
    const std::vector<UnifiedConcept>& concepts
) {
    std::unordered_map<Modality, float> importance;
    
    for (const auto& concept : concepts) {
        for (Modality m : concept.modalities) {
            importance[m] += concept.confidence;
        }
    }
    
    // Normalize
    float total = 0.0f;
    for (const auto& [m, imp] : importance) {
        total += imp;
    }
    
    std::vector<float> result;
    if (total > 0.0f) {
        for (auto& [m, imp] : importance) {
            result.push_back(imp / total);
        }
    }
    
    return result;
}

} // namespace rros

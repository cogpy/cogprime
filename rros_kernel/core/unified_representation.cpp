#include "unified_representation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace rros {

// Implementation struct
struct UnifiedRepresentation::Impl {
    std::unordered_map<std::string, float> config;
    std::unordered_map<std::string, UnifiedConcept> concept_registry;
    
    // Metrics
    size_t representations_created;
    size_t representations_merged;
    size_t abstractions_performed;
    size_t alignments_performed;
    
    Impl() : representations_created(0), representations_merged(0),
             abstractions_performed(0), alignments_performed(0) {
        // Default configuration
        config["embedding_dim"] = 128.0f;
        config["abstract_feature_dim"] = 64.0f;
        config["similarity_threshold"] = 0.7f;
        config["alignment_strength"] = 0.8f;
        config["abstraction_rate"] = 0.5f;
    }
};

// UnifiedConcept implementation
UnifiedConcept::UnifiedConcept() 
    : concept_id(""), abstraction_level(0.0f), confidence(1.0f) {}

UnifiedConcept::UnifiedConcept(const std::string& id, const std::vector<float>& emb)
    : concept_id(id), embedding(emb), abstraction_level(0.0f), confidence(1.0f) {}

// UnifiedRepresentation implementation
UnifiedRepresentation::UnifiedRepresentation(const std::unordered_map<std::string, float>& config)
    : impl_(std::make_unique<Impl>()) {
    if (!config.empty()) {
        for (const auto& [key, value] : config) {
            impl_->config[key] = value;
        }
    }
}

UnifiedRepresentation::~UnifiedRepresentation() = default;

UnifiedConcept UnifiedRepresentation::create_representation(
    const std::vector<float>& domain_data,
    Modality modality
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    UnifiedConcept concept;
    concept.concept_id = "concept_" + std::to_string(impl_->representations_created);
    
    // Create dense embedding
    size_t embedding_dim = static_cast<size_t>(impl_->config["embedding_dim"]);
    concept.embedding.resize(embedding_dim);
    
    // Transform domain data to embedding space
    for (size_t i = 0; i < embedding_dim && i < domain_data.size(); ++i) {
        concept.embedding[i] = domain_data[i];
    }
    // Pad with normalized values if needed
    for (size_t i = domain_data.size(); i < embedding_dim; ++i) {
        concept.embedding[i] = 0.1f * std::sin(static_cast<float>(i) * 0.1f);
    }
    
    // Extract abstract features
    concept.abstract_features = extract_abstract_features(domain_data);
    
    // Set modality
    concept.modalities.push_back(modality);
    
    // Compute abstraction level
    concept.abstraction_level = compute_abstraction_level(concept.abstract_features);
    
    // Set confidence based on data quality
    float variance = 0.0f;
    float mean = std::accumulate(domain_data.begin(), domain_data.end(), 0.0f) / domain_data.size();
    for (float val : domain_data) {
        variance += (val - mean) * (val - mean);
    }
    variance /= domain_data.size();
    concept.confidence = std::min(1.0f, std::max(0.3f, 1.0f - variance));
    
    impl_->representations_created++;
    impl_->concept_registry[concept.concept_id] = concept;
    
    return concept;
}

UnifiedConcept UnifiedRepresentation::merge_representations(
    const std::vector<UnifiedConcept>& concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concepts.empty()) {
        return UnifiedConcept();
    }
    
    UnifiedConcept merged;
    merged.concept_id = "merged_" + std::to_string(impl_->representations_merged);
    
    // Merge embeddings (weighted average)
    size_t max_dim = 0;
    for (const auto& c : concepts) {
        max_dim = std::max(max_dim, c.embedding.size());
    }
    
    merged.embedding.resize(max_dim, 0.0f);
    float total_confidence = 0.0f;
    
    for (const auto& c : concepts) {
        for (size_t i = 0; i < c.embedding.size(); ++i) {
            merged.embedding[i] += c.embedding[i] * c.confidence;
        }
        total_confidence += c.confidence;
    }
    
    // Normalize by total confidence
    if (total_confidence > 0.0f) {
        for (auto& val : merged.embedding) {
            val /= total_confidence;
        }
    }
    
    // Merge abstract features
    max_dim = 0;
    for (const auto& c : concepts) {
        max_dim = std::max(max_dim, c.abstract_features.size());
    }
    merged.abstract_features.resize(max_dim, 0.0f);
    
    for (const auto& c : concepts) {
        for (size_t i = 0; i < c.abstract_features.size(); ++i) {
            merged.abstract_features[i] += c.abstract_features[i] * c.confidence;
        }
    }
    
    if (total_confidence > 0.0f) {
        for (auto& val : merged.abstract_features) {
            val /= total_confidence;
        }
    }
    
    // Merge modalities (unique union)
    for (const auto& c : concepts) {
        for (Modality m : c.modalities) {
            if (std::find(merged.modalities.begin(), merged.modalities.end(), m) == merged.modalities.end()) {
                merged.modalities.push_back(m);
            }
        }
    }
    
    // Average abstraction level
    merged.abstraction_level = 0.0f;
    for (const auto& c : concepts) {
        merged.abstraction_level += c.abstraction_level * c.confidence;
    }
    merged.abstraction_level /= total_confidence;
    
    // Set merged confidence
    merged.confidence = total_confidence / concepts.size();
    
    impl_->representations_merged++;
    impl_->concept_registry[merged.concept_id] = merged;
    
    return merged;
}

UnifiedConcept UnifiedRepresentation::abstract_concept(
    const UnifiedConcept& concept,
    float target_level
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    UnifiedConcept abstracted = concept;
    abstracted.concept_id = concept.concept_id + "_abstract_" + std::to_string(impl_->abstractions_performed);
    
    // Increase abstraction level
    float rate = impl_->config["abstraction_rate"];
    abstracted.abstraction_level = concept.abstraction_level + 
                                   (target_level - concept.abstraction_level) * rate;
    
    // Abstract the features (reduce dimensionality, increase generality)
    size_t new_dim = static_cast<size_t>(concept.abstract_features.size() * (1.0f - abstracted.abstraction_level * 0.5f));
    abstracted.abstract_features.resize(new_dim);
    
    // Smooth and generalize embeddings
    for (size_t i = 0; i < abstracted.embedding.size(); ++i) {
        // Apply smoothing
        float smoothed = abstracted.embedding[i];
        if (i > 0) smoothed += abstracted.embedding[i-1] * 0.2f;
        if (i < abstracted.embedding.size() - 1) smoothed += abstracted.embedding[i+1] * 0.2f;
        abstracted.embedding[i] = smoothed / 1.4f;
    }
    
    // Confidence decreases with abstraction
    abstracted.confidence = concept.confidence * (1.0f - abstracted.abstraction_level * 0.3f);
    
    impl_->abstractions_performed++;
    
    return abstracted;
}

std::vector<float> UnifiedRepresentation::ground_to_domain(
    const UnifiedConcept& concept,
    Modality modality
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Transform abstract representation to domain-specific format
    std::vector<float> grounded;
    
    // Use modality-specific transformation
    grounded = modality_transform(concept.embedding, Modality::ABSTRACT, modality);
    
    // Apply abstraction de-grounding (inverse of abstraction)
    float deabstraction_factor = 1.0f - concept.abstraction_level;
    for (auto& val : grounded) {
        val *= deabstraction_factor;
    }
    
    return grounded;
}

float UnifiedRepresentation::compute_similarity(
    const UnifiedConcept& c1,
    const UnifiedConcept& c2
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Cosine similarity on embeddings
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    size_t min_size = std::min(c1.embedding.size(), c2.embedding.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += c1.embedding[i] * c2.embedding[i];
        norm1 += c1.embedding[i] * c1.embedding[i];
        norm2 += c2.embedding[i] * c2.embedding[i];
    }
    
    // Handle remaining dimensions
    for (size_t i = min_size; i < c1.embedding.size(); ++i) {
        norm1 += c1.embedding[i] * c1.embedding[i];
    }
    for (size_t i = min_size; i < c2.embedding.size(); ++i) {
        norm2 += c2.embedding[i] * c2.embedding[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 < 1e-6f || norm2 < 1e-6f) {
        return 0.0f;
    }
    
    float similarity = dot_product / (norm1 * norm2);
    
    // Bonus for matching modalities
    size_t common_modalities = 0;
    for (Modality m1 : c1.modalities) {
        if (std::find(c2.modalities.begin(), c2.modalities.end(), m1) != c2.modalities.end()) {
            common_modalities++;
        }
    }
    float modality_bonus = 0.1f * common_modalities / std::max(c1.modalities.size(), c2.modalities.size());
    
    return std::min(1.0f, similarity + modality_bonus);
}

std::vector<UnifiedConcept> UnifiedRepresentation::align_representations(
    const std::vector<UnifiedConcept>& concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (concepts.size() < 2) {
        return concepts;
    }
    
    // Compute centroid
    std::vector<float> centroid(concepts[0].embedding.size(), 0.0f);
    for (const auto& c : concepts) {
        for (size_t i = 0; i < c.embedding.size() && i < centroid.size(); ++i) {
            centroid[i] += c.embedding[i];
        }
    }
    for (auto& val : centroid) {
        val /= concepts.size();
    }
    
    // Align each concept towards centroid
    std::vector<UnifiedConcept> aligned;
    float strength = impl_->config["alignment_strength"];
    
    for (const auto& c : concepts) {
        UnifiedConcept aligned_concept = c;
        
        for (size_t i = 0; i < c.embedding.size() && i < centroid.size(); ++i) {
            aligned_concept.embedding[i] = c.embedding[i] + 
                                          (centroid[i] - c.embedding[i]) * strength;
        }
        
        aligned.push_back(aligned_concept);
    }
    
    impl_->alignments_performed++;
    
    return aligned;
}

std::unordered_map<std::string, float> UnifiedRepresentation::get_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return {
        {"representations_created", static_cast<float>(impl_->representations_created)},
        {"representations_merged", static_cast<float>(impl_->representations_merged)},
        {"abstractions_performed", static_cast<float>(impl_->abstractions_performed)},
        {"alignments_performed", static_cast<float>(impl_->alignments_performed)},
        {"concepts_registered", static_cast<float>(impl_->concept_registry.size())}
    };
}

void UnifiedRepresentation::update_config(const std::unordered_map<std::string, float>& new_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [key, value] : new_config) {
        impl_->config[key] = value;
    }
}

void UnifiedRepresentation::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->concept_registry.clear();
    impl_->representations_created = 0;
    impl_->representations_merged = 0;
    impl_->abstractions_performed = 0;
    impl_->alignments_performed = 0;
}

// Private helper functions
std::vector<float> UnifiedRepresentation::extract_abstract_features(const std::vector<float>& data) {
    size_t feature_dim = static_cast<size_t>(impl_->config["abstract_feature_dim"]);
    std::vector<float> features(feature_dim);
    
    // Extract statistical features
    if (!data.empty()) {
        float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
        float variance = 0.0f;
        for (float val : data) {
            variance += (val - mean) * (val - mean);
        }
        variance /= data.size();
        
        float min_val = *std::min_element(data.begin(), data.end());
        float max_val = *std::max_element(data.begin(), data.end());
        
        features[0] = mean;
        features[1] = std::sqrt(variance);
        features[2] = max_val - min_val;  // Range
        
        // Compute higher-order moments
        for (size_t i = 3; i < feature_dim; ++i) {
            features[i] = std::sin(static_cast<float>(i) * mean) * 
                         std::cos(static_cast<float>(i) * variance);
        }
    }
    
    return features;
}

float UnifiedRepresentation::compute_abstraction_level(const std::vector<float>& features) {
    if (features.empty()) return 0.0f;
    
    // Abstraction level based on feature variance and smoothness
    float variance = 0.0f;
    float mean = std::accumulate(features.begin(), features.end(), 0.0f) / features.size();
    for (float val : features) {
        variance += (val - mean) * (val - mean);
    }
    variance /= features.size();
    
    // High variance = low abstraction (concrete)
    // Low variance = high abstraction (abstract)
    return std::min(1.0f, 1.0f / (1.0f + variance));
}

std::vector<float> UnifiedRepresentation::modality_transform(
    const std::vector<float>& data,
    Modality from,
    Modality to
) {
    // Simple transformation matrix approach
    std::vector<float> transformed = data;
    
    // Apply modality-specific scaling and transformation
    if (from != to) {
        float scale = 1.0f;
        if (to == Modality::ABSTRACT) scale = 0.8f;
        else if (to == Modality::VISUAL) scale = 1.2f;
        else if (to == Modality::AUDITORY) scale = 1.1f;
        
        for (auto& val : transformed) {
            val *= scale;
        }
    }
    
    return transformed;
}

} // namespace rros

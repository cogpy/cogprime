#include "knowledge_distiller.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {

KnowledgeDistiller::KnowledgeDistiller(
    const std::unordered_map<std::string, float>& config)
    : config_(config) {
    
    if (config_.find("temperature") == config_.end()) {
        config_["temperature"] = 2.0f;
    }
    if (config_.find("importance_threshold") == config_.end()) {
        config_["importance_threshold"] = 0.1f;
    }
}

KnowledgeDistiller::~KnowledgeDistiller() = default;

std::vector<float> KnowledgeDistiller::distill(
    const std::vector<float>& source_model,
    size_t target_size,
    float preserve_performance) {
    
    std::lock_guard<std::mutex> lock(distiller_mutex_);
    
    if (source_model.empty() || target_size == 0 || target_size >= source_model.size()) {
        return source_model;
    }
    
    // Compute importance scores for all parameters
    std::vector<float> importance_scores = compute_importance_scores(source_model);
    
    // Select most important parameters
    std::vector<float> distilled = select_important_parameters(
        source_model,
        importance_scores,
        target_size
    );
    
    // Apply temperature scaling to preserve soft knowledge
    float temperature = config_["temperature"];
    for (float& val : distilled) {
        val = val / temperature;
    }
    
    return distilled;
}

void KnowledgeDistiller::update_config(
    const std::unordered_map<std::string, float>& config) {
    std::lock_guard<std::mutex> lock(distiller_mutex_);
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
}

void KnowledgeDistiller::reset() {
    std::lock_guard<std::mutex> lock(distiller_mutex_);
    // Nothing to reset currently
}

std::vector<float> KnowledgeDistiller::compute_importance_scores(
    const std::vector<float>& model) const {
    
    std::vector<float> scores;
    scores.reserve(model.size());
    
    // Compute importance based on magnitude and variance contribution
    float mean = std::accumulate(model.begin(), model.end(), 0.0f) / 
                 static_cast<float>(model.size());
    
    for (float value : model) {
        // Importance = magnitude * deviation from mean
        float importance = std::abs(value) * std::abs(value - mean);
        scores.push_back(importance);
    }
    
    // Normalize scores
    float max_score = *std::max_element(scores.begin(), scores.end());
    if (max_score > 1e-6f) {
        for (float& score : scores) {
            score /= max_score;
        }
    }
    
    return scores;
}

std::vector<float> KnowledgeDistiller::select_important_parameters(
    const std::vector<float>& model,
    const std::vector<float>& importance_scores,
    size_t target_size) const {
    
    // Create indices sorted by importance
    std::vector<size_t> indices(model.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
        [&importance_scores](size_t a, size_t b) {
            return importance_scores[a] > importance_scores[b];
        });
    
    // Select top parameters
    std::vector<float> distilled;
    distilled.reserve(target_size);
    
    for (size_t i = 0; i < std::min(target_size, indices.size()); ++i) {
        distilled.push_back(model[indices[i]]);
    }
    
    return distilled;
}

} // namespace rros

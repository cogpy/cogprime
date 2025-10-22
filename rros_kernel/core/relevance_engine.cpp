#include "relevance_engine.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

RelevanceEngine::RelevanceEngine(const std::unordered_map<std::string, float>& config) :
    config_(config),
    relevance_threshold_(0.3f)
{
    if (config_.find("relevance_threshold") != config_.end()) {
        relevance_threshold_ = config_["relevance_threshold"];
    }
    
    // Initialize relevance weights for different aspects
    relevance_weights_ = {1.0f, 0.8f, 0.9f, 0.7f, 0.85f}; // Default weights
}

RelevanceResult RelevanceEngine::process(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    RelevanceResult result;
    
    // Compute base salience map
    result.salience_map = compute_salience(input);
    
    // Apply contextual relevance modulation
    for (size_t i = 0; i < result.salience_map.size(); ++i) {
        result.salience_map[i] = evaluate_contextual_relevance(result.salience_map[i], context);
    }
    
    // Normalize salience values
    normalize_salience(result.salience_map);
    
    // Compute summary statistics
    result.max_salience = *std::max_element(result.salience_map.begin(), result.salience_map.end());
    result.average_salience = std::accumulate(result.salience_map.begin(), 
                                            result.salience_map.end(), 0.0f) / result.salience_map.size();
    
    // Find relevant indices (above threshold)
    for (size_t i = 0; i < result.salience_map.size(); ++i) {
        if (result.salience_map[i] > relevance_threshold_) {
            result.relevant_indices.push_back(i);
        }
    }
    
    // Compute confidence based on salience distribution
    float variance = 0.0f;
    for (float sal : result.salience_map) {
        float diff = sal - result.average_salience;
        variance += diff * diff;
    }
    variance /= result.salience_map.size();
    
    // Higher variance indicates clearer relevance distinction
    result.confidence = std::min(1.0f, variance * 2.0f + result.max_salience);
    
    return result;
}

std::vector<float> RelevanceEngine::compute_salience(const std::vector<float>& input) {
    std::vector<float> salience(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        // Multi-factor salience computation
        float magnitude = std::abs(input[i]);
        
        // Local contrast (difference from neighbors)
        float contrast = 0.0f;
        int neighbor_count = 0;
        for (int offset = -2; offset <= 2; ++offset) {
            int idx = static_cast<int>(i) + offset;
            if (idx >= 0 && idx < static_cast<int>(input.size()) && offset != 0) {
                contrast += std::abs(input[i] - input[idx]);
                ++neighbor_count;
            }
        }
        if (neighbor_count > 0) contrast /= neighbor_count;
        
        // Global rarity (distance from mean)
        float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
        float rarity = std::abs(input[i] - mean);
        
        // Combine factors
        salience[i] = magnitude * 0.4f + contrast * 0.4f + rarity * 0.2f;
    }
    
    return salience;
}

float RelevanceEngine::evaluate_contextual_relevance(
    float base_relevance,
    const std::unordered_map<std::string, float>& context
) {
    float contextual_boost = 1.0f;
    
    // Apply context-specific modulation
    if (context.find("attention_focus") != context.end()) {
        contextual_boost *= (1.0f + context.at("attention_focus") * 0.5f);
    }
    
    if (context.find("goal_relevance") != context.end()) {
        contextual_boost *= (1.0f + context.at("goal_relevance") * 0.7f);
    }
    
    if (context.find("memory_resonance") != context.end()) {
        contextual_boost *= (1.0f + context.at("memory_resonance") * 0.3f);
    }
    
    return base_relevance * contextual_boost;
}

void RelevanceEngine::normalize_salience(std::vector<float>& salience) {
    if (salience.empty()) return;
    
    float max_val = *std::max_element(salience.begin(), salience.end());
    if (max_val > 0.0f) {
        for (float& val : salience) {
            val /= max_val;
        }
    }
}

void RelevanceEngine::update_config(const std::unordered_map<std::string, float>& config) {
    for (const auto& [key, value] : config) {
        config_[key] = value;
        
        if (key == "relevance_threshold") {
            relevance_threshold_ = value;
        }
    }
}

void RelevanceEngine::reset() {
    // Reset to default state
    relevance_weights_ = {1.0f, 0.8f, 0.9f, 0.7f, 0.85f};
}

std::unordered_map<std::string, float> RelevanceEngine::get_thresholds() const {
    return {
        {"relevance_threshold", relevance_threshold_},
        {"salience_threshold", 0.1f},
        {"context_weight", 0.5f}
    };
}

} // namespace rros
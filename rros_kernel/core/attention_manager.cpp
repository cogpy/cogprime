#include "attention_manager.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

AttentionManager::AttentionManager(const std::unordered_map<std::string, float>& config) :
    config_(config),
    attention_budget_(1.0f),
    decay_rate_(0.95f)
{
    if (config_.find("attention_budget") != config_.end()) {
        attention_budget_ = config_["attention_budget"];
    }
    if (config_.find("attention_decay") != config_.end()) {
        decay_rate_ = config_["attention_decay"];
    }
}

std::vector<float> AttentionManager::update(const std::vector<float>& salience_map) {
    // Apply decay to current allocation
    apply_decay();
    
    // Compute new allocation based on salience
    std::vector<float> new_allocation = allocate(salience_map);
    
    // Update current allocation
    current_allocation_ = new_allocation;
    
    return new_allocation;
}

std::vector<float> AttentionManager::allocate(const std::vector<float>& relevance_scores) {
    if (relevance_scores.empty()) {
        return {};
    }
    
    std::vector<float> allocation(relevance_scores.size());
    
    // Compute total relevance for normalization
    float total_relevance = std::accumulate(relevance_scores.begin(), relevance_scores.end(), 0.0f);
    
    if (total_relevance == 0.0f) {
        // Uniform allocation if no relevance detected
        float uniform_weight = attention_budget_ / relevance_scores.size();
        std::fill(allocation.begin(), allocation.end(), uniform_weight);
        return allocation;
    }
    
    // Softmax-like allocation with temperature control
    float temperature = config_.find("attention_temperature") != config_.end() ? 
                       config_["attention_temperature"] : 2.0f;
    
    std::vector<float> exp_scores(relevance_scores.size());
    float exp_sum = 0.0f;
    
    for (size_t i = 0; i < relevance_scores.size(); ++i) {
        exp_scores[i] = std::exp(relevance_scores[i] * temperature);
        exp_sum += exp_scores[i];
    }
    
    // Normalize to budget
    for (size_t i = 0; i < allocation.size(); ++i) {
        allocation[i] = (exp_scores[i] / exp_sum) * attention_budget_;
    }
    
    // Apply competition effects
    float competition = compute_competition(relevance_scores);
    for (float& weight : allocation) {
        weight *= (1.0f - competition * 0.3f); // Reduce allocation under high competition
    }
    
    normalize_allocation(allocation);
    
    return allocation;
}

void AttentionManager::apply_decay() {
    for (float& weight : current_allocation_) {
        weight *= decay_rate_;
    }
}

void AttentionManager::normalize_allocation(std::vector<float>& allocation) {
    float total = std::accumulate(allocation.begin(), allocation.end(), 0.0f);
    
    if (total > attention_budget_ && total > 0.0f) {
        float scale = attention_budget_ / total;
        for (float& weight : allocation) {
            weight *= scale;
        }
    }
}

float AttentionManager::compute_competition(const std::vector<float>& scores) {
    if (scores.size() <= 1) return 0.0f;
    
    // Compute variance as measure of competition
    float mean = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();
    float variance = 0.0f;
    
    for (float score : scores) {
        float diff = score - mean;
        variance += diff * diff;
    }
    variance /= scores.size();
    
    // Low variance = high competition (similar scores)
    // High variance = low competition (clear winner)
    return 1.0f / (1.0f + variance * 4.0f);
}

void AttentionManager::update_config(const std::unordered_map<std::string, float>& config) {
    for (const auto& [key, value] : config) {
        config_[key] = value;
        
        if (key == "attention_budget") {
            attention_budget_ = value;
        } else if (key == "attention_decay") {
            decay_rate_ = value;
        }
    }
}

void AttentionManager::reset() {
    current_allocation_.clear();
    attention_budget_ = 1.0f;
    decay_rate_ = 0.95f;
}

float AttentionManager::get_utilization() const {
    return std::accumulate(current_allocation_.begin(), current_allocation_.end(), 0.0f) / attention_budget_;
}

} // namespace rros
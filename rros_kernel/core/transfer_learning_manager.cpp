#include "transfer_learning_manager.hpp"
#include "meta_learning_engine.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {

TransferableKnowledge::TransferableKnowledge() : confidence(0.0f) {}

TransferLearningManager::TransferLearningManager(
    const std::unordered_map<std::string, float>& config)
    : config_(config) {
    
    // Set defaults
    if (config_.find("alignment_threshold") == config_.end()) {
        config_["alignment_threshold"] = 0.6f;
    }
    if (config_.find("adaptation_strength") == config_.end()) {
        config_["adaptation_strength"] = 0.5f;
    }
}

TransferLearningManager::~TransferLearningManager() = default;

float TransferLearningManager::transfer(
    const Domain& source_domain,
    const Domain& target_domain,
    const LearningTask& task) {
    
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    // Extract knowledge from source domain
    std::vector<LearningTask> dummy_tasks;
    TransferableKnowledge source_knowledge = extract_knowledge(source_domain, dummy_tasks);
    
    // Adapt knowledge to target domain
    TransferableKnowledge adapted_knowledge = adapt_knowledge(source_knowledge, target_domain);
    
    // Compute alignment between domains
    float alignment = align_feature_spaces(
        source_domain.feature_space,
        target_domain.feature_space
    );
    
    // Transfer effectiveness is combination of alignment and adapted knowledge confidence
    float effectiveness = 0.6f * alignment + 0.4f * adapted_knowledge.confidence;
    
    // Store adapted knowledge in knowledge base
    knowledge_base_[target_domain.name] = adapted_knowledge;
    
    return effectiveness;
}

TransferableKnowledge TransferLearningManager::extract_knowledge(
    const Domain& domain,
    const std::vector<LearningTask>& tasks) {
    
    TransferableKnowledge knowledge;
    
    // Initialize feature weights from domain features
    knowledge.feature_weights = domain.feature_space;
    
    // Extract patterns from tasks if available
    if (!tasks.empty()) {
        // Compute average task characteristics
        std::unordered_map<std::string, float> avg_characteristics;
        for (const auto& task : tasks) {
            for (const auto& [key, value] : task.task_characteristics) {
                avg_characteristics[key] += value;
            }
        }
        for (auto& [key, value] : avg_characteristics) {
            value /= static_cast<float>(tasks.size());
        }
        knowledge.learned_patterns = avg_characteristics;
    }
    
    // Set confidence based on domain characteristics
    auto confidence_it = domain.characteristics.find("reliability");
    knowledge.confidence = (confidence_it != domain.characteristics.end()) ?
        confidence_it->second : 0.7f;
    
    return knowledge;
}

TransferableKnowledge TransferLearningManager::adapt_knowledge(
    const TransferableKnowledge& knowledge,
    const Domain& target_domain) {
    
    TransferableKnowledge adapted;
    
    // Find domain-invariant features
    adapted.domain_invariant_features = find_domain_invariant_features(
        Domain("source", knowledge.feature_weights),
        target_domain
    );
    
    // Adapt feature weights to target domain
    float adaptation_strength = config_["adaptation_strength"];
    size_t min_size = std::min(knowledge.feature_weights.size(), 
                               target_domain.feature_space.size());
    
    adapted.feature_weights.resize(min_size);
    for (size_t i = 0; i < min_size; ++i) {
        // Blend source weights with target domain features
        adapted.feature_weights[i] = 
            (1.0f - adaptation_strength) * knowledge.feature_weights[i] +
            adaptation_strength * target_domain.feature_space[i];
    }
    
    // Copy learned patterns
    adapted.learned_patterns = knowledge.learned_patterns;
    
    // Adjust confidence based on adaptation
    adapted.confidence = knowledge.confidence * 
        (0.8f + 0.2f * (1.0f - adaptation_strength));
    
    return adapted;
}

void TransferLearningManager::update_config(
    const std::unordered_map<std::string, float>& config) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
}

void TransferLearningManager::reset() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    knowledge_base_.clear();
}

// Private methods

float TransferLearningManager::align_feature_spaces(
    const std::vector<float>& source_features,
    const std::vector<float>& target_features) const {
    
    if (source_features.empty() || target_features.empty()) {
        return 0.0f;
    }
    
    // Compute cosine similarity for alignment
    size_t min_size = std::min(source_features.size(), target_features.size());
    
    float dot_product = 0.0f;
    float source_norm = 0.0f;
    float target_norm = 0.0f;
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += source_features[i] * target_features[i];
        source_norm += source_features[i] * source_features[i];
        target_norm += target_features[i] * target_features[i];
    }
    
    float denominator = std::sqrt(source_norm) * std::sqrt(target_norm);
    if (denominator < 1e-6f) {
        return 0.0f;
    }
    
    return dot_product / denominator;
}

std::vector<float> TransferLearningManager::find_domain_invariant_features(
    const Domain& source,
    const Domain& target) const {
    
    std::vector<float> invariant_features;
    
    size_t min_size = std::min(source.feature_space.size(), target.feature_space.size());
    
    // Features are invariant if they're similar across domains
    for (size_t i = 0; i < min_size; ++i) {
        float diff = std::abs(source.feature_space[i] - target.feature_space[i]);
        if (diff < 0.2f) { // Threshold for invariance
            invariant_features.push_back(0.5f * (source.feature_space[i] + target.feature_space[i]));
        }
    }
    
    return invariant_features;
}

} // namespace rros

#include "few_shot_learner.hpp"
#include "meta_learning_engine.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {

FewShotLearner::FewShotLearner(const std::unordered_map<std::string, float>& config)
    : config_(config) {
    
    if (config_.find("similarity_threshold") == config_.end()) {
        config_["similarity_threshold"] = 0.7f;
    }
    if (config_.find("prototype_weight") == config_.end()) {
        config_["prototype_weight"] = 0.8f;
    }
}

FewShotLearner::~FewShotLearner() = default;

float FewShotLearner::learn(const LearningTask& task, uint32_t num_shots) {
    std::lock_guard<std::mutex> lock(learner_mutex_);
    
    if (task.training_examples.empty() || task.labels.empty()) {
        return 0.0f;
    }
    
    // Compute prototypes from training examples
    prototypes_ = compute_prototypes(task.training_examples, task.labels, num_shots);
    
    if (prototypes_.empty()) {
        return 0.0f;
    }
    
    // Evaluate accuracy using prototype matching
    int correct = 0;
    int total = static_cast<int>(std::min(task.training_examples.size(), task.labels.size()));
    
    for (size_t i = 0; i < static_cast<size_t>(total); ++i) {
        // Find closest prototype
        float max_similarity = -1.0f;
        int predicted_class = 0;
        
        for (size_t j = 0; j < prototypes_.size(); ++j) {
            float similarity = compute_similarity(task.training_examples[i], prototypes_[j]);
            if (similarity > max_similarity) {
                max_similarity = similarity;
                predicted_class = static_cast<int>(j);
            }
        }
        
        // Check if prediction matches label
        if (static_cast<int>(task.labels[i]) == predicted_class && 
            max_similarity >= config_["similarity_threshold"]) {
            correct++;
        }
    }
    
    return total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 0.0f;
}

void FewShotLearner::update_config(const std::unordered_map<std::string, float>& config) {
    std::lock_guard<std::mutex> lock(learner_mutex_);
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
}

void FewShotLearner::reset() {
    std::lock_guard<std::mutex> lock(learner_mutex_);
    prototypes_.clear();
}

std::vector<std::vector<float>> FewShotLearner::compute_prototypes(
    const std::vector<std::vector<float>>& examples,
    const std::vector<float>& labels,
    uint32_t num_shots) const {
    
    if (examples.empty() || labels.empty()) {
        return {};
    }
    
    // Find unique classes
    std::vector<int> unique_classes;
    for (float label : labels) {
        int class_id = static_cast<int>(label);
        if (std::find(unique_classes.begin(), unique_classes.end(), class_id) == unique_classes.end()) {
            unique_classes.push_back(class_id);
        }
    }
    
    // Compute prototype for each class
    std::vector<std::vector<float>> prototypes;
    for (int class_id : unique_classes) {
        std::vector<std::vector<float>> class_examples;
        
        // Collect examples for this class (up to num_shots)
        for (size_t i = 0; i < examples.size() && class_examples.size() < num_shots; ++i) {
            if (static_cast<int>(labels[i]) == class_id) {
                class_examples.push_back(examples[i]);
            }
        }
        
        if (class_examples.empty()) {
            continue;
        }
        
        // Compute mean as prototype
        size_t feature_dim = class_examples[0].size();
        std::vector<float> prototype(feature_dim, 0.0f);
        
        for (const auto& example : class_examples) {
            for (size_t j = 0; j < std::min(feature_dim, example.size()); ++j) {
                prototype[j] += example[j];
            }
        }
        
        for (float& val : prototype) {
            val /= static_cast<float>(class_examples.size());
        }
        
        prototypes.push_back(prototype);
    }
    
    return prototypes;
}

float FewShotLearner::compute_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b) const {
    
    if (a.empty() || b.empty()) {
        return 0.0f;
    }
    
    // Compute cosine similarity
    size_t min_size = std::min(a.size(), b.size());
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
    return (denominator > 1e-6f) ? (dot_product / denominator) : 0.0f;
}

} // namespace rros

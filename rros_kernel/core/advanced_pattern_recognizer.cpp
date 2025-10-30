#include "advanced_pattern_recognizer.hpp"
#include <random>
#include <numeric>
#include <cstring>

namespace rros {

AdvancedPatternRecognizer::AdvancedPatternRecognizer(size_t feature_dim)
    : feature_dim_(feature_dim),
      local_dim_(128),
      global_dim_(128),
      total_patterns_recognized_(0),
      pattern_type_counts_({0, 0, 0, 0}),
      cumulative_confidence_(0.0f) {
    pattern_history_.reserve(MAX_HISTORY_SIZE);
}

PatternRecognitionResult AdvancedPatternRecognizer::recognize(const std::vector<float>& input) {
    PatternRecognitionResult result;
    
    // Extract multi-scale features
    auto local_features = extract_local_patterns(input);
    auto global_features = extract_global_patterns(input);
    
    // Combine features
    std::vector<float> combined_features;
    combined_features.reserve(local_features.size() + global_features.size());
    combined_features.insert(combined_features.end(), local_features.begin(), local_features.end());
    combined_features.insert(combined_features.end(), global_features.begin(), global_features.end());
    
    // Classify pattern type
    auto type_confidences = classify_pattern_type(combined_features);
    
    // Determine dominant pattern type
    auto max_it = std::max_element(type_confidences.begin(), type_confidences.end());
    size_t dominant_idx = std::distance(type_confidences.begin(), max_it);
    
    result.dominant_type = static_cast<PatternType>(dominant_idx);
    result.type_confidences = type_confidences;
    result.local_features = local_features;
    result.global_features = global_features;
    result.pattern_strength = *max_it;
    result.novelty_score = calculate_novelty(combined_features);
    
    // Update statistics
    total_patterns_recognized_++;
    pattern_type_counts_[dominant_idx]++;
    cumulative_confidence_ += result.pattern_strength;
    
    // Store in history for novelty detection
    if (pattern_history_.size() >= MAX_HISTORY_SIZE) {
        pattern_history_.erase(pattern_history_.begin());
    }
    pattern_history_.push_back(combined_features);
    
    return result;
}

std::vector<float> AdvancedPatternRecognizer::extract_local_patterns(const std::vector<float>& input) {
    // Local pattern extraction: capture fine-grained features
    // Implements a lightweight neural network transformation
    
    // First layer: feature_dim -> 256
    auto layer1 = dense_transform(input, 256, "relu");
    
    // Second layer: 256 -> local_dim (128)
    auto layer2 = dense_transform(layer1, local_dim_, "relu");
    
    return layer2;
}

std::vector<float> AdvancedPatternRecognizer::extract_global_patterns(const std::vector<float>& input) {
    // Global pattern extraction: capture broad structural features
    
    // First layer: feature_dim -> 256
    auto layer1 = dense_transform(input, 256, "relu");
    
    // Second layer: 256 -> global_dim (128)
    auto layer2 = dense_transform(layer1, global_dim_, "relu");
    
    return layer2;
}

std::array<float, 4> AdvancedPatternRecognizer::classify_pattern_type(
    const std::vector<float>& combined_features) {
    
    // Pattern type classification network
    // 256 (combined) -> 64 -> 4 (pattern types)
    
    // Layer 1: 256 -> 64
    auto layer1 = dense_transform(combined_features, 64, "relu");
    
    // Layer 2: 64 -> 32
    auto layer2 = dense_transform(layer1, 32, "relu");
    
    // Output layer: 32 -> 4
    auto logits = dense_transform(layer2, 4, "none");
    
    // Apply softmax to get probabilities
    auto probs = softmax(logits);
    
    std::array<float, 4> result;
    std::copy(probs.begin(), probs.begin() + 4, result.begin());
    
    return result;
}

float AdvancedPatternRecognizer::calculate_novelty(const std::vector<float>& pattern_features) {
    if (pattern_history_.empty()) {
        return 1.0f;  // First pattern is completely novel
    }
    
    // Calculate average cosine similarity with historical patterns
    float total_similarity = 0.0f;
    size_t compare_count = std::min(pattern_history_.size(), size_t(20));  // Compare with recent 20
    
    // Compute norm of current pattern
    float current_norm = 0.0f;
    for (float val : pattern_features) {
        current_norm += val * val;
    }
    current_norm = std::sqrt(current_norm);
    
    for (size_t i = pattern_history_.size() - compare_count; i < pattern_history_.size(); ++i) {
        const auto& hist_pattern = pattern_history_[i];
        
        // Compute cosine similarity
        float dot_product = 0.0f;
        float hist_norm = 0.0f;
        
        size_t min_size = std::min(pattern_features.size(), hist_pattern.size());
        for (size_t j = 0; j < min_size; ++j) {
            dot_product += pattern_features[j] * hist_pattern[j];
            hist_norm += hist_pattern[j] * hist_pattern[j];
        }
        hist_norm = std::sqrt(hist_norm);
        
        float similarity = dot_product / (current_norm * hist_norm + 1e-8f);
        total_similarity += similarity;
    }
    
    float avg_similarity = total_similarity / compare_count;
    
    // Novelty is inverse of similarity
    return std::max(0.0f, std::min(1.0f, 1.0f - avg_similarity));
}

std::unordered_map<std::string, float> AdvancedPatternRecognizer::get_statistics() const {
    std::unordered_map<std::string, float> stats;
    
    stats["total_patterns"] = static_cast<float>(total_patterns_recognized_);
    stats["avg_confidence"] = total_patterns_recognized_ > 0 ? 
        cumulative_confidence_ / total_patterns_recognized_ : 0.0f;
    
    stats["causal_count"] = static_cast<float>(pattern_type_counts_[0]);
    stats["analogical_count"] = static_cast<float>(pattern_type_counts_[1]);
    stats["hierarchical_count"] = static_cast<float>(pattern_type_counts_[2]);
    stats["temporal_count"] = static_cast<float>(pattern_type_counts_[3]);
    
    return stats;
}

void AdvancedPatternRecognizer::reset() {
    pattern_history_.clear();
    total_patterns_recognized_ = 0;
    pattern_type_counts_ = {0, 0, 0, 0};
    cumulative_confidence_ = 0.0f;
}

// Helper functions for neural network-like transformations

std::vector<float> AdvancedPatternRecognizer::dense_transform(
    const std::vector<float>& input,
    size_t output_dim,
    const std::string& activation) {
    
    // Simple dense layer transformation with random projection
    // In production, these would be learned weights
    
    std::vector<float> output(output_dim, 0.0f);
    
    // Use deterministic random for reproducibility based on dimensions
    std::mt19937 rng(input.size() * output_dim);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(input.size())));
    
    // Matrix multiplication with random weights
    for (size_t i = 0; i < output_dim; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < input.size(); ++j) {
            float weight = dist(rng);
            sum += input[j] * weight;
        }
        // Add bias
        sum += dist(rng) * 0.1f;
        output[i] = sum;
    }
    
    // Apply activation
    return apply_activation(output, activation);
}

std::vector<float> AdvancedPatternRecognizer::apply_activation(
    const std::vector<float>& input,
    const std::string& activation) {
    
    std::vector<float> output = input;
    
    if (activation == "relu") {
        for (float& val : output) {
            val = relu(val);
        }
    } else if (activation == "sigmoid") {
        for (float& val : output) {
            val = sigmoid(val);
        }
    } else if (activation == "tanh") {
        for (float& val : output) {
            val = tanh_activation(val);
        }
    }
    // "none" activation returns input as-is
    
    return output;
}

std::vector<float> AdvancedPatternRecognizer::softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    // Find max for numerical stability
    float max_val = *std::max_element(input.begin(), input.end());
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (float& val : output) {
        val /= sum;
    }
    
    return output;
}

} // namespace rros

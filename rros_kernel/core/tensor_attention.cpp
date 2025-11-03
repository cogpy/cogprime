/**
 * @file tensor_attention.cpp
 * @brief Implementation of optimized tensor attention operations
 */

#include "tensor_attention.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace rros {
namespace ggml {

//=============================================================================
// TensorAttention Implementation
//=============================================================================

TensorAttention::TensorAttention(const AttentionConfig& config)
    : config_(config) {
    if (config_.use_cache) {
        cache_ = std::make_unique<TensorCache>(config_.cache_size);
    }
    if (config_.parallel_mode) {
        executor_ = std::make_unique<ParallelExecutor>();
    }
}

TensorAttention::~TensorAttention() = default;

std::vector<float> TensorAttention::compute_attention(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& targets) {
    
    PROFILE_KERNEL_OP("TensorAttention::compute_attention");
    
    if (targets.empty()) {
        return {};
    }
    
    std::vector<float> similarities(targets.size());
    
    // Compute similarities (potentially cached and parallel)
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i];
        
        // Try cache first
        if (config_.use_cache) {
            std::string cache_key = make_cache_key(query, target);
            std::vector<float> cached_sim;
            if (cache_->retrieve(cache_key, cached_sim) && !cached_sim.empty()) {
                similarities[i] = cached_sim[0];
                continue;
            }
        }
        
        // Compute similarity
        float sim = compute_similarity(query, target);
        similarities[i] = sim;
        
        // Cache result
        if (config_.use_cache) {
            std::string cache_key = make_cache_key(query, target);
            cache_->store(cache_key, {sim});
        }
    }
    
    // Apply softmax with temperature
    auto weights = softmax(similarities, config_.temperature);
    
    // Apply sparsity threshold
    if (config_.sparsity_threshold > 0.0f) {
        float sum = 0.0f;
        for (auto& w : weights) {
            if (w < config_.sparsity_threshold) {
                w = 0.0f;
            }
            sum += w;
        }
        // Renormalize
        if (sum > 0.0f) {
            for (auto& w : weights) {
                w /= sum;
            }
        }
    }
    
    return weights;
}

std::vector<float> TensorAttention::apply_attention(
    const std::vector<float>& weights,
    const std::vector<std::vector<float>>& values) {
    
    PROFILE_KERNEL_OP("TensorAttention::apply_attention");
    
    if (weights.empty() || values.empty()) {
        return {};
    }
    
    size_t dim = values[0].size();
    std::vector<float> result(dim, 0.0f);
    
    for (size_t i = 0; i < weights.size() && i < values.size(); ++i) {
        for (size_t j = 0; j < dim && j < values[i].size(); ++j) {
            result[j] += weights[i] * values[i][j];
        }
    }
    
    return result;
}

std::vector<std::vector<float>> TensorAttention::self_attention(
    const std::vector<std::vector<float>>& inputs) {
    
    PROFILE_KERNEL_OP("TensorAttention::self_attention");
    
    std::vector<std::vector<float>> outputs;
    outputs.reserve(inputs.size());
    
    for (const auto& query : inputs) {
        auto weights = compute_attention(query, inputs);
        auto output = apply_attention(weights, inputs);
        outputs.push_back(output);
    }
    
    return outputs;
}

TensorCache::CacheStats TensorAttention::get_cache_stats() const {
    if (cache_) {
        return cache_->get_stats();
    }
    return TensorCache::CacheStats{};
}

void TensorAttention::clear_cache() {
    if (cache_) {
        cache_->clear();
    }
}

float TensorAttention::compute_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    
    return OptimizedCognitiveOps::cosine_similarity(a, b);
}

std::vector<float> TensorAttention::softmax(
    const std::vector<float>& logits,
    float temperature) {
    
    if (logits.empty()) {
        return {};
    }
    
    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp(x/T - max/T)
    std::vector<float> exp_values(logits.size());
    float sum = 0.0f;
    
    for (size_t i = 0; i < logits.size(); ++i) {
        float exp_val = std::exp((logits[i] - max_logit) / temperature);
        exp_values[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (auto& val : exp_values) {
            val /= sum;
        }
    }
    
    return exp_values;
}

std::string TensorAttention::make_cache_key(
    const std::vector<float>& a,
    const std::vector<float>& b) const {
    
    // Create simple hash-based key (first few elements)
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    
    size_t n = std::min(size_t(4), std::min(a.size(), b.size()));
    for (size_t i = 0; i < n; ++i) {
        oss << a[i] << "," << b[i] << ";";
    }
    
    return oss.str();
}

//=============================================================================
// TensorRelevance Implementation
//=============================================================================

TensorRelevance::TensorRelevance(const RelevanceConfig& config)
    : config_(config), current_threshold_(config.threshold) {
    
    if (config_.use_cache) {
        cache_ = std::make_unique<TensorCache>(config_.cache_size);
    }
    executor_ = std::make_unique<ParallelExecutor>();
}

TensorRelevance::~TensorRelevance() = default;

float TensorRelevance::compute_relevance(
    const std::vector<float>& input,
    const std::vector<float>& context) {
    
    PROFILE_KERNEL_OP("TensorRelevance::compute_relevance");
    
    // Try cache
    if (config_.use_cache) {
        std::ostringstream key;
        key << input.size() << ":" << context.size();
        for (size_t i = 0; i < std::min(size_t(3), input.size()); ++i) {
            key << ":" << input[i];
        }
        
        std::vector<float> cached;
        if (cache_->retrieve(key.str(), cached) && !cached.empty()) {
            return cached[0];
        }
    }
    
    // Compute relevance as cosine similarity
    float relevance = OptimizedCognitiveOps::cosine_similarity(input, context);
    
    // Apply threshold
    if (relevance < current_threshold_) {
        relevance = 0.0f;
    }
    
    return relevance;
}

std::vector<float> TensorRelevance::compute_relevance_map(
    const std::vector<std::vector<float>>& inputs,
    const std::vector<float>& context) {
    
    PROFILE_KERNEL_OP("TensorRelevance::compute_relevance_map");
    
    std::vector<float> relevance_scores(inputs.size());
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        relevance_scores[i] = compute_relevance(inputs[i], context);
    }
    
    return relevance_scores;
}

void TensorRelevance::update_threshold(const std::vector<float>& recent_scores) {
    if (!config_.adaptive || recent_scores.empty()) {
        return;
    }
    
    // Update threshold as running average
    float mean = std::accumulate(recent_scores.begin(), recent_scores.end(), 0.0f) 
                 / recent_scores.size();
    
    current_threshold_ = 0.9f * current_threshold_ + 0.1f * mean;
    
    // Clamp to reasonable range
    current_threshold_ = std::max(0.1f, std::min(0.8f, current_threshold_));
}

//=============================================================================
// OptimizedCognitiveOps Implementation
//=============================================================================

float OptimizedCognitiveOps::dot_product(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    
    if (a.empty() || b.empty()) {
        return 0.0f;
    }
    
    size_t n = std::min(a.size(), b.size());
    float sum = 0.0f;
    
    // Simple loop - compiler may auto-vectorize
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

float OptimizedCognitiveOps::norm(const std::vector<float>& vec) {
    if (vec.empty()) {
        return 0.0f;
    }
    
    float sum_sq = 0.0f;
    for (float val : vec) {
        sum_sq += val * val;
    }
    
    return std::sqrt(sum_sq);
}

float OptimizedCognitiveOps::cosine_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    
    if (a.empty() || b.empty()) {
        return 0.0f;
    }
    
    float dot = dot_product(a, b);
    float norm_a = norm(a);
    float norm_b = norm(b);
    
    if (norm_a < 1e-9f || norm_b < 1e-9f) {
        return 0.0f;
    }
    
    return dot / (norm_a * norm_b);
}

std::vector<float> OptimizedCognitiveOps::add(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    
    size_t n = std::min(a.size(), b.size());
    std::vector<float> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
    
    return result;
}

std::vector<float> OptimizedCognitiveOps::multiply(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    
    size_t n = std::min(a.size(), b.size());
    std::vector<float> result(n);
    
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
    
    return result;
}

std::vector<float> OptimizedCognitiveOps::scale(
    const std::vector<float>& vec,
    float scalar) {
    
    std::vector<float> result(vec.size());
    
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * scalar;
    }
    
    return result;
}

std::vector<float> OptimizedCognitiveOps::normalize(
    const std::vector<float>& vec) {
    
    float n = norm(vec);
    if (n < 1e-9f) {
        return vec;
    }
    
    return scale(vec, 1.0f / n);
}

std::vector<float> OptimizedCognitiveOps::relu(
    const std::vector<float>& vec) {
    
    std::vector<float> result(vec.size());
    
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = std::max(0.0f, vec[i]);
    }
    
    return result;
}

std::vector<float> OptimizedCognitiveOps::tanh(
    const std::vector<float>& vec) {
    
    std::vector<float> result(vec.size());
    
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = std::tanh(vec[i]);
    }
    
    return result;
}

} // namespace ggml
} // namespace rros

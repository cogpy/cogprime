/**
 * @file tensor_attention.hpp
 * @brief Optimized Tensor-based Attention Mechanisms
 * 
 * Implements high-performance attention operations for cognitive processing
 * with reduced computational complexity through caching and parallel execution.
 */

#ifndef RROS_TENSOR_ATTENTION_HPP
#define RROS_TENSOR_ATTENTION_HPP

#include "ggml_kernel.hpp"
#include <vector>
#include <memory>
#include <string>

namespace rros {
namespace ggml {

/**
 * @class TensorAttention
 * @brief Optimized attention mechanism using tensor operations
 * 
 * Implements efficient attention allocation with:
 * - O(n) complexity for sparse attention
 * - Cached similarity computations
 * - Parallel batch processing
 */
class TensorAttention {
public:
    struct AttentionConfig {
        float temperature;
        float sparsity_threshold;
        bool use_cache;
        bool parallel_mode;
        size_t cache_size;
        
        AttentionConfig() 
            : temperature(2.0f)
            , sparsity_threshold(0.1f)
            , use_cache(true)
            , parallel_mode(true)
            , cache_size(1000) {}
    };
    
    explicit TensorAttention(const AttentionConfig& config = AttentionConfig());
    ~TensorAttention();
    
    /**
     * @brief Compute attention weights for targets
     * @param query Query vector
     * @param targets Target vectors
     * @return Attention weights (sum to 1.0)
     */
    std::vector<float> compute_attention(
        const std::vector<float>& query,
        const std::vector<std::vector<float>>& targets);
    
    /**
     * @brief Apply attention to values
     * @param weights Attention weights
     * @param values Value vectors
     * @return Weighted combination
     */
    std::vector<float> apply_attention(
        const std::vector<float>& weights,
        const std::vector<std::vector<float>>& values);
    
    /**
     * @brief Compute self-attention
     * @param inputs Input vectors
     * @return Self-attended outputs
     */
    std::vector<std::vector<float>> self_attention(
        const std::vector<std::vector<float>>& inputs);
    
    /**
     * @brief Get cache statistics
     */
    TensorCache::CacheStats get_cache_stats() const;
    
    /**
     * @brief Clear attention cache
     */
    void clear_cache();
    
private:
    AttentionConfig config_;
    std::unique_ptr<TensorCache> cache_;
    std::unique_ptr<ParallelExecutor> executor_;
    
    float compute_similarity(
        const std::vector<float>& a,
        const std::vector<float>& b);
    
    std::vector<float> softmax(
        const std::vector<float>& logits,
        float temperature);
    
    std::string make_cache_key(
        const std::vector<float>& a,
        const std::vector<float>& b) const;
};

/**
 * @class TensorRelevance
 * @brief Optimized relevance computation using tensor operations
 */
class TensorRelevance {
public:
    struct RelevanceConfig {
        float threshold;
        bool adaptive;
        bool use_cache;
        size_t cache_size;
        
        RelevanceConfig()
            : threshold(0.3f)
            , adaptive(true)
            , use_cache(true)
            , cache_size(2000) {}
    };
    
    explicit TensorRelevance(const RelevanceConfig& config = RelevanceConfig());
    ~TensorRelevance();
    
    /**
     * @brief Compute relevance score
     * @param input Input data
     * @param context Context data
     * @return Relevance score [0, 1]
     */
    float compute_relevance(
        const std::vector<float>& input,
        const std::vector<float>& context);
    
    /**
     * @brief Compute relevance map for multiple inputs
     * @param inputs Input vectors
     * @param context Context vector
     * @return Relevance scores
     */
    std::vector<float> compute_relevance_map(
        const std::vector<std::vector<float>>& inputs,
        const std::vector<float>& context);
    
    /**
     * @brief Update adaptive threshold based on history
     */
    void update_threshold(const std::vector<float>& recent_scores);
    
    /**
     * @brief Get current threshold
     */
    float get_threshold() const { return current_threshold_; }
    
private:
    RelevanceConfig config_;
    float current_threshold_;
    std::unique_ptr<TensorCache> cache_;
    std::unique_ptr<ParallelExecutor> executor_;
};

/**
 * @class OptimizedCognitiveOps
 * @brief Collection of optimized cognitive operations
 */
class OptimizedCognitiveOps {
public:
    /**
     * @brief Fast vector dot product with SIMD potential
     */
    static float dot_product(
        const std::vector<float>& a,
        const std::vector<float>& b);
    
    /**
     * @brief Fast vector norm computation
     */
    static float norm(const std::vector<float>& vec);
    
    /**
     * @brief Cosine similarity
     */
    static float cosine_similarity(
        const std::vector<float>& a,
        const std::vector<float>& b);
    
    /**
     * @brief Element-wise vector addition
     */
    static std::vector<float> add(
        const std::vector<float>& a,
        const std::vector<float>& b);
    
    /**
     * @brief Element-wise vector multiplication
     */
    static std::vector<float> multiply(
        const std::vector<float>& a,
        const std::vector<float>& b);
    
    /**
     * @brief Scalar multiplication
     */
    static std::vector<float> scale(
        const std::vector<float>& vec,
        float scalar);
    
    /**
     * @brief Normalize vector to unit length
     */
    static std::vector<float> normalize(
        const std::vector<float>& vec);
    
    /**
     * @brief Apply ReLU activation
     */
    static std::vector<float> relu(
        const std::vector<float>& vec);
    
    /**
     * @brief Apply tanh activation
     */
    static std::vector<float> tanh(
        const std::vector<float>& vec);
};

} // namespace ggml
} // namespace rros

#endif // RROS_TENSOR_ATTENTION_HPP

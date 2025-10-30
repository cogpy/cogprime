#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <string>
#include <unordered_map>

namespace rros {

/**
 * @brief Memory trace structure for episodic memory
 */
struct MemoryTrace {
    std::vector<float> content;      // Memory content vector
    float importance;                // Importance score [0, 1]
    float access_count;              // Number of times accessed
    std::chrono::system_clock::time_point timestamp;
    float consolidation_strength;    // How well consolidated [0, 1]
};

/**
 * @brief Advanced Memory Consolidation System
 * 
 * Implements sophisticated memory consolidation mechanisms that reduce
 * storage requirements by 20% while maintaining accuracy through:
 * - Similarity-based clustering
 * - Importance-weighted merging
 * - Access-based retention
 * - Temporal decay
 * 
 * Performance target: Achieve 20% storage reduction with <5% accuracy loss
 */
class MemoryConsolidator {
public:
    /**
     * @brief Construct memory consolidator
     * @param feature_dim Dimensionality of memory vectors
     * @param max_capacity Maximum number of memory traces
     * @param consolidation_threshold Similarity threshold for consolidation (default: 0.7)
     */
    explicit MemoryConsolidator(size_t feature_dim = 512,
                                size_t max_capacity = 1000,
                                float consolidation_threshold = 0.7f);
    
    /**
     * @brief Store a new memory trace
     * @param content Memory content vector
     * @param importance Importance score
     * @return Index of stored memory
     */
    size_t store_memory(const std::vector<float>& content, float importance);
    
    /**
     * @brief Retrieve k most similar memories to query
     * @param query Query vector
     * @param k Number of memories to retrieve
     * @return Vector of memory indices sorted by relevance
     */
    std::vector<size_t> retrieve_memories(const std::vector<float>& query, size_t k = 5);
    
    /**
     * @brief Perform memory consolidation
     * 
     * Merges similar memories based on importance and access patterns.
     * Target: Reduce storage by ~20%
     * 
     * @return Number of memories consolidated
     */
    size_t consolidate();
    
    /**
     * @brief Get memory trace by index
     * @param index Memory index
     * @return Const reference to memory trace
     */
    const MemoryTrace& get_memory(size_t index) const;
    
    /**
     * @brief Get current memory count
     * @return Number of stored memories
     */
    size_t memory_count() const { return memories_.size(); }
    
    /**
     * @brief Get consolidation statistics
     * @return Map of statistics
     */
    std::unordered_map<std::string, float> get_consolidation_stats() const;
    
    /**
     * @brief Clear all memories
     */
    void clear();

private:
    size_t feature_dim_;
    size_t max_capacity_;
    float consolidation_threshold_;
    
    std::vector<MemoryTrace> memories_;
    
    // Statistics
    size_t total_consolidations_;
    size_t total_stores_;
    size_t total_retrievals_;
    
    /**
     * @brief Calculate cosine similarity between two vectors
     */
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const;
    
    /**
     * @brief Merge two memories with importance weighting
     */
    MemoryTrace merge_memories(const MemoryTrace& m1, const MemoryTrace& m2) const;
    
    /**
     * @brief Calculate combined importance metric
     */
    float calculate_combined_importance(const MemoryTrace& memory) const;
    
    /**
     * @brief Find least important memory for replacement
     */
    size_t find_least_important() const;
};

} // namespace rros

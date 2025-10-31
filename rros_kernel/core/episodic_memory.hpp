#pragma once

#include "temporal_knowledge.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

namespace rros {

/**
 * @brief Episodic memory entry with temporal context
 */
struct EpisodicMemory {
    size_t memory_id;                      // Unique memory identifier
    std::vector<float> content;            // Memory content vector
    uint64_t encoding_time;                // When memory was created
    uint64_t last_accessed;                // Last retrieval time
    size_t access_count;                   // Number of retrievals
    float consolidation_level;             // Consolidation strength [0.0, 1.0]
    std::vector<size_t> temporal_context;  // Associated temporal events
    std::unordered_map<std::string, float> emotional_tags; // Emotional associations
    float vividness;                       // Memory vividness [0.0, 1.0]
};

/**
 * @brief Temporal index for efficient memory retrieval
 */
struct TemporalIndex {
    uint64_t time_bin;                     // Time bin identifier
    std::vector<size_t> memory_ids;        // Memories in this time bin
    float average_importance;              // Average importance in bin
};

/**
 * @brief Episodic Memory System with Temporal Indexing
 * 
 * Advanced episodic memory system supporting:
 * - Temporal indexing for O(log n) retrieval
 * - Context-dependent consolidation
 * - Emotional tagging and association
 * - Temporal clustering and organization
 * - Forgetting curve modeling
 * 
 * Performance target: ≤10µs retrieval, ≤5µs encoding
 */
class EpisodicMemorySystem {
public:
    /**
     * @brief Construct episodic memory system
     * @param capacity Maximum number of memories
     * @param time_bin_size Time bin size in microseconds (default: 1 second)
     */
    explicit EpisodicMemorySystem(
        size_t capacity = 5000,
        uint64_t time_bin_size = 1000000
    );
    
    /**
     * @brief Encode new episodic memory
     * @param content Memory content vector
     * @param importance Memory importance
     * @param temporal_events Associated temporal events
     * @return Memory ID
     */
    size_t encode_memory(
        const std::vector<float>& content,
        float importance,
        const std::vector<size_t>& temporal_events = {}
    );
    
    /**
     * @brief Retrieve memories by temporal proximity
     * @param query_time Query timestamp
     * @param time_window Time window in microseconds
     * @param max_results Maximum number of results
     * @return Vector of memory IDs
     */
    std::vector<size_t> retrieve_by_time(
        uint64_t query_time,
        uint64_t time_window,
        size_t max_results = 10
    );
    
    /**
     * @brief Retrieve memories by content similarity
     * @param query Query vector
     * @param max_results Maximum number of results
     * @return Vector of memory IDs sorted by relevance
     */
    std::vector<size_t> retrieve_by_content(
        const std::vector<float>& query,
        size_t max_results = 10
    );
    
    /**
     * @brief Retrieve memories combining temporal and content cues
     * @param query Content query vector
     * @param query_time Temporal query
     * @param time_window Time window in microseconds
     * @param temporal_weight Weight for temporal vs content [0.0, 1.0]
     * @param max_results Maximum number of results
     * @return Vector of memory IDs
     */
    std::vector<size_t> retrieve_contextual(
        const std::vector<float>& query,
        uint64_t query_time,
        uint64_t time_window,
        float temporal_weight = 0.5f,
        size_t max_results = 10
    );
    
    /**
     * @brief Apply temporal consolidation
     * 
     * Consolidates memories based on:
     * - Temporal proximity
     * - Access patterns
     * - Importance scores
     * 
     * @return Number of memories consolidated
     */
    size_t consolidate_memories();
    
    /**
     * @brief Apply forgetting curve
     * 
     * Models natural forgetting by reducing vividness and
     * potentially removing least important memories.
     * 
     * @param decay_rate Forgetting rate per time unit
     */
    void apply_forgetting(float decay_rate = 0.001f);
    
    /**
     * @brief Add emotional tag to memory
     * @param memory_id Memory identifier
     * @param emotion Emotion name
     * @param intensity Emotion intensity [0.0, 1.0]
     */
    void tag_emotion(
        size_t memory_id,
        const std::string& emotion,
        float intensity
    );
    
    /**
     * @brief Get memory by ID
     * @param memory_id Memory identifier
     * @return Const reference to episodic memory
     */
    const EpisodicMemory& get_memory(size_t memory_id) const;
    
    /**
     * @brief Get episodic memory statistics
     * @return Map of statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Clear all episodic memories
     */
    void clear();

private:
    size_t capacity_;
    uint64_t time_bin_size_;
    
    std::vector<EpisodicMemory> memories_;
    std::unordered_map<uint64_t, TemporalIndex> temporal_indices_;
    
    size_t next_memory_id_;
    
    // Helper methods
    uint64_t get_current_time_us() const;
    uint64_t compute_time_bin(uint64_t timestamp) const;
    float compute_content_similarity(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) const;
    float compute_temporal_relevance(
        uint64_t memory_time,
        uint64_t query_time,
        uint64_t time_window
    ) const;
    void update_temporal_index(size_t memory_id, uint64_t timestamp);
    size_t find_least_important_memory() const;
};

} // namespace rros

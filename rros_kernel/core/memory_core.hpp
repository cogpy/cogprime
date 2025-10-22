#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace rros {

/**
 * @brief Memory trace representing stored experience
 */
struct MemoryTrace {
    std::vector<float> data;           // Stored data
    float relevance;                   // Relevance at storage time
    std::vector<float> attention;      // Attention weights at storage
    uint64_t timestamp;                // Storage timestamp
    float decay;                       // Current decay level
    size_t access_count;               // Number of retrievals
};

/**
 * @brief Core memory management system
 * 
 * Manages storage and retrieval of experiences with relevance-based
 * organization and attention-weighted encoding.
 */
class MemoryCore {
public:
    explicit MemoryCore(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Store new experience in memory
     * @param data Experience data to store
     * @param relevance Relevance of the experience
     * @param attention Attention weights during encoding
     * @return Memory trace ID
     */
    size_t store(
        const std::vector<float>& data,
        float relevance,
        const std::vector<float>& attention
    );
    
    /**
     * @brief Retrieve experiences similar to query
     * @param query Query data for retrieval
     * @param max_results Maximum number of results to return
     * @return Vector of matching memory traces
     */
    std::vector<std::shared_ptr<MemoryTrace>> retrieve(
        const std::vector<float>& query,
        size_t max_results = 10
    );
    
    /**
     * @brief Update memory configuration
     * @param config New configuration parameters
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset memory to initial state
     */
    void reset();
    
    /**
     * @brief Get memory utilization as fraction of capacity
     * @return Utilization level [0.0, 1.0]
     */
    float get_utilization() const;
    
    /**
     * @brief Perform memory consolidation and cleanup
     */
    void consolidate();

private:
    std::unordered_map<std::string, float> config_;
    std::vector<std::shared_ptr<MemoryTrace>> traces_;
    size_t capacity_;
    float decay_rate_;
    uint64_t next_id_;
    
    // Internal methods
    float compute_similarity(const std::vector<float>& a, const std::vector<float>& b);
    void apply_decay();
    void prune_weak_traces();
    size_t find_insertion_point(float relevance);
};

} // namespace rros
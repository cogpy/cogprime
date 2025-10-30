#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <mutex>
#include <deque>

namespace rros {

/**
 * @brief Meta-cognitive monitoring data point
 * 
 * Captures a snapshot of cognitive processing for introspection
 */
struct CognitiveSnapshot {
    std::chrono::high_resolution_clock::time_point timestamp;
    std::unordered_map<std::string, float> metrics;
    std::vector<float> state_vector;
    std::string processing_stage;
    float relevance_score;
    float confidence;
    uint64_t cycle_id;
    
    CognitiveSnapshot();
};

/**
 * @brief Meta-cognitive process metrics
 * 
 * Tracks metrics about cognitive processes over time
 */
struct ProcessMetrics {
    float average_processing_time_us;
    float average_relevance;
    float average_confidence;
    float success_rate;
    uint64_t total_cycles;
    uint64_t successful_cycles;
    std::chrono::high_resolution_clock::time_point last_update;
    
    ProcessMetrics();
    void update(float processing_time, float relevance, float confidence, bool success);
};

/**
 * @brief Meta-cognitive monitoring and introspection system
 * 
 * Monitors cognitive processes, tracks performance, and provides
 * introspection capabilities for self-reflection and optimization.
 * Implements meta-cognitive awareness as defined in Episode 10
 * (Higher-Order Thought) and Episode 41 (Wisdom and Contemplation).
 */
class MetaCognitiveMonitor {
public:
    /**
     * @brief Constructor initializes monitoring system
     * @param config Configuration parameters
     */
    explicit MetaCognitiveMonitor(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~MetaCognitiveMonitor();
    
    /**
     * @brief Record cognitive snapshot for introspection
     * @param snapshot Snapshot of current cognitive state
     */
    void record_snapshot(const CognitiveSnapshot& snapshot);
    
    /**
     * @brief Get cognitive state introspection
     * @param lookback_cycles Number of cycles to analyze
     * @return Introspection metrics and insights
     */
    std::unordered_map<std::string, float> introspect_state(uint32_t lookback_cycles = 100);
    
    /**
     * @brief Analyze cognitive process efficiency
     * @param process_name Name of the process to analyze
     * @return Process efficiency metrics
     */
    ProcessMetrics analyze_process(const std::string& process_name) const;
    
    /**
     * @brief Detect cognitive patterns in recent processing
     * @param pattern_type Type of pattern to detect
     * @return Pattern strength [0.0, 1.0]
     */
    float detect_pattern(const std::string& pattern_type);
    
    /**
     * @brief Get meta-cognitive awareness score
     * @return Current meta-cognitive awareness level [0.0, 1.0]
     */
    float get_awareness_score() const;
    
    /**
     * @brief Update process metrics
     * @param process_name Process identifier
     * @param processing_time Processing time in microseconds
     * @param relevance Relevance score
     * @param confidence Confidence score
     * @param success Whether processing succeeded
     */
    void update_process_metrics(
        const std::string& process_name,
        float processing_time,
        float relevance,
        float confidence,
        bool success
    );
    
    /**
     * @brief Reset monitoring state
     */
    void reset();
    
    /**
     * @brief Get monitoring statistics
     * @return Current monitoring statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Snapshot history
    std::deque<CognitiveSnapshot> snapshot_history_;
    size_t max_history_size_;
    
    // Process metrics tracking
    std::unordered_map<std::string, ProcessMetrics> process_metrics_;
    
    // Pattern detection state
    std::unordered_map<std::string, float> detected_patterns_;
    
    // Meta-cognitive awareness tracking
    float awareness_score_;
    std::chrono::high_resolution_clock::time_point last_awareness_update_;
    
    // Thread safety
    mutable std::mutex monitor_mutex_;
    
    // Statistics
    uint64_t total_snapshots_;
    
    // Internal methods
    void update_awareness_score();
    float compute_pattern_strength(
        const std::string& pattern_type,
        const std::vector<CognitiveSnapshot>& snapshots
    );
    void prune_history();
    float compute_temporal_coherence(const std::vector<CognitiveSnapshot>& snapshots) const;
    float compute_relevance_stability(const std::vector<CognitiveSnapshot>& snapshots) const;
};

} // namespace rros

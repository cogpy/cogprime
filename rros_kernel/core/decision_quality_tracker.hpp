#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <deque>
#include <mutex>

namespace rros {

/**
 * @brief Decision outcome record for quality assessment
 */
struct DecisionOutcome {
    uint64_t decision_id;
    std::chrono::high_resolution_clock::time_point decision_time;
    std::chrono::high_resolution_clock::time_point outcome_time;
    std::vector<float> decision_context;
    std::vector<float> decision_output;
    float predicted_quality;
    float actual_quality;
    float reward_signal;
    bool success;
    std::string decision_type;
    
    DecisionOutcome();
};

/**
 * @brief Decision quality metrics aggregated over time
 */
struct QualityMetrics {
    float average_quality;
    float quality_variance;
    float prediction_accuracy;
    float calibration_score;
    uint64_t total_decisions;
    uint64_t successful_decisions;
    std::chrono::high_resolution_clock::time_point last_update;
    
    QualityMetrics();
};

/**
 * @brief Decision quality assessment and tracking system
 * 
 * Monitors decision-making processes, evaluates decision quality,
 * and provides metrics for self-optimization. Implements decision
 * quality tracking as part of meta-cognitive monitoring (Episode 10)
 * and practical wisdom assessment (Episode 6 - Aristotle's Wisdom).
 */
class DecisionQualityTracker {
public:
    /**
     * @brief Constructor initializes tracking system
     * @param config Configuration parameters
     */
    explicit DecisionQualityTracker(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~DecisionQualityTracker();
    
    /**
     * @brief Record a decision for quality tracking
     * @param decision_context Context in which decision was made
     * @param decision_output Decision made by the system
     * @param predicted_quality Predicted quality of decision
     * @param decision_type Type/category of decision
     * @return Unique decision ID for later outcome recording
     */
    uint64_t record_decision(
        const std::vector<float>& decision_context,
        const std::vector<float>& decision_output,
        float predicted_quality,
        const std::string& decision_type = "default"
    );
    
    /**
     * @brief Record outcome of a previously made decision
     * @param decision_id ID from record_decision call
     * @param actual_quality Actual observed quality
     * @param reward_signal Reward signal from environment
     * @param success Whether decision was successful
     */
    void record_outcome(
        uint64_t decision_id,
        float actual_quality,
        float reward_signal,
        bool success
    );
    
    /**
     * @brief Get decision quality metrics
     * @param decision_type Optional filter by decision type
     * @return Quality metrics for decisions
     */
    QualityMetrics get_quality_metrics(const std::string& decision_type = "") const;
    
    /**
     * @brief Assess correlation between predicted and actual quality
     * @param lookback Number of recent decisions to analyze
     * @return Correlation coefficient [-1.0, 1.0]
     */
    float assess_prediction_correlation(uint32_t lookback = 100);
    
    /**
     * @brief Get calibration score (how well predictions match outcomes)
     * @param decision_type Optional filter by decision type
     * @return Calibration score [0.0, 1.0], 1.0 is perfectly calibrated
     */
    float get_calibration_score(const std::string& decision_type = "") const;
    
    /**
     * @brief Analyze decision quality trends over time
     * @param window_size Number of decisions per window
     * @return Trend slope (positive = improving, negative = declining)
     */
    float analyze_quality_trend(uint32_t window_size = 50);
    
    /**
     * @brief Get decision success rate
     * @param decision_type Optional filter by decision type
     * @return Success rate [0.0, 1.0]
     */
    float get_success_rate(const std::string& decision_type = "") const;
    
    /**
     * @brief Reset tracking state
     */
    void reset();
    
    /**
     * @brief Get tracking statistics
     * @return Current tracking statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Decision history
    std::deque<DecisionOutcome> decision_history_;
    size_t max_history_size_;
    
    // Pending decisions (waiting for outcomes)
    std::unordered_map<uint64_t, DecisionOutcome> pending_decisions_;
    
    // Quality metrics by decision type
    std::unordered_map<std::string, QualityMetrics> type_metrics_;
    
    // Global quality metrics
    QualityMetrics global_metrics_;
    
    // Decision ID counter
    uint64_t next_decision_id_;
    
    // Thread safety
    mutable std::mutex tracker_mutex_;
    
    // Statistics
    uint64_t total_decisions_recorded_;
    uint64_t total_outcomes_recorded_;
    
    // Internal methods
    void update_metrics(const DecisionOutcome& outcome);
    float compute_calibration(const std::vector<DecisionOutcome>& outcomes) const;
    float compute_correlation(const std::vector<DecisionOutcome>& outcomes) const;
    void prune_history();
};

} // namespace rros

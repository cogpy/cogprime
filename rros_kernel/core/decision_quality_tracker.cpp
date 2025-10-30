#include "decision_quality_tracker.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

// DecisionOutcome implementation
DecisionOutcome::DecisionOutcome()
    : decision_id(0)
    , decision_time(std::chrono::high_resolution_clock::now())
    , outcome_time(std::chrono::high_resolution_clock::now())
    , predicted_quality(0.0f)
    , actual_quality(0.0f)
    , reward_signal(0.0f)
    , success(false)
{}

// QualityMetrics implementation
QualityMetrics::QualityMetrics()
    : average_quality(0.0f)
    , quality_variance(0.0f)
    , prediction_accuracy(0.0f)
    , calibration_score(0.0f)
    , total_decisions(0)
    , successful_decisions(0)
    , last_update(std::chrono::high_resolution_clock::now())
{}

// DecisionQualityTracker implementation
DecisionQualityTracker::DecisionQualityTracker(
    const std::unordered_map<std::string, float>& config
)
    : config_(config)
    , max_history_size_(static_cast<size_t>(
        config.count("max_history_size") ? config.at("max_history_size") : 1000
      ))
    , next_decision_id_(1)
    , total_decisions_recorded_(0)
    , total_outcomes_recorded_(0)
{
    decision_history_.clear();
    pending_decisions_.clear();
}

DecisionQualityTracker::~DecisionQualityTracker() {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    decision_history_.clear();
    pending_decisions_.clear();
}

uint64_t DecisionQualityTracker::record_decision(
    const std::vector<float>& decision_context,
    const std::vector<float>& decision_output,
    float predicted_quality,
    const std::string& decision_type
) {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    DecisionOutcome outcome;
    outcome.decision_id = next_decision_id_++;
    outcome.decision_time = std::chrono::high_resolution_clock::now();
    outcome.decision_context = decision_context;
    outcome.decision_output = decision_output;
    outcome.predicted_quality = predicted_quality;
    outcome.decision_type = decision_type;
    
    pending_decisions_[outcome.decision_id] = outcome;
    total_decisions_recorded_++;
    
    return outcome.decision_id;
}

void DecisionQualityTracker::record_outcome(
    uint64_t decision_id,
    float actual_quality,
    float reward_signal,
    bool success
) {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    auto it = pending_decisions_.find(decision_id);
    if (it == pending_decisions_.end()) {
        return; // Decision not found
    }
    
    DecisionOutcome& outcome = it->second;
    outcome.outcome_time = std::chrono::high_resolution_clock::now();
    outcome.actual_quality = actual_quality;
    outcome.reward_signal = reward_signal;
    outcome.success = success;
    
    // Move to history BEFORE calling update_metrics
    decision_history_.push_back(outcome);
    total_outcomes_recorded_++;
    
    // Now update metrics (this can safely access decision_history_)
    update_metrics(outcome);
    
    // Remove from pending after all processing is done
    pending_decisions_.erase(it);
    
    // Prune history if needed
    if (decision_history_.size() > max_history_size_) {
        prune_history();
    }
}

QualityMetrics DecisionQualityTracker::get_quality_metrics(
    const std::string& decision_type
) const {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    if (decision_type.empty()) {
        return global_metrics_;
    }
    
    auto it = type_metrics_.find(decision_type);
    if (it != type_metrics_.end()) {
        return it->second;
    }
    
    return QualityMetrics();
}

float DecisionQualityTracker::assess_prediction_correlation(uint32_t lookback) {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    if (decision_history_.empty()) {
        return 0.0f;
    }
    
    size_t start_idx = decision_history_.size() > lookback ?
                       decision_history_.size() - lookback : 0;
    
    std::vector<DecisionOutcome> recent_decisions(
        decision_history_.begin() + start_idx,
        decision_history_.end()
    );
    
    return compute_correlation(recent_decisions);
}

float DecisionQualityTracker::get_calibration_score(
    const std::string& decision_type
) const {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    if (decision_type.empty()) {
        return global_metrics_.calibration_score;
    }
    
    auto it = type_metrics_.find(decision_type);
    if (it != type_metrics_.end()) {
        return it->second.calibration_score;
    }
    
    return 0.0f;
}

float DecisionQualityTracker::analyze_quality_trend(uint32_t window_size) {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    if (decision_history_.size() < window_size * 2) {
        return 0.0f; // Not enough data
    }
    
    // Compare early window to late window
    size_t early_start = 0;
    size_t early_end = window_size;
    size_t late_start = decision_history_.size() - window_size;
    size_t late_end = decision_history_.size();
    
    float early_avg = 0.0f;
    for (size_t i = early_start; i < early_end; i++) {
        early_avg += decision_history_[i].actual_quality;
    }
    early_avg /= window_size;
    
    float late_avg = 0.0f;
    for (size_t i = late_start; i < late_end; i++) {
        late_avg += decision_history_[i].actual_quality;
    }
    late_avg /= window_size;
    
    // Trend slope (normalized by time)
    float trend = late_avg - early_avg;
    
    return trend;
}

float DecisionQualityTracker::get_success_rate(const std::string& decision_type) const {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    if (decision_type.empty()) {
        if (global_metrics_.total_decisions == 0) {
            return 0.0f;
        }
        return static_cast<float>(global_metrics_.successful_decisions) /
               static_cast<float>(global_metrics_.total_decisions);
    }
    
    auto it = type_metrics_.find(decision_type);
    if (it != type_metrics_.end()) {
        const auto& metrics = it->second;
        if (metrics.total_decisions == 0) {
            return 0.0f;
        }
        return static_cast<float>(metrics.successful_decisions) /
               static_cast<float>(metrics.total_decisions);
    }
    
    return 0.0f;
}

void DecisionQualityTracker::reset() {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    decision_history_.clear();
    pending_decisions_.clear();
    type_metrics_.clear();
    global_metrics_ = QualityMetrics();
    next_decision_id_ = 1;
    total_decisions_recorded_ = 0;
    total_outcomes_recorded_ = 0;
}

std::unordered_map<std::string, float> DecisionQualityTracker::get_statistics() const {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    
    std::unordered_map<std::string, float> stats;
    stats["total_decisions_recorded"] = static_cast<float>(total_decisions_recorded_);
    stats["total_outcomes_recorded"] = static_cast<float>(total_outcomes_recorded_);
    stats["pending_decisions"] = static_cast<float>(pending_decisions_.size());
    stats["history_size"] = static_cast<float>(decision_history_.size());
    
    // Calculate success rate directly without calling get_success_rate (avoid deadlock)
    if (global_metrics_.total_decisions == 0) {
        stats["global_success_rate"] = 0.0f;
    } else {
        stats["global_success_rate"] = static_cast<float>(global_metrics_.successful_decisions) /
                                       static_cast<float>(global_metrics_.total_decisions);
    }
    
    stats["global_calibration"] = global_metrics_.calibration_score;
    stats["tracked_types"] = static_cast<float>(type_metrics_.size());
    
    return stats;
}

// Private methods

void DecisionQualityTracker::update_metrics(const DecisionOutcome& outcome) {
    const float alpha = 0.1f; // EMA smoothing factor
    
    // Update global metrics
    global_metrics_.total_decisions++;
    if (outcome.success) {
        global_metrics_.successful_decisions++;
    }
    
    global_metrics_.average_quality = (1.0f - alpha) * global_metrics_.average_quality +
                                      alpha * outcome.actual_quality;
    
    float quality_error = outcome.actual_quality - outcome.predicted_quality;
    global_metrics_.prediction_accuracy = (1.0f - alpha) * global_metrics_.prediction_accuracy +
                                          alpha * (1.0f - std::abs(quality_error));
    
    global_metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    // Update calibration score only if we have enough data
    if (decision_history_.size() >= 5) {
        size_t lookback = std::min(size_t(100), decision_history_.size());
        std::vector<DecisionOutcome> recent;
        for (size_t i = decision_history_.size() - lookback; i < decision_history_.size(); i++) {
            recent.push_back(decision_history_[i]);
        }
        global_metrics_.calibration_score = compute_calibration(recent);
    }
    
    // Update type-specific metrics
    if (type_metrics_.find(outcome.decision_type) == type_metrics_.end()) {
        type_metrics_[outcome.decision_type] = QualityMetrics();
    }
    
    auto& type_metric = type_metrics_[outcome.decision_type];
    type_metric.total_decisions++;
    if (outcome.success) {
        type_metric.successful_decisions++;
    }
    
    type_metric.average_quality = (1.0f - alpha) * type_metric.average_quality +
                                  alpha * outcome.actual_quality;
    type_metric.prediction_accuracy = (1.0f - alpha) * type_metric.prediction_accuracy +
                                      alpha * (1.0f - std::abs(quality_error));
    type_metric.last_update = std::chrono::high_resolution_clock::now();
}

float DecisionQualityTracker::compute_calibration(
    const std::vector<DecisionOutcome>& outcomes
) const {
    if (outcomes.empty()) {
        return 0.0f;
    }
    
    // Calibration measures how well predicted quality matches actual quality
    // Perfect calibration = 1.0, poor calibration = 0.0
    
    float total_error = 0.0f;
    for (const auto& outcome : outcomes) {
        float error = std::abs(outcome.predicted_quality - outcome.actual_quality);
        total_error += error;
    }
    
    float avg_error = total_error / outcomes.size();
    
    // Convert error to calibration score
    float calibration = 1.0f / (1.0f + avg_error);
    
    return calibration;
}

float DecisionQualityTracker::compute_correlation(
    const std::vector<DecisionOutcome>& outcomes
) const {
    if (outcomes.size() < 2) {
        return 0.0f;
    }
    
    // Compute Pearson correlation coefficient between predicted and actual quality
    
    float predicted_mean = 0.0f;
    float actual_mean = 0.0f;
    
    for (const auto& outcome : outcomes) {
        predicted_mean += outcome.predicted_quality;
        actual_mean += outcome.actual_quality;
    }
    
    predicted_mean /= outcomes.size();
    actual_mean /= outcomes.size();
    
    float numerator = 0.0f;
    float predicted_var = 0.0f;
    float actual_var = 0.0f;
    
    for (const auto& outcome : outcomes) {
        float predicted_diff = outcome.predicted_quality - predicted_mean;
        float actual_diff = outcome.actual_quality - actual_mean;
        
        numerator += predicted_diff * actual_diff;
        predicted_var += predicted_diff * predicted_diff;
        actual_var += actual_diff * actual_diff;
    }
    
    float denominator = std::sqrt(predicted_var * actual_var);
    
    if (denominator < 1e-6f) {
        return 0.0f;
    }
    
    float correlation = numerator / denominator;
    
    // Clamp to [-1, 1]
    return std::max(-1.0f, std::min(1.0f, correlation));
}

void DecisionQualityTracker::prune_history() {
    // Remove oldest 20% of history
    size_t remove_count = max_history_size_ / 5;
    if (decision_history_.size() > remove_count) {
        decision_history_.erase(
            decision_history_.begin(),
            decision_history_.begin() + remove_count
        );
    }
}

} // namespace rros

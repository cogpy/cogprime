#include "meta_cognitive_monitor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

// CognitiveSnapshot implementation
CognitiveSnapshot::CognitiveSnapshot()
    : timestamp(std::chrono::high_resolution_clock::now())
    , relevance_score(0.0f)
    , confidence(0.0f)
    , cycle_id(0)
{}

// ProcessMetrics implementation
ProcessMetrics::ProcessMetrics()
    : average_processing_time_us(0.0f)
    , average_relevance(0.0f)
    , average_confidence(0.0f)
    , success_rate(0.0f)
    , total_cycles(0)
    , successful_cycles(0)
    , last_update(std::chrono::high_resolution_clock::now())
{}

void ProcessMetrics::update(float processing_time, float relevance, float confidence, bool success) {
    const float alpha = 0.1f; // Exponential moving average factor
    
    average_processing_time_us = (1.0f - alpha) * average_processing_time_us + alpha * processing_time;
    average_relevance = (1.0f - alpha) * average_relevance + alpha * relevance;
    average_confidence = (1.0f - alpha) * average_confidence + alpha * confidence;
    
    total_cycles++;
    if (success) {
        successful_cycles++;
    }
    
    success_rate = static_cast<float>(successful_cycles) / static_cast<float>(total_cycles);
    last_update = std::chrono::high_resolution_clock::now();
}

// MetaCognitiveMonitor implementation
MetaCognitiveMonitor::MetaCognitiveMonitor(const std::unordered_map<std::string, float>& config)
    : config_(config)
    , max_history_size_(static_cast<size_t>(config.count("max_history_size") ? config.at("max_history_size") : 1000))
    , awareness_score_(0.5f)
    , last_awareness_update_(std::chrono::high_resolution_clock::now())
    , total_snapshots_(0)
{
    snapshot_history_.clear();
}

MetaCognitiveMonitor::~MetaCognitiveMonitor() {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    snapshot_history_.clear();
    process_metrics_.clear();
}

void MetaCognitiveMonitor::record_snapshot(const CognitiveSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    snapshot_history_.push_back(snapshot);
    total_snapshots_++;
    
    // Prune history if needed
    if (snapshot_history_.size() > max_history_size_) {
        prune_history();
    }
    
    // Update awareness score periodically
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_awareness_update_
    ).count();
    
    if (duration > 100) { // Update every 100ms
        update_awareness_score();
        last_awareness_update_ = now;
    }
}

std::unordered_map<std::string, float> MetaCognitiveMonitor::introspect_state(uint32_t lookback_cycles) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    std::unordered_map<std::string, float> introspection;
    
    if (snapshot_history_.empty()) {
        return introspection;
    }
    
    // Get recent snapshots
    size_t start_idx = snapshot_history_.size() > lookback_cycles ? 
                       snapshot_history_.size() - lookback_cycles : 0;
    
    std::vector<CognitiveSnapshot> recent_snapshots(
        snapshot_history_.begin() + start_idx,
        snapshot_history_.end()
    );
    
    // Compute introspection metrics
    float avg_relevance = 0.0f;
    float avg_confidence = 0.0f;
    float relevance_variance = 0.0f;
    
    for (const auto& snapshot : recent_snapshots) {
        avg_relevance += snapshot.relevance_score;
        avg_confidence += snapshot.confidence;
    }
    
    if (!recent_snapshots.empty()) {
        avg_relevance /= recent_snapshots.size();
        avg_confidence /= recent_snapshots.size();
        
        // Compute variance
        for (const auto& snapshot : recent_snapshots) {
            float diff = snapshot.relevance_score - avg_relevance;
            relevance_variance += diff * diff;
        }
        relevance_variance /= recent_snapshots.size();
    }
    
    introspection["average_relevance"] = avg_relevance;
    introspection["average_confidence"] = avg_confidence;
    introspection["relevance_stability"] = compute_relevance_stability(recent_snapshots);
    introspection["temporal_coherence"] = compute_temporal_coherence(recent_snapshots);
    introspection["relevance_variance"] = relevance_variance;
    introspection["awareness_score"] = awareness_score_;
    introspection["snapshot_count"] = static_cast<float>(recent_snapshots.size());
    
    return introspection;
}

ProcessMetrics MetaCognitiveMonitor::analyze_process(const std::string& process_name) const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    auto it = process_metrics_.find(process_name);
    if (it != process_metrics_.end()) {
        return it->second;
    }
    
    return ProcessMetrics();
}

float MetaCognitiveMonitor::detect_pattern(const std::string& pattern_type) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    if (snapshot_history_.empty()) {
        return 0.0f;
    }
    
    // Compute pattern strength based on recent snapshots
    size_t lookback = std::min(snapshot_history_.size(), size_t(50));
    std::vector<CognitiveSnapshot> recent_snapshots(
        snapshot_history_.end() - lookback,
        snapshot_history_.end()
    );
    
    float pattern_strength = compute_pattern_strength(pattern_type, recent_snapshots);
    detected_patterns_[pattern_type] = pattern_strength;
    
    return pattern_strength;
}

float MetaCognitiveMonitor::get_awareness_score() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return awareness_score_;
}

void MetaCognitiveMonitor::update_process_metrics(
    const std::string& process_name,
    float processing_time,
    float relevance,
    float confidence,
    bool success
) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    if (process_metrics_.find(process_name) == process_metrics_.end()) {
        process_metrics_[process_name] = ProcessMetrics();
    }
    
    process_metrics_[process_name].update(processing_time, relevance, confidence, success);
}

void MetaCognitiveMonitor::reset() {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    snapshot_history_.clear();
    process_metrics_.clear();
    detected_patterns_.clear();
    awareness_score_ = 0.5f;
    total_snapshots_ = 0;
}

std::unordered_map<std::string, float> MetaCognitiveMonitor::get_statistics() const {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    
    std::unordered_map<std::string, float> stats;
    stats["total_snapshots"] = static_cast<float>(total_snapshots_);
    stats["current_history_size"] = static_cast<float>(snapshot_history_.size());
    stats["awareness_score"] = awareness_score_;
    stats["tracked_processes"] = static_cast<float>(process_metrics_.size());
    stats["detected_patterns"] = static_cast<float>(detected_patterns_.size());
    
    return stats;
}

// Private methods

void MetaCognitiveMonitor::update_awareness_score() {
    if (snapshot_history_.empty()) {
        awareness_score_ = 0.5f;
        return;
    }
    
    // Meta-cognitive awareness based on:
    // 1. Temporal coherence of processing
    // 2. Stability of relevance realizations
    // 3. Consistency of confidence estimates
    
    size_t lookback = std::min(snapshot_history_.size(), size_t(100));
    std::vector<CognitiveSnapshot> recent_snapshots(
        snapshot_history_.end() - lookback,
        snapshot_history_.end()
    );
    
    float coherence = compute_temporal_coherence(recent_snapshots);
    float stability = compute_relevance_stability(recent_snapshots);
    
    // Compute confidence consistency
    float confidence_mean = 0.0f;
    for (const auto& snapshot : recent_snapshots) {
        confidence_mean += snapshot.confidence;
    }
    confidence_mean /= recent_snapshots.size();
    
    float confidence_variance = 0.0f;
    for (const auto& snapshot : recent_snapshots) {
        float diff = snapshot.confidence - confidence_mean;
        confidence_variance += diff * diff;
    }
    confidence_variance /= recent_snapshots.size();
    
    float confidence_consistency = 1.0f / (1.0f + std::sqrt(confidence_variance));
    
    // Combine factors
    awareness_score_ = 0.4f * coherence + 0.3f * stability + 0.3f * confidence_consistency;
    
    // Clamp to [0, 1]
    awareness_score_ = std::max(0.0f, std::min(1.0f, awareness_score_));
}

float MetaCognitiveMonitor::compute_pattern_strength(
    const std::string& pattern_type,
    const std::vector<CognitiveSnapshot>& snapshots
) {
    if (snapshots.empty()) {
        return 0.0f;
    }
    
    if (pattern_type == "increasing_relevance") {
        // Detect upward trend in relevance
        int increasing_count = 0;
        for (size_t i = 1; i < snapshots.size(); i++) {
            if (snapshots[i].relevance_score > snapshots[i-1].relevance_score) {
                increasing_count++;
            }
        }
        return static_cast<float>(increasing_count) / static_cast<float>(snapshots.size() - 1);
    }
    else if (pattern_type == "decreasing_relevance") {
        // Detect downward trend in relevance
        int decreasing_count = 0;
        for (size_t i = 1; i < snapshots.size(); i++) {
            if (snapshots[i].relevance_score < snapshots[i-1].relevance_score) {
                decreasing_count++;
            }
        }
        return static_cast<float>(decreasing_count) / static_cast<float>(snapshots.size() - 1);
    }
    else if (pattern_type == "oscillating") {
        // Detect oscillating pattern
        int direction_changes = 0;
        for (size_t i = 2; i < snapshots.size(); i++) {
            float d1 = snapshots[i-1].relevance_score - snapshots[i-2].relevance_score;
            float d2 = snapshots[i].relevance_score - snapshots[i-1].relevance_score;
            if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
                direction_changes++;
            }
        }
        return static_cast<float>(direction_changes) / static_cast<float>(snapshots.size() - 2);
    }
    
    return 0.0f;
}

void MetaCognitiveMonitor::prune_history() {
    // Remove oldest 20% of history
    size_t remove_count = max_history_size_ / 5;
    if (snapshot_history_.size() > remove_count) {
        snapshot_history_.erase(
            snapshot_history_.begin(),
            snapshot_history_.begin() + remove_count
        );
    }
}

float MetaCognitiveMonitor::compute_temporal_coherence(
    const std::vector<CognitiveSnapshot>& snapshots
) const {
    if (snapshots.size() < 2) {
        return 1.0f;
    }
    
    // Measure smoothness of state transitions
    float total_transition_cost = 0.0f;
    
    for (size_t i = 1; i < snapshots.size(); i++) {
        const auto& prev = snapshots[i-1];
        const auto& curr = snapshots[i];
        
        // Compute L2 distance between state vectors
        float distance = 0.0f;
        size_t min_size = std::min(prev.state_vector.size(), curr.state_vector.size());
        
        for (size_t j = 0; j < min_size; j++) {
            float diff = curr.state_vector[j] - prev.state_vector[j];
            distance += diff * diff;
        }
        
        distance = std::sqrt(distance);
        total_transition_cost += distance;
    }
    
    // Normalize by number of transitions
    float avg_transition = total_transition_cost / (snapshots.size() - 1);
    
    // Convert to coherence score (lower transition cost = higher coherence)
    float coherence = 1.0f / (1.0f + avg_transition);
    
    return coherence;
}

float MetaCognitiveMonitor::compute_relevance_stability(
    const std::vector<CognitiveSnapshot>& snapshots
) const {
    if (snapshots.empty()) {
        return 0.0f;
    }
    
    // Compute standard deviation of relevance scores
    float mean = 0.0f;
    for (const auto& snapshot : snapshots) {
        mean += snapshot.relevance_score;
    }
    mean /= snapshots.size();
    
    float variance = 0.0f;
    for (const auto& snapshot : snapshots) {
        float diff = snapshot.relevance_score - mean;
        variance += diff * diff;
    }
    variance /= snapshots.size();
    
    float std_dev = std::sqrt(variance);
    
    // Convert to stability score (lower variance = higher stability)
    float stability = 1.0f / (1.0f + std_dev);
    
    return stability;
}

} // namespace rros

#include "memory_core.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace rros {

MemoryCore::MemoryCore(const std::unordered_map<std::string, float>& config) :
    config_(config),
    capacity_(10000),
    decay_rate_(0.99f),
    next_id_(0)
{
    if (config_.find("memory_capacity") != config_.end()) {
        capacity_ = static_cast<size_t>(config_["memory_capacity"]);
    }
    if (config_.find("memory_decay") != config_.end()) {
        decay_rate_ = config_["memory_decay"];
    }
}

size_t MemoryCore::store(
    const std::vector<float>& data,
    float relevance,
    const std::vector<float>& attention
) {
    auto trace = std::make_shared<MemoryTrace>();
    trace->data = data;
    trace->relevance = relevance;
    trace->attention = attention;
    trace->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    trace->decay = 1.0f;
    trace->access_count = 0;
    
    // Find insertion point based on relevance (keep sorted)
    size_t insert_pos = find_insertion_point(relevance);
    
    // Insert trace
    if (insert_pos >= traces_.size()) {
        traces_.push_back(trace);
    } else {
        traces_.insert(traces_.begin() + insert_pos, trace);
    }
    
    // Prune if over capacity
    if (traces_.size() > capacity_) {
        prune_weak_traces();
    }
    
    return next_id_++;
}

std::vector<std::shared_ptr<MemoryTrace>> MemoryCore::retrieve(
    const std::vector<float>& query,
    size_t max_results
) {
    std::vector<std::pair<float, std::shared_ptr<MemoryTrace>>> candidates;
    
    // Compute similarity scores for all traces
    for (auto& trace : traces_) {
        if (trace && !trace->data.empty()) {
            float similarity = compute_similarity(query, trace->data);
            
            // Weight by relevance and decay
            float score = similarity * trace->relevance * trace->decay;
            
            candidates.emplace_back(score, trace);
        }
    }
    
    // Sort by score (descending)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract top results
    std::vector<std::shared_ptr<MemoryTrace>> results;
    size_t count = std::min(max_results, candidates.size());
    
    for (size_t i = 0; i < count; ++i) {
        auto trace = candidates[i].second;
        trace->access_count++;
        results.push_back(trace);
    }
    
    return results;
}

float MemoryCore::compute_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    
    // Cosine similarity
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    
    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

void MemoryCore::apply_decay() {
    auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    
    for (auto& trace : traces_) {
        if (trace) {
            // Time-based decay
            uint64_t age = current_time - trace->timestamp;
            float time_factor = std::exp(-static_cast<float>(age) / 86400000.0f); // 24 hours in ms
            
            // Access-based reinforcement
            float access_factor = 1.0f + std::log(1.0f + trace->access_count) * 0.1f;
            
            trace->decay = decay_rate_ * time_factor * access_factor;
        }
    }
}

void MemoryCore::prune_weak_traces() {
    // Remove traces with very low decay
    traces_.erase(
        std::remove_if(traces_.begin(), traces_.end(),
                      [](const std::shared_ptr<MemoryTrace>& trace) {
                          return !trace || trace->decay < 0.01f;
                      }),
        traces_.end()
    );
    
    // If still over capacity, remove oldest low-relevance traces
    while (traces_.size() > capacity_) {
        auto min_it = std::min_element(traces_.begin(), traces_.end(),
                                      [](const auto& a, const auto& b) {
                                          float score_a = a ? a->relevance * a->decay : 0.0f;
                                          float score_b = b ? b->relevance * b->decay : 0.0f;
                                          return score_a < score_b;
                                      });
        
        if (min_it != traces_.end()) {
            traces_.erase(min_it);
        } else {
            break;
        }
    }
}

size_t MemoryCore::find_insertion_point(float relevance) {
    // Binary search for insertion point (keep sorted by relevance)
    size_t left = 0, right = traces_.size();
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        
        if (traces_[mid] && traces_[mid]->relevance > relevance) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

void MemoryCore::update_config(const std::unordered_map<std::string, float>& config) {
    for (const auto& [key, value] : config) {
        config_[key] = value;
        
        if (key == "memory_capacity") {
            capacity_ = static_cast<size_t>(value);
        } else if (key == "memory_decay") {
            decay_rate_ = value;
        }
    }
}

void MemoryCore::reset() {
    traces_.clear();
    next_id_ = 0;
}

float MemoryCore::get_utilization() const {
    return static_cast<float>(traces_.size()) / static_cast<float>(capacity_);
}

void MemoryCore::consolidate() {
    apply_decay();
    prune_weak_traces();
    
    // Sort by relevance * decay for efficiency
    std::sort(traces_.begin(), traces_.end(),
              [](const auto& a, const auto& b) {
                  float score_a = a ? a->relevance * a->decay : 0.0f;
                  float score_b = b ? b->relevance * b->decay : 0.0f;
                  return score_a > score_b;
              });
}

} // namespace rros
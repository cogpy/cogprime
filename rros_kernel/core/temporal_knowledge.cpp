#include "temporal_knowledge.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rros {

TemporalKnowledge::TemporalKnowledge(
    size_t event_capacity,
    size_t pattern_capacity
) : event_capacity_(event_capacity),
    pattern_capacity_(pattern_capacity),
    next_event_id_(0),
    next_pattern_id_(0)
{
    events_.reserve(event_capacity);
    patterns_.reserve(pattern_capacity);
}

uint64_t TemporalKnowledge::get_current_time_us() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

size_t TemporalKnowledge::add_event(
    const std::vector<float>& state_vector,
    float importance
) {
    if (events_.size() >= event_capacity_) {
        // Remove oldest, least important event
        auto min_it = std::min_element(events_.begin(), events_.end(),
            [](const TemporalEvent& a, const TemporalEvent& b) {
                return a.importance < b.importance;
            });
        if (min_it != events_.end()) {
            // Remove from ID mapping
            size_t removed_index = std::distance(events_.begin(), min_it);
            // Find and remove the ID mapping
            for (auto it = event_id_to_index_.begin(); it != event_id_to_index_.end(); ++it) {
                if (it->second == removed_index) {
                    event_id_to_index_.erase(it);
                    break;
                }
            }
            events_.erase(min_it);
            
            // Update remaining indices
            for (auto& pair : event_id_to_index_) {
                if (pair.second > removed_index) {
                    pair.second--;
                }
            }
        }
    }
    
    TemporalEvent event;
    event.timestamp = get_current_time_us();
    event.state_vector = state_vector;
    event.importance = std::max(0.0f, std::min(1.0f, importance));
    
    size_t event_id = next_event_id_++;
    size_t event_index = events_.size();
    events_.push_back(event);
    event_id_to_index_[event_id] = event_index;
    
    return event_id;
}

size_t TemporalKnowledge::get_event_index(size_t event_id) const {
    auto it = event_id_to_index_.find(event_id);
    if (it == event_id_to_index_.end()) {
        throw std::out_of_range("Invalid event ID");
    }
    return it->second;
}

void TemporalKnowledge::add_causal_relation(
    size_t cause_id,
    size_t effect_id,
    float strength,
    float confidence
) {
    // Validate event IDs exist
    size_t cause_idx = get_event_index(cause_id);
    size_t effect_idx = get_event_index(effect_id);
    
    CausalRelation relation;
    relation.cause_event = cause_id;
    relation.effect_event = effect_id;
    relation.strength = std::max(0.0f, std::min(1.0f, strength));
    relation.confidence = std::max(0.0f, std::min(1.0f, confidence));
    
    // Calculate time lag
    uint64_t cause_time = events_[cause_idx].timestamp;
    uint64_t effect_time = events_[effect_idx].timestamp;
    relation.time_lag = (effect_time > cause_time) ? 
        (effect_time - cause_time) : 0;
    
    // Update event causal links
    events_[cause_idx].causal_successors.push_back(effect_id);
    events_[effect_idx].causal_predecessors.push_back(cause_id);
    
    causal_relations_.push_back(relation);
}

size_t TemporalKnowledge::register_pattern(const TemporalPattern& pattern) {
    if (patterns_.size() >= pattern_capacity_) {
        // Remove least predictive pattern
        auto min_it = std::min_element(patterns_.begin(), patterns_.end(),
            [](const TemporalPattern& a, const TemporalPattern& b) {
                return a.predictive_power < b.predictive_power;
            });
        if (min_it != patterns_.end()) {
            patterns_.erase(min_it);
        }
    }
    
    patterns_.push_back(pattern);
    return next_pattern_id_++;
}

std::vector<size_t> TemporalKnowledge::get_events_in_range(
    uint64_t start_time,
    uint64_t end_time
) const {
    std::vector<size_t> result;
    
    for (size_t i = 0; i < events_.size(); ++i) {
        if (events_[i].timestamp >= start_time && 
            events_[i].timestamp <= end_time) {
            result.push_back(i);
        }
    }
    
    return result;
}

const TemporalEvent& TemporalKnowledge::get_event(size_t event_id) const {
    size_t index = get_event_index(event_id);
    return events_[index];
}

std::vector<CausalRelation> TemporalKnowledge::get_causal_predecessors(
    size_t event_id
) const {
    std::vector<CausalRelation> result;
    
    for (const auto& relation : causal_relations_) {
        if (relation.effect_event == event_id) {
            result.push_back(relation);
        }
    }
    
    return result;
}

std::vector<CausalRelation> TemporalKnowledge::get_causal_successors(
    size_t event_id
) const {
    std::vector<CausalRelation> result;
    
    for (const auto& relation : causal_relations_) {
        if (relation.cause_event == event_id) {
            result.push_back(relation);
        }
    }
    
    return result;
}

size_t TemporalKnowledge::create_abstraction(
    const std::vector<size_t>& event_ids,
    uint64_t time_scale
) {
    if (event_ids.empty()) {
        throw std::invalid_argument("Cannot create abstraction from empty event set");
    }
    
    TemporalAbstraction abstraction;
    abstraction.name = "abstraction_" + std::to_string(abstractions_.size());
    abstraction.time_scale = time_scale;
    abstraction.component_events = event_ids;
    
    // Compute abstract features by averaging event states
    size_t first_idx = get_event_index(event_ids[0]);
    size_t feature_dim = events_[first_idx].state_vector.size();
    abstraction.abstract_features.resize(feature_dim, 0.0f);
    
    for (size_t event_id : event_ids) {
        size_t idx = get_event_index(event_id);
        
        const auto& event = events_[idx];
        for (size_t i = 0; i < feature_dim && i < event.state_vector.size(); ++i) {
            abstraction.abstract_features[i] += event.state_vector[i];
        }
    }
    
    // Normalize
    for (float& val : abstraction.abstract_features) {
        val /= static_cast<float>(event_ids.size());
    }
    
    // Compute coherence based on variance
    float variance = 0.0f;
    for (size_t event_id : event_ids) {
        size_t idx = get_event_index(event_id);
        
        const auto& event = events_[idx];
        for (size_t i = 0; i < feature_dim && i < event.state_vector.size(); ++i) {
            float diff = event.state_vector[i] - abstraction.abstract_features[i];
            variance += diff * diff;
        }
    }
    variance /= (feature_dim * event_ids.size());
    abstraction.coherence = 1.0f / (1.0f + variance);
    
    abstractions_.push_back(abstraction);
    return abstractions_.size() - 1;
}

float TemporalKnowledge::compute_sequence_similarity(
    const std::vector<std::vector<float>>& seq1,
    const std::vector<std::vector<float>>& seq2
) const {
    if (seq1.empty() || seq2.empty()) return 0.0f;
    
    size_t min_len = std::min(seq1.size(), seq2.size());
    float total_similarity = 0.0f;
    
    for (size_t i = 0; i < min_len; ++i) {
        const auto& v1 = seq1[i];
        const auto& v2 = seq2[i];
        
        if (v1.empty() || v2.empty()) continue;
        
        // Cosine similarity
        float dot = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        size_t min_dim = std::min(v1.size(), v2.size());
        for (size_t j = 0; j < min_dim; ++j) {
            dot += v1[j] * v2[j];
            norm1 += v1[j] * v1[j];
            norm2 += v2[j] * v2[j];
        }
        
        if (norm1 > 0 && norm2 > 0) {
            total_similarity += dot / (std::sqrt(norm1) * std::sqrt(norm2));
        }
    }
    
    return total_similarity / min_len;
}

std::unordered_map<std::string, float> TemporalKnowledge::get_statistics() const {
    std::unordered_map<std::string, float> stats;
    
    stats["event_count"] = static_cast<float>(events_.size());
    stats["pattern_count"] = static_cast<float>(patterns_.size());
    stats["causal_relation_count"] = static_cast<float>(causal_relations_.size());
    stats["abstraction_count"] = static_cast<float>(abstractions_.size());
    
    // Average importance
    float avg_importance = 0.0f;
    for (const auto& event : events_) {
        avg_importance += event.importance;
    }
    if (!events_.empty()) {
        avg_importance /= events_.size();
    }
    stats["avg_importance"] = avg_importance;
    
    // Average causal strength
    float avg_causal_strength = 0.0f;
    for (const auto& relation : causal_relations_) {
        avg_causal_strength += relation.strength;
    }
    if (!causal_relations_.empty()) {
        avg_causal_strength /= causal_relations_.size();
    }
    stats["avg_causal_strength"] = avg_causal_strength;
    
    return stats;
}

void TemporalKnowledge::clear() {
    events_.clear();
    patterns_.clear();
    causal_relations_.clear();
    abstractions_.clear();
    next_event_id_ = 0;
    next_pattern_id_ = 0;
}

} // namespace rros

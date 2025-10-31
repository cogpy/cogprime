#include "episodic_memory.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>

namespace rros {

EpisodicMemorySystem::EpisodicMemorySystem(
    size_t capacity,
    uint64_t time_bin_size
) : capacity_(capacity),
    time_bin_size_(time_bin_size),
    next_memory_id_(0)
{
    memories_.reserve(capacity);
}

uint64_t EpisodicMemorySystem::get_current_time_us() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

uint64_t EpisodicMemorySystem::compute_time_bin(uint64_t timestamp) const {
    return timestamp / time_bin_size_;
}

size_t EpisodicMemorySystem::encode_memory(
    const std::vector<float>& content,
    float importance,
    const std::vector<size_t>& temporal_events
) {
    // Check capacity and remove least important if needed
    if (memories_.size() >= capacity_) {
        size_t to_remove = find_least_important_memory();
        if (to_remove < memories_.size()) {
            size_t memory_id = memories_[to_remove].memory_id;
            
            // Remove from temporal index
            uint64_t time_bin = compute_time_bin(memories_[to_remove].encoding_time);
            auto& index = temporal_indices_[time_bin];
            index.memory_ids.erase(
                std::remove(index.memory_ids.begin(), index.memory_ids.end(), to_remove),
                index.memory_ids.end()
            );
            
            // Remove from ID mapping
            memory_id_to_index_.erase(memory_id);
            
            memories_.erase(memories_.begin() + to_remove);
            
            // Update remaining indices
            for (auto& pair : memory_id_to_index_) {
                if (pair.second > to_remove) {
                    pair.second--;
                }
            }
        }
    }
    
    EpisodicMemory memory;
    memory.memory_id = next_memory_id_++;
    memory.content = content;
    memory.encoding_time = get_current_time_us();
    memory.last_accessed = memory.encoding_time;
    memory.access_count = 0;
    memory.consolidation_level = 0.5f; // Initial consolidation
    memory.temporal_context = temporal_events;
    memory.vividness = 1.0f; // Maximum vividness at encoding
    
    size_t memory_idx = memories_.size();
    memories_.push_back(memory);
    memory_id_to_index_[memory.memory_id] = memory_idx;
    
    // Update temporal index
    update_temporal_index(memory_idx, memory.encoding_time);
    
    return memory.memory_id;
}

std::vector<size_t> EpisodicMemorySystem::retrieve_by_time(
    uint64_t query_time,
    uint64_t time_window,
    size_t max_results
) {
    std::vector<std::pair<size_t, float>> candidates;
    
    uint64_t start_time = (query_time > time_window) ? 
        (query_time - time_window) : 0;
    uint64_t end_time = query_time + time_window;
    
    // Search relevant time bins
    uint64_t start_bin = compute_time_bin(start_time);
    uint64_t end_bin = compute_time_bin(end_time);
    
    for (uint64_t bin = start_bin; bin <= end_bin; ++bin) {
        auto it = temporal_indices_.find(bin);
        if (it != temporal_indices_.end()) {
            for (size_t memory_idx : it->second.memory_ids) {
                if (memory_idx >= memories_.size()) continue;
                
                const auto& memory = memories_[memory_idx];
                if (memory.encoding_time >= start_time && 
                    memory.encoding_time <= end_time) {
                    
                    float relevance = compute_temporal_relevance(
                        memory.encoding_time,
                        query_time,
                        time_window
                    );
                    
                    candidates.emplace_back(memory_idx, relevance);
                }
            }
        }
    }
    
    // Sort by relevance
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract top results
    std::vector<size_t> results;
    size_t count = std::min(max_results, candidates.size());
    for (size_t i = 0; i < count; ++i) {
        results.push_back(candidates[i].first);
        // Update access statistics
        memories_[candidates[i].first].access_count++;
        memories_[candidates[i].first].last_accessed = get_current_time_us();
    }
    
    return results;
}

std::vector<size_t> EpisodicMemorySystem::retrieve_by_content(
    const std::vector<float>& query,
    size_t max_results
) {
    std::vector<std::pair<size_t, float>> candidates;
    
    for (size_t i = 0; i < memories_.size(); ++i) {
        float similarity = compute_content_similarity(query, memories_[i].content);
        candidates.emplace_back(i, similarity * memories_[i].vividness);
    }
    
    // Sort by similarity
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract top results
    std::vector<size_t> results;
    size_t count = std::min(max_results, candidates.size());
    for (size_t i = 0; i < count; ++i) {
        results.push_back(candidates[i].first);
        // Update access statistics
        memories_[candidates[i].first].access_count++;
        memories_[candidates[i].first].last_accessed = get_current_time_us();
    }
    
    return results;
}

std::vector<size_t> EpisodicMemorySystem::retrieve_contextual(
    const std::vector<float>& query,
    uint64_t query_time,
    uint64_t time_window,
    float temporal_weight,
    size_t max_results
) {
    temporal_weight = std::max(0.0f, std::min(1.0f, temporal_weight));
    float content_weight = 1.0f - temporal_weight;
    
    std::vector<std::pair<size_t, float>> candidates;
    
    uint64_t start_time = (query_time > time_window) ? 
        (query_time - time_window) : 0;
    uint64_t end_time = query_time + time_window;
    
    for (size_t i = 0; i < memories_.size(); ++i) {
        const auto& memory = memories_[i];
        
        float temporal_score = compute_temporal_relevance(
            memory.encoding_time,
            query_time,
            time_window
        );
        
        float content_score = compute_content_similarity(query, memory.content);
        
        float combined_score = temporal_weight * temporal_score + 
                              content_weight * content_score;
        combined_score *= memory.vividness;
        
        candidates.emplace_back(i, combined_score);
    }
    
    // Sort by combined score
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract top results
    std::vector<size_t> results;
    size_t count = std::min(max_results, candidates.size());
    for (size_t i = 0; i < count; ++i) {
        results.push_back(candidates[i].first);
        // Update access statistics
        memories_[candidates[i].first].access_count++;
        memories_[candidates[i].first].last_accessed = get_current_time_us();
    }
    
    return results;
}

size_t EpisodicMemorySystem::consolidate_memories() {
    size_t consolidated = 0;
    
    // Group memories by temporal proximity
    std::vector<std::vector<size_t>> clusters;
    std::vector<bool> assigned(memories_.size(), false);
    
    for (size_t i = 0; i < memories_.size(); ++i) {
        if (assigned[i]) continue;
        
        std::vector<size_t> cluster;
        cluster.push_back(i);
        assigned[i] = true;
        
        // Find temporally close memories with similar content
        for (size_t j = i + 1; j < memories_.size(); ++j) {
            if (assigned[j]) continue;
            
            uint64_t time_diff = (memories_[j].encoding_time > memories_[i].encoding_time) ?
                (memories_[j].encoding_time - memories_[i].encoding_time) :
                (memories_[i].encoding_time - memories_[j].encoding_time);
            
            if (time_diff < time_bin_size_ * 10) { // Within 10 time bins
                float similarity = compute_content_similarity(
                    memories_[i].content,
                    memories_[j].content
                );
                
                if (similarity > 0.8f) { // High similarity
                    cluster.push_back(j);
                    assigned[j] = true;
                }
            }
        }
        
        if (cluster.size() > 1) {
            clusters.push_back(cluster);
        }
    }
    
    // Increase consolidation level for clustered memories
    for (const auto& cluster : clusters) {
        for (size_t idx : cluster) {
            memories_[idx].consolidation_level = std::min(1.0f,
                memories_[idx].consolidation_level + 0.1f);
        }
        consolidated += cluster.size();
    }
    
    return consolidated;
}

void EpisodicMemorySystem::apply_forgetting(float decay_rate) {
    uint64_t current_time = get_current_time_us();
    
    for (auto& memory : memories_) {
        // Time since last access
        uint64_t time_diff = current_time - memory.last_accessed;
        float time_factor = static_cast<float>(time_diff) / 1000000.0f; // Convert to seconds
        
        // Apply forgetting curve (exponential decay)
        float decay = std::exp(-decay_rate * time_factor);
        
        // Memories with higher consolidation resist forgetting
        float resistance = 0.5f + 0.5f * memory.consolidation_level;
        
        memory.vividness *= (decay + (1.0f - decay) * resistance);
        memory.vividness = std::max(0.0f, memory.vividness);
    }
    
    // Remove memories with very low vividness
    memories_.erase(
        std::remove_if(memories_.begin(), memories_.end(),
            [](const EpisodicMemory& m) { return m.vividness < 0.01f; }),
        memories_.end()
    );
}

void EpisodicMemorySystem::tag_emotion(
    size_t memory_id,
    const std::string& emotion,
    float intensity
) {
    auto it = memory_id_to_index_.find(memory_id);
    if (it != memory_id_to_index_.end()) {
        auto& memory = memories_[it->second];
        memory.emotional_tags[emotion] = std::max(0.0f, std::min(1.0f, intensity));
        // Emotional memories are more vivid
        memory.vividness = std::min(1.0f, memory.vividness + intensity * 0.1f);
    }
}

const EpisodicMemory& EpisodicMemorySystem::get_memory(size_t memory_id) const {
    auto it = memory_id_to_index_.find(memory_id);
    if (it == memory_id_to_index_.end()) {
        throw std::out_of_range("Memory ID not found");
    }
    return memories_[it->second];
}

float EpisodicMemorySystem::compute_content_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b
) const {
    if (a.empty() || b.empty()) return 0.0f;
    
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    size_t min_size = std::min(a.size(), b.size());
    for (size_t i = 0; i < min_size; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a > 0 && norm_b > 0) {
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
    
    return 0.0f;
}

float EpisodicMemorySystem::compute_temporal_relevance(
    uint64_t memory_time,
    uint64_t query_time,
    uint64_t time_window
) const {
    uint64_t time_diff = (memory_time > query_time) ?
        (memory_time - query_time) : (query_time - memory_time);
    
    if (time_diff > time_window) return 0.0f;
    
    // Gaussian decay
    float normalized_diff = static_cast<float>(time_diff) / 
                           static_cast<float>(time_window);
    return std::exp(-normalized_diff * normalized_diff);
}

void EpisodicMemorySystem::update_temporal_index(
    size_t memory_idx,
    uint64_t timestamp
) {
    uint64_t time_bin = compute_time_bin(timestamp);
    
    auto& index = temporal_indices_[time_bin];
    index.time_bin = time_bin;
    index.memory_ids.push_back(memory_idx);
    
    // Update average importance
    float total_importance = 0.0f;
    size_t count = 0;
    for (size_t idx : index.memory_ids) {
        if (idx < memories_.size()) {
            total_importance += memories_[idx].consolidation_level;
            count++;
        }
    }
    index.average_importance = (count > 0) ? (total_importance / count) : 0.0f;
}

size_t EpisodicMemorySystem::find_least_important_memory() const {
    if (memories_.empty()) return 0;
    
    size_t min_idx = 0;
    float min_score = std::numeric_limits<float>::max();
    
    for (size_t i = 0; i < memories_.size(); ++i) {
        const auto& memory = memories_[i];
        
        // Combined score: vividness, consolidation, and access count
        float score = memory.vividness * 
                     memory.consolidation_level * 
                     (1.0f + std::log1p(static_cast<float>(memory.access_count)));
        
        if (score < min_score) {
            min_score = score;
            min_idx = i;
        }
    }
    
    return min_idx;
}

std::unordered_map<std::string, float> EpisodicMemorySystem::get_statistics() const {
    std::unordered_map<std::string, float> stats;
    
    stats["memory_count"] = static_cast<float>(memories_.size());
    stats["capacity_utilization"] = static_cast<float>(memories_.size()) / 
                                    static_cast<float>(capacity_);
    
    // Average vividness
    float avg_vividness = 0.0f;
    float avg_consolidation = 0.0f;
    float avg_access_count = 0.0f;
    
    for (const auto& memory : memories_) {
        avg_vividness += memory.vividness;
        avg_consolidation += memory.consolidation_level;
        avg_access_count += static_cast<float>(memory.access_count);
    }
    
    if (!memories_.empty()) {
        avg_vividness /= memories_.size();
        avg_consolidation /= memories_.size();
        avg_access_count /= memories_.size();
    }
    
    stats["avg_vividness"] = avg_vividness;
    stats["avg_consolidation"] = avg_consolidation;
    stats["avg_access_count"] = avg_access_count;
    stats["temporal_index_count"] = static_cast<float>(temporal_indices_.size());
    
    return stats;
}

void EpisodicMemorySystem::clear() {
    memories_.clear();
    memory_id_to_index_.clear();
    temporal_indices_.clear();
    next_memory_id_ = 0;
}

} // namespace rros

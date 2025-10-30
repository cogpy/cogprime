#include "memory_consolidator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {

MemoryConsolidator::MemoryConsolidator(size_t feature_dim,
                                       size_t max_capacity,
                                       float consolidation_threshold)
    : feature_dim_(feature_dim),
      max_capacity_(max_capacity),
      consolidation_threshold_(consolidation_threshold),
      total_consolidations_(0),
      total_stores_(0),
      total_retrievals_(0) {
    memories_.reserve(max_capacity);
}

size_t MemoryConsolidator::store_memory(const std::vector<float>& content, float importance) {
    MemoryTrace trace;
    trace.content = content;
    trace.importance = importance;
    trace.access_count = 0.0f;
    trace.timestamp = std::chrono::system_clock::now();
    trace.consolidation_strength = 0.0f;
    
    total_stores_++;
    
    // If at capacity, replace least important memory
    if (memories_.size() >= max_capacity_) {
        size_t replace_idx = find_least_important();
        memories_[replace_idx] = trace;
        return replace_idx;
    }
    
    memories_.push_back(trace);
    return memories_.size() - 1;
}

std::vector<size_t> MemoryConsolidator::retrieve_memories(
    const std::vector<float>& query, size_t k) {
    
    total_retrievals_++;
    
    if (memories_.empty()) {
        return {};
    }
    
    // Calculate similarities with importance boosting
    std::vector<std::pair<float, size_t>> similarities;
    similarities.reserve(memories_.size());
    
    for (size_t i = 0; i < memories_.size(); ++i) {
        float sim = cosine_similarity(query, memories_[i].content);
        
        // Boost by importance
        float boosted_sim = sim * (1.0f + 0.3f * memories_[i].importance);
        
        similarities.push_back({boosted_sim, i});
        
        // Update access count
        memories_[i].access_count += 1.0f;
    }
    
    // Sort by similarity (descending)
    std::partial_sort(similarities.begin(),
                     similarities.begin() + std::min(k, similarities.size()),
                     similarities.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract top k indices
    std::vector<size_t> result;
    size_t count = std::min(k, similarities.size());
    result.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        result.push_back(similarities[i].second);
    }
    
    return result;
}

size_t MemoryConsolidator::consolidate() {
    if (memories_.size() < 10) {
        return 0;  // Need minimum memories for consolidation
    }
    
    size_t consolidated_count = 0;
    bool merged = true;
    
    // Iteratively merge similar memories
    while (merged && memories_.size() > max_capacity_ * 0.8f) {
        merged = false;
        
        // Find most similar pair above threshold
        float max_similarity = consolidation_threshold_;
        size_t merge_i = 0, merge_j = 0;
        
        for (size_t i = 0; i < memories_.size() && !merged; ++i) {
            for (size_t j = i + 1; j < memories_.size(); ++j) {
                float sim = cosine_similarity(memories_[i].content, memories_[j].content);
                
                if (sim > max_similarity) {
                    max_similarity = sim;
                    merge_i = i;
                    merge_j = j;
                    merged = true;
                    break;
                }
            }
        }
        
        if (merged) {
            // Merge the two memories
            auto merged_memory = merge_memories(memories_[merge_i], memories_[merge_j]);
            
            // Replace first memory with merged version
            memories_[merge_i] = merged_memory;
            
            // Remove second memory
            memories_.erase(memories_.begin() + merge_j);
            
            consolidated_count++;
            total_consolidations_++;
        }
    }
    
    return consolidated_count;
}

const MemoryTrace& MemoryConsolidator::get_memory(size_t index) const {
    return memories_.at(index);
}

std::unordered_map<std::string, float> MemoryConsolidator::get_consolidation_stats() const {
    std::unordered_map<std::string, float> stats{};
    
    stats["memory_count"] = static_cast<float>(memories_.size());
    stats["max_capacity"] = static_cast<float>(max_capacity_);
    stats["utilization"] = static_cast<float>(memories_.size()) / max_capacity_;
    stats["total_consolidations"] = static_cast<float>(total_consolidations_);
    stats["total_stores"] = static_cast<float>(total_stores_);
    stats["total_retrievals"] = static_cast<float>(total_retrievals_);
    
    if (!memories_.empty()) {
        float avg_importance = 0.0f;
        float avg_access = 0.0f;
        for (const auto& mem : memories_) {
            avg_importance += mem.importance;
            avg_access += mem.access_count;
        }
        stats["avg_importance"] = avg_importance / memories_.size();
        stats["avg_access_count"] = avg_access / memories_.size();
    }
    
    // Calculate storage reduction
    if (total_stores_ > 0) {
        float reduction = (static_cast<float>(total_stores_ - memories_.size()) / 
                          total_stores_) * 100.0f;
        stats["storage_reduction_pct"] = reduction;
    }
    
    return stats;
}

void MemoryConsolidator::clear() {
    memories_.clear();
    total_consolidations_ = 0;
    total_stores_ = 0;
    total_retrievals_ = 0;
}

// Private helper functions

float MemoryConsolidator::cosine_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b) const {
    
    if (a.size() != b.size()) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a < 1e-8f || norm_b < 1e-8f) {
        return 0.0f;
    }
    
    return dot_product / (norm_a * norm_b);
}

MemoryTrace MemoryConsolidator::merge_memories(
    const MemoryTrace& m1,
    const MemoryTrace& m2) const {
    
    MemoryTrace merged;
    merged.content.resize(feature_dim_);
    
    // Weighted average based on importance
    float w1 = m1.importance;
    float w2 = m2.importance;
    float total_w = w1 + w2 + 1e-8f;
    
    for (size_t i = 0; i < feature_dim_; ++i) {
        merged.content[i] = (w1 * m1.content[i] + w2 * m2.content[i]) / total_w;
    }
    
    // Merge other attributes
    merged.importance = std::max(m1.importance, m2.importance);
    merged.access_count = m1.access_count + m2.access_count;
    merged.timestamp = std::max(m1.timestamp, m2.timestamp);
    merged.consolidation_strength = (m1.consolidation_strength + m2.consolidation_strength) / 2.0f + 0.1f;
    
    return merged;
}

float MemoryConsolidator::calculate_combined_importance(const MemoryTrace& memory) const {
    // Combined metric: importance (70%) + normalized access count (30%)
    float max_access = 0.0f;
    for (const auto& mem : memories_) {
        max_access = std::max(max_access, mem.access_count);
    }
    
    float normalized_access = max_access > 0.0f ? memory.access_count / max_access : 0.0f;
    
    return memory.importance * 0.7f + normalized_access * 0.3f;
}

size_t MemoryConsolidator::find_least_important() const {
    if (memories_.empty()) {
        return 0;
    }
    
    float min_importance = std::numeric_limits<float>::max();
    size_t min_idx = 0;
    
    for (size_t i = 0; i < memories_.size(); ++i) {
        float combined_importance = calculate_combined_importance(memories_[i]);
        if (combined_importance < min_importance) {
            min_importance = combined_importance;
            min_idx = i;
        }
    }
    
    return min_idx;
}

} // namespace rros

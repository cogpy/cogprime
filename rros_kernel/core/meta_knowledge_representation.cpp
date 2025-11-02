#include "meta_knowledge_representation.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace rros {

// MetaKnowledgeEntry implementation
MetaKnowledgeEntry::MetaKnowledgeEntry()
    : confidence(0.5f)
    , utility(0.5f)
    , creation_time(std::chrono::high_resolution_clock::now())
    , last_access(std::chrono::high_resolution_clock::now())
    , access_count(0)
{}

// KnowledgeNode implementation
KnowledgeNode::KnowledgeNode() {}

KnowledgeNode::KnowledgeNode(const std::string& id)
    : node_id(id)
{}

// KnowledgeQuery implementation
KnowledgeQuery::KnowledgeQuery()
    : max_results(10)
{}

// VectorIndex implementation
std::vector<std::pair<std::string, float>> 
MetaKnowledgeRepresentation::VectorIndex::search(
    const std::vector<float>& query,
    size_t k
) const {
    std::vector<std::pair<std::string, float>> results;
    
    for (const auto& [id, embedding] : entries) {
        // Compute cosine similarity
        float dot = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        size_t min_size = std::min(query.size(), embedding.size());
        for (size_t i = 0; i < min_size; i++) {
            dot += query[i] * embedding[i];
            norm1 += query[i] * query[i];
            norm2 += embedding[i] * embedding[i];
        }
        
        float similarity = 0.0f;
        if (norm1 > 0 && norm2 > 0) {
            similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));
        }
        
        results.push_back({id, similarity});
    }
    
    // Sort by similarity (descending)
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top k
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}

// MetaKnowledgeRepresentation implementation
MetaKnowledgeRepresentation::MetaKnowledgeRepresentation(
    const std::unordered_map<std::string, float>& config
)
    : config_(config)
    , total_entries_(0)
    , total_retrievals_(0)
    , total_consolidations_(0)
{}

MetaKnowledgeRepresentation::~MetaKnowledgeRepresentation() {}

std::string MetaKnowledgeRepresentation::store_knowledge(
    const std::unordered_map<std::string, float>& knowledge,
    const std::string& context
) {
    MetaKnowledgeEntry entry;
    entry.id = generate_entry_id();
    entry.type = context;
    entry.embedding = encode_knowledge(knowledge);
    entry.symbolic_form = encode_symbolic(knowledge, context);
    entry.confidence = 0.7f;
    entry.utility = 0.6f;
    
    knowledge_store_[entry.id] = entry;
    total_entries_++;
    
    update_vector_index(entry.id, entry.embedding);
    
    return entry.id;
}

std::vector<MetaKnowledgeEntry> MetaKnowledgeRepresentation::retrieve_knowledge(
    const KnowledgeQuery& query
) {
    total_retrievals_++;
    
    std::vector<MetaKnowledgeEntry> results;
    
    if (query.query_type == "similarity" && !query.query_vector.empty()) {
        auto search_results = vector_index_.search(query.query_vector, query.max_results);
        
        for (const auto& [id, similarity] : search_results) {
            auto it = knowledge_store_.find(id);
            if (it != knowledge_store_.end()) {
                auto entry = it->second;
                entry.access_count++;
                entry.last_access = std::chrono::high_resolution_clock::now();
                knowledge_store_[id] = entry;
                results.push_back(entry);
            }
        }
    }
    
    return results;
}

void MetaKnowledgeRepresentation::update_knowledge(
    const std::string& entry_id,
    const std::unordered_map<std::string, float>& updates
) {
    auto it = knowledge_store_.find(entry_id);
    if (it == knowledge_store_.end()) {
        return;
    }
    
    // Update embedding
    auto new_embedding = encode_knowledge(updates);
    
    // Merge with existing embedding
    auto& entry = it->second;
    if (entry.embedding.size() == new_embedding.size()) {
        for (size_t i = 0; i < entry.embedding.size(); i++) {
            entry.embedding[i] = 0.7f * entry.embedding[i] + 0.3f * new_embedding[i];
        }
    }
    
    entry.last_access = std::chrono::high_resolution_clock::now();
    update_vector_index(entry_id, entry.embedding);
}

std::string MetaKnowledgeRepresentation::create_node(const KnowledgeNode& node) {
    std::string node_id = node.node_id.empty() ? generate_entry_id() : node.node_id;
    
    KnowledgeNode stored_node = node;
    stored_node.node_id = node_id;
    
    knowledge_graph_[node_id] = stored_node;
    
    return node_id;
}

void MetaKnowledgeRepresentation::create_relationship(
    const std::string& source_id,
    const std::string& target_id,
    const std::string& relation_type,
    float strength
) {
    Relationship rel;
    rel.source = source_id;
    rel.target = target_id;
    rel.type = relation_type;
    rel.strength = strength;
    
    relationships_.push_back(rel);
    
    // Update node connections
    auto source_it = knowledge_graph_.find(source_id);
    if (source_it != knowledge_graph_.end()) {
        source_it->second.connected_nodes.push_back(target_id);
    }
}

std::vector<KnowledgeNode> MetaKnowledgeRepresentation::query_graph(
    const std::string& start_node,
    uint32_t max_depth
) {
    std::vector<KnowledgeNode> subgraph;
    std::vector<std::string> to_visit = {start_node};
    std::unordered_map<std::string, uint32_t> visited;
    
    while (!to_visit.empty()) {
        std::string current = to_visit.back();
        to_visit.pop_back();
        
        if (visited.find(current) != visited.end()) {
            continue;
        }
        
        auto it = knowledge_graph_.find(current);
        if (it == knowledge_graph_.end()) {
            continue;
        }
        
        uint32_t current_depth = visited.size() > 0 ? 
            visited[current] : 0;
        
        if (current_depth >= max_depth) {
            continue;
        }
        
        visited[current] = current_depth;
        subgraph.push_back(it->second);
        
        // Add connected nodes
        for (const auto& connected : it->second.connected_nodes) {
            if (visited.find(connected) == visited.end()) {
                to_visit.push_back(connected);
            }
        }
    }
    
    return subgraph;
}

size_t MetaKnowledgeRepresentation::consolidate_knowledge(float similarity_threshold) {
    size_t consolidated_count = 0;
    
    std::vector<std::pair<std::string, std::string>> to_merge;
    
    // Find similar entries
    std::vector<std::string> ids;
    for (const auto& [id, entry] : knowledge_store_) {
        ids.push_back(id);
    }
    
    for (size_t i = 0; i < ids.size(); i++) {
        for (size_t j = i + 1; j < ids.size(); j++) {
            const auto& entry1 = knowledge_store_[ids[i]];
            const auto& entry2 = knowledge_store_[ids[j]];
            
            float similarity = compute_similarity(entry1.embedding, entry2.embedding);
            
            if (similarity >= similarity_threshold) {
                to_merge.push_back({ids[i], ids[j]});
            }
        }
    }
    
    // Merge entries
    for (const auto& [target, source] : to_merge) {
        merge_entries(target, source);
        consolidated_count++;
    }
    
    total_consolidations_ += consolidated_count;
    
    return consolidated_count;
}

std::unordered_map<std::string, float> MetaKnowledgeRepresentation::generate_summary(
    const std::string& knowledge_type
) {
    std::unordered_map<std::string, float> summary;
    
    std::vector<MetaKnowledgeEntry> entries;
    for (const auto& [id, entry] : knowledge_store_) {
        if (entry.type == knowledge_type) {
            entries.push_back(entry);
        }
    }
    
    if (entries.empty()) {
        return summary;
    }
    
    summary["count"] = static_cast<float>(entries.size());
    
    float avg_confidence = 0.0f;
    float avg_utility = 0.0f;
    
    for (const auto& entry : entries) {
        avg_confidence += entry.confidence;
        avg_utility += entry.utility;
    }
    
    summary["average_confidence"] = avg_confidence / entries.size();
    summary["average_utility"] = avg_utility / entries.size();
    
    return summary;
}

float MetaKnowledgeRepresentation::compute_coherence() const {
    if (knowledge_store_.empty()) {
        return 1.0f;
    }
    
    // Coherence based on:
    // 1. Similarity within knowledge types
    // 2. Connectivity in knowledge graph
    
    float total_coherence = 0.0f;
    int comparisons = 0;
    
    std::unordered_map<std::string, std::vector<std::string>> by_type;
    for (const auto& [id, entry] : knowledge_store_) {
        by_type[entry.type].push_back(id);
    }
    
    for (const auto& [type, ids] : by_type) {
        if (ids.size() < 2) continue;
        
        for (size_t i = 0; i < ids.size() && i < 10; i++) {
            for (size_t j = i + 1; j < ids.size() && j < 10; j++) {
                const auto& e1 = knowledge_store_.at(ids[i]);
                const auto& e2 = knowledge_store_.at(ids[j]);
                
                float similarity = compute_similarity(e1.embedding, e2.embedding);
                total_coherence += similarity;
                comparisons++;
            }
        }
    }
    
    return comparisons > 0 ? total_coherence / comparisons : 0.5f;
}

std::unordered_map<std::string, float> MetaKnowledgeRepresentation::detect_gaps(
    const std::string& domain
) {
    std::unordered_map<std::string, float> gaps;
    
    // Count entries by type
    std::unordered_map<std::string, int> type_counts;
    for (const auto& [id, entry] : knowledge_store_) {
        type_counts[entry.type]++;
    }
    
    // Identify underrepresented types
    int total = knowledge_store_.size();
    for (const auto& [type, count] : type_counts) {
        float representation = static_cast<float>(count) / total;
        if (representation < 0.1f) {
            gaps[type] = 1.0f - representation;
        }
    }
    
    return gaps;
}

std::vector<std::string> MetaKnowledgeRepresentation::export_symbolic(
    const std::vector<std::string>& entry_ids
) {
    std::vector<std::string> symbolic_forms;
    
    if (entry_ids.empty()) {
        for (const auto& [id, entry] : knowledge_store_) {
            symbolic_forms.push_back(entry.symbolic_form);
        }
    } else {
        for (const auto& id : entry_ids) {
            auto it = knowledge_store_.find(id);
            if (it != knowledge_store_.end()) {
                symbolic_forms.push_back(it->second.symbolic_form);
            }
        }
    }
    
    return symbolic_forms;
}

std::unordered_map<std::string, float> MetaKnowledgeRepresentation::get_statistics() const {
    std::unordered_map<std::string, float> stats;
    stats["total_entries"] = static_cast<float>(total_entries_);
    stats["total_retrievals"] = static_cast<float>(total_retrievals_);
    stats["total_consolidations"] = static_cast<float>(total_consolidations_);
    stats["current_entries"] = static_cast<float>(knowledge_store_.size());
    stats["graph_nodes"] = static_cast<float>(knowledge_graph_.size());
    stats["relationships"] = static_cast<float>(relationships_.size());
    stats["coherence"] = compute_coherence();
    
    return stats;
}

void MetaKnowledgeRepresentation::reset() {
    knowledge_store_.clear();
    knowledge_graph_.clear();
    relationships_.clear();
    vector_index_.entries.clear();
    total_entries_ = 0;
    total_retrievals_ = 0;
    total_consolidations_ = 0;
}

// Private methods

std::vector<float> MetaKnowledgeRepresentation::encode_knowledge(
    const std::unordered_map<std::string, float>& knowledge
) {
    std::vector<float> encoding;
    
    for (const auto& [key, value] : knowledge) {
        encoding.push_back(value);
    }
    
    // Normalize
    if (!encoding.empty()) {
        float sum = std::accumulate(encoding.begin(), encoding.end(), 0.0f);
        if (sum > 0) {
            for (float& val : encoding) {
                val /= sum;
            }
        }
    }
    
    return encoding;
}

std::string MetaKnowledgeRepresentation::generate_entry_id() {
    static uint64_t counter = 0;
    std::stringstream ss;
    ss << "mk_" << std::setfill('0') << std::setw(10) << counter++;
    return ss.str();
}

float MetaKnowledgeRepresentation::compute_similarity(
    const std::vector<float>& v1,
    const std::vector<float>& v2
) const {
    if (v1.empty() || v2.empty()) {
        return 0.0f;
    }
    
    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    size_t min_size = std::min(v1.size(), v2.size());
    for (size_t i = 0; i < min_size; i++) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    if (norm1 > 0 && norm2 > 0) {
        return dot / (std::sqrt(norm1) * std::sqrt(norm2));
    }
    
    return 0.0f;
}

void MetaKnowledgeRepresentation::update_vector_index(
    const std::string& entry_id,
    const std::vector<float>& embedding
) {
    // Remove old entry if exists
    auto& entries = vector_index_.entries;
    entries.erase(
        std::remove_if(entries.begin(), entries.end(),
                      [&entry_id](const auto& p) { return p.first == entry_id; }),
        entries.end()
    );
    
    // Add new entry
    entries.push_back({entry_id, embedding});
}

void MetaKnowledgeRepresentation::merge_entries(
    const std::string& target_id,
    const std::string& source_id
) {
    auto target_it = knowledge_store_.find(target_id);
    auto source_it = knowledge_store_.find(source_id);
    
    if (target_it == knowledge_store_.end() || source_it == knowledge_store_.end()) {
        return;
    }
    
    auto& target = target_it->second;
    const auto& source = source_it->second;
    
    // Merge embeddings (average)
    if (target.embedding.size() == source.embedding.size()) {
        for (size_t i = 0; i < target.embedding.size(); i++) {
            target.embedding[i] = (target.embedding[i] + source.embedding[i]) / 2.0f;
        }
    }
    
    // Update confidence and utility
    target.confidence = (target.confidence + source.confidence) / 2.0f;
    target.utility = (target.utility + source.utility) / 2.0f;
    target.access_count += source.access_count;
    
    // Remove source
    knowledge_store_.erase(source_id);
    
    update_vector_index(target_id, target.embedding);
}

std::string MetaKnowledgeRepresentation::encode_symbolic(
    const std::unordered_map<std::string, float>& knowledge,
    const std::string& context
) {
    std::stringstream ss;
    ss << "[" << context << "] ";
    
    for (const auto& [key, value] : knowledge) {
        ss << key << "=" << std::fixed << std::setprecision(3) << value << " ";
    }
    
    return ss.str();
}

void MetaKnowledgeRepresentation::prune_old_entries() {
    // Remove entries not accessed in a long time
    auto now = std::chrono::high_resolution_clock::now();
    std::vector<std::string> to_remove;
    
    for (const auto& [id, entry] : knowledge_store_) {
        auto age = std::chrono::duration_cast<std::chrono::hours>(
            now - entry.last_access
        ).count();
        
        if (age > 24 * 7 && entry.access_count < 5) {  // 1 week old, rarely accessed
            to_remove.push_back(id);
        }
    }
    
    for (const auto& id : to_remove) {
        knowledge_store_.erase(id);
    }
}

} // namespace rros

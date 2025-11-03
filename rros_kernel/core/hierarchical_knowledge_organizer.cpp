#include "hierarchical_knowledge_organizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

namespace rros {

HierarchicalKnowledgeOrganizer::HierarchicalKnowledgeOrganizer(
    const std::unordered_map<std::string, float>& config
) : max_hierarchy_depth_(10),
    max_children_per_node_(10),
    attention_decay_rate_(0.9f)
{
    // Apply configuration
    if (config.count("max_hierarchy_depth")) {
        max_hierarchy_depth_ = static_cast<uint32_t>(config.at("max_hierarchy_depth"));
    }
    if (config.count("max_children_per_node")) {
        max_children_per_node_ = static_cast<uint32_t>(config.at("max_children_per_node"));
    }
    if (config.count("attention_decay_rate")) {
        attention_decay_rate_ = config.at("attention_decay_rate");
    }
}

HierarchicalKnowledgeOrganizer::~HierarchicalKnowledgeOrganizer() {
    // Clean shutdown
}

bool HierarchicalKnowledgeOrganizer::add_node(const HierarchicalNode& node) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node.node_id)) {
        return false;  // Node already exists
    }
    
    // Add to main storage
    nodes_[node.node_id] = node;
    
    // Update index
    node_index_.insert(node.node_id, node.node_id);
    
    // Update level index
    if (!level_index_.count(node.level)) {
        level_index_[node.level] = {};
    }
    level_index_[node.level].push_back(node.node_id);
    
    // Update max depth
    if (node.level > max_observed_depth_) {
        max_observed_depth_ = node.level;
    }
    
    // Update parent's children list
    if (!node.parent_id.empty() && nodes_.count(node.parent_id)) {
        nodes_[node.parent_id].children_ids.push_back(node.node_id);
    }
    
    return true;
}

bool HierarchicalKnowledgeOrganizer::remove_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (!nodes_.count(node_id)) {
        return false;
    }
    
    auto& node = nodes_[node_id];
    
    // Remove from parent's children
    if (!node.parent_id.empty() && nodes_.count(node.parent_id)) {
        auto& parent_children = nodes_[node.parent_id].children_ids;
        parent_children.erase(
            std::remove(parent_children.begin(), parent_children.end(), node_id),
            parent_children.end()
        );
    }
    
    // Reassign children to parent
    for (const auto& child_id : node.children_ids) {
        if (nodes_.count(child_id)) {
            nodes_[child_id].parent_id = node.parent_id;
            if (!node.parent_id.empty() && nodes_.count(node.parent_id)) {
                nodes_[node.parent_id].children_ids.push_back(child_id);
            }
        }
    }
    
    // Remove from level index
    auto& level_nodes = level_index_[node.level];
    level_nodes.erase(
        std::remove(level_nodes.begin(), level_nodes.end(), node_id),
        level_nodes.end()
    );
    
    // Remove from main storage
    nodes_.erase(node_id);
    
    return true;
}

std::pair<bool, HierarchicalNode> HierarchicalKnowledgeOrganizer::get_node(
    const std::string& node_id
) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node_id)) {
        auto& node = const_cast<HierarchicalNode&>(nodes_.at(node_id));
        node.access_count++;
        node.last_access = std::chrono::high_resolution_clock::now();
        return {true, nodes_.at(node_id)};
    }
    
    return {false, HierarchicalNode{}};
}

void HierarchicalKnowledgeOrganizer::update_node_data(
    const std::string& node_id,
    const std::vector<float>& data
) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node_id)) {
        nodes_[node_id].data = data;
    }
}

void HierarchicalKnowledgeOrganizer::set_parent(
    const std::string& node_id,
    const std::string& parent_id
) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (!nodes_.count(node_id)) return;
    
    auto& node = nodes_[node_id];
    
    // Remove from old parent
    if (!node.parent_id.empty() && nodes_.count(node.parent_id)) {
        auto& old_parent_children = nodes_[node.parent_id].children_ids;
        old_parent_children.erase(
            std::remove(old_parent_children.begin(), old_parent_children.end(), node_id),
            old_parent_children.end()
        );
    }
    
    // Set new parent
    node.parent_id = parent_id;
    
    // Add to new parent
    if (!parent_id.empty() && nodes_.count(parent_id)) {
        nodes_[parent_id].children_ids.push_back(node_id);
        
        // Update level
        uint32_t old_level = node.level;
        node.level = nodes_[parent_id].level + 1;
        update_level_index(node_id, old_level, node.level);
    }
}

std::vector<SearchResult> HierarchicalKnowledgeOrganizer::search(
    const HierarchicalQuery& query
) const {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<SearchResult> results;
    
    // Search nodes up to max_depth
    for (const auto& [id, node] : nodes_) {
        if (node.level <= query.max_depth) {
            float similarity = calculate_similarity(query.query_vector, node.data);
            
            if (similarity >= query.similarity_threshold) {
                SearchResult result;
                result.node_id = id;
                result.similarity_score = similarity;
                result.depth = node.level;
                result.data = node.data;
                
                results.push_back(result);
            }
        }
    }
    
    // Sort by similarity
    std::sort(results.begin(), results.end(),
             [](const auto& a, const auto& b) { 
                 return a.similarity_score > b.similarity_score; 
             });
    
    // Limit results
    if (results.size() > query.max_results) {
        results.resize(query.max_results);
    }
    
    // Track performance
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time
    ).count();
    
    ++total_queries_;
    total_query_time_us_ += duration;
    
    return results;
}

std::vector<HierarchicalNode> HierarchicalKnowledgeOrganizer::get_ancestors(
    const std::string& node_id
) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<HierarchicalNode> ancestors;
    
    if (!nodes_.count(node_id)) {
        return ancestors;
    }
    
    std::string current_id = nodes_.at(node_id).parent_id;
    
    while (!current_id.empty() && nodes_.count(current_id)) {
        ancestors.push_back(nodes_.at(current_id));
        current_id = nodes_.at(current_id).parent_id;
    }
    
    return ancestors;
}

std::vector<HierarchicalNode> HierarchicalKnowledgeOrganizer::get_descendants(
    const std::string& node_id,
    uint32_t max_depth
) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<HierarchicalNode> descendants;
    
    if (!nodes_.count(node_id)) {
        return descendants;
    }
    
    auto subtree_ids = collect_subtree(node_id, max_depth);
    
    for (const auto& id : subtree_ids) {
        if (id != node_id && nodes_.count(id)) {
            descendants.push_back(nodes_.at(id));
        }
    }
    
    return descendants;
}

std::vector<HierarchicalNode> HierarchicalKnowledgeOrganizer::get_siblings(
    const std::string& node_id
) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<HierarchicalNode> siblings;
    
    if (!nodes_.count(node_id)) {
        return siblings;
    }
    
    const auto& node = nodes_.at(node_id);
    
    if (node.parent_id.empty() || !nodes_.count(node.parent_id)) {
        return siblings;
    }
    
    const auto& parent = nodes_.at(node.parent_id);
    
    for (const auto& child_id : parent.children_ids) {
        if (child_id != node_id && nodes_.count(child_id)) {
            siblings.push_back(nodes_.at(child_id));
        }
    }
    
    return siblings;
}

std::vector<SearchResult> HierarchicalKnowledgeOrganizer::knn_search(
    const std::vector<float>& query_vector,
    uint32_t k,
    const std::string& start_node_id
) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<SearchResult> all_results;
    
    // If start node specified, search from that subtree
    std::vector<std::string> search_nodes;
    
    if (!start_node_id.empty() && nodes_.count(start_node_id)) {
        search_nodes = collect_subtree(start_node_id, UINT32_MAX);
    } else {
        for (const auto& [id, node] : nodes_) {
            search_nodes.push_back(id);
        }
    }
    
    // Calculate similarities
    for (const auto& node_id : search_nodes) {
        const auto& node = nodes_.at(node_id);
        float similarity = calculate_similarity(query_vector, node.data);
        
        SearchResult result;
        result.node_id = node_id;
        result.similarity_score = similarity;
        result.depth = node.level;
        result.data = node.data;
        
        all_results.push_back(result);
    }
    
    // Sort and take top k
    std::partial_sort(all_results.begin(), 
                     all_results.begin() + std::min(k, static_cast<uint32_t>(all_results.size())),
                     all_results.end(),
                     [](const auto& a, const auto& b) { 
                         return a.similarity_score > b.similarity_score; 
                     });
    
    if (all_results.size() > k) {
        all_results.resize(k);
    }
    
    return all_results;
}

std::string HierarchicalKnowledgeOrganizer::auto_organize(
    const std::vector<HierarchicalNode>& nodes,
    uint32_t max_children
) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes.empty()) {
        return "";
    }
    
    // Create root node
    HierarchicalNode root;
    root.node_id = "root";
    root.level = 0;
    root.importance = 1.0f;
    
    nodes_["root"] = root;
    
    // Organize nodes by clustering
    std::vector<HierarchicalNode> remaining = nodes;
    uint32_t current_level = 1;
    std::vector<std::string> current_parents = {"root"};
    
    while (!remaining.empty()) {
        std::vector<std::string> next_parents;
        
        for (const auto& parent_id : current_parents) {
            // Assign up to max_children nodes to this parent
            size_t children_count = std::min(
                static_cast<size_t>(max_children),
                remaining.size()
            );
            
            for (size_t i = 0; i < children_count && !remaining.empty(); ++i) {
                auto node = remaining.back();
                remaining.pop_back();
                
                node.parent_id = parent_id;
                node.level = current_level;
                
                add_node(node);
                next_parents.push_back(node.node_id);
            }
        }
        
        current_parents = next_parents;
        ++current_level;
    }
    
    return "root";
}

uint32_t HierarchicalKnowledgeOrganizer::balance_hierarchy() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    uint32_t rebalanced = 0;
    
    // Find imbalanced nodes (too many children)
    for (auto& [id, node] : nodes_) {
        if (node.children_ids.size() > max_children_per_node_) {
            rebalance_subtree(id);
            ++rebalanced;
        }
    }
    
    return rebalanced;
}

uint32_t HierarchicalKnowledgeOrganizer::merge_similar_nodes(float similarity_threshold) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    uint32_t merged = 0;
    std::vector<std::string> to_remove;
    
    // Find similar nodes at same level
    for (const auto& [level, node_ids] : level_index_) {
        for (size_t i = 0; i < node_ids.size(); ++i) {
            for (size_t j = i + 1; j < node_ids.size(); ++j) {
                const auto& node1 = nodes_[node_ids[i]];
                const auto& node2 = nodes_[node_ids[j]];
                
                float similarity = calculate_similarity(node1.data, node2.data);
                
                if (similarity >= similarity_threshold) {
                    // Merge node2 into node1
                    for (const auto& child_id : node2.children_ids) {
                        if (nodes_.count(child_id)) {
                            nodes_[child_id].parent_id = node_ids[i];
                            nodes_[node_ids[i]].children_ids.push_back(child_id);
                        }
                    }
                    
                    to_remove.push_back(node_ids[j]);
                    ++merged;
                }
            }
        }
    }
    
    // Remove merged nodes
    for (const auto& id : to_remove) {
        remove_node(id);
    }
    
    return merged;
}

uint32_t HierarchicalKnowledgeOrganizer::prune_infrequent_nodes(uint64_t access_threshold) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    uint32_t pruned = 0;
    std::vector<std::string> to_remove;
    
    for (const auto& [id, node] : nodes_) {
        if (node.access_count < access_threshold && !node.children_ids.empty()) {
            to_remove.push_back(id);
        }
    }
    
    for (const auto& id : to_remove) {
        if (remove_node(id)) {
            ++pruned;
        }
    }
    
    return pruned;
}

std::unordered_map<uint32_t, float> HierarchicalKnowledgeOrganizer::allocate_hierarchical_attention(
    float total_attention
) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::unordered_map<uint32_t, float> allocation;
    
    // Calculate total importance per level
    std::unordered_map<uint32_t, float> level_importance;
    
    for (const auto& [id, node] : nodes_) {
        level_importance[node.level] += node.importance;
    }
    
    // Allocate attention proportional to importance with decay
    float total_weighted_importance = 0.0f;
    for (const auto& [level, importance] : level_importance) {
        float decay = std::pow(attention_decay_rate_, static_cast<float>(level));
        total_weighted_importance += importance * decay;
    }
    
    if (total_weighted_importance > 0.0f) {
        for (const auto& [level, importance] : level_importance) {
            float decay = std::pow(attention_decay_rate_, static_cast<float>(level));
            allocation[level] = (importance * decay / total_weighted_importance) * total_attention;
        }
    }
    
    return allocation;
}

void HierarchicalKnowledgeOrganizer::update_importance(
    const std::string& node_id,
    float delta
) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node_id)) {
        nodes_[node_id].importance += delta;
        nodes_[node_id].importance = std::max(0.0f, nodes_[node_id].importance);
    }
}

void HierarchicalKnowledgeOrganizer::propagate_importance(
    const std::string& node_id,
    float decay
) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (!nodes_.count(node_id)) return;
    
    float current_importance = nodes_[node_id].importance;
    std::string parent_id = nodes_[node_id].parent_id;
    
    while (!parent_id.empty() && nodes_.count(parent_id)) {
        current_importance *= decay;
        nodes_[parent_id].importance += current_importance;
        parent_id = nodes_[parent_id].parent_id;
    }
}

std::unordered_map<std::string, float> HierarchicalKnowledgeOrganizer::get_stats() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::unordered_map<std::string, float> stats;
    
    stats["total_nodes"] = static_cast<float>(nodes_.size());
    stats["max_depth"] = static_cast<float>(max_observed_depth_.load());
    stats["avg_query_time_us"] = get_avg_query_time();
    stats["total_queries"] = static_cast<float>(total_queries_.load());
    
    // Calculate average children per node
    float total_children = 0.0f;
    for (const auto& [id, node] : nodes_) {
        total_children += static_cast<float>(node.children_ids.size());
    }
    stats["avg_children_per_node"] = nodes_.empty() ? 0.0f : total_children / nodes_.size();
    
    return stats;
}

float HierarchicalKnowledgeOrganizer::get_avg_query_time() const {
    uint64_t queries = total_queries_.load();
    return queries > 0 ? static_cast<float>(total_query_time_us_.load()) / queries : 0.0f;
}

uint32_t HierarchicalKnowledgeOrganizer::get_max_depth() const {
    return max_observed_depth_.load();
}

std::unordered_map<uint32_t, uint32_t> HierarchicalKnowledgeOrganizer::get_level_distribution() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::unordered_map<uint32_t, uint32_t> distribution;
    
    for (const auto& [level, node_ids] : level_index_) {
        distribution[level] = static_cast<uint32_t>(node_ids.size());
    }
    
    return distribution;
}

bool HierarchicalKnowledgeOrganizer::verify_performance() const {
    // Check if query time is O(log n)
    float avg_time = get_avg_query_time();
    size_t node_count = nodes_.size();
    
    if (node_count == 0) return true;
    
    // Expected time should grow logarithmically
    float expected_max_time = std::log2(static_cast<float>(node_count)) * 100.0f;  // 100us per log level
    
    return avg_time <= expected_max_time;
}

void HierarchicalKnowledgeOrganizer::update_config(
    const std::unordered_map<std::string, float>& config
) {
    if (config.count("max_hierarchy_depth")) {
        max_hierarchy_depth_ = static_cast<uint32_t>(config.at("max_hierarchy_depth"));
    }
    if (config.count("max_children_per_node")) {
        max_children_per_node_ = static_cast<uint32_t>(config.at("max_children_per_node"));
    }
    if (config.count("attention_decay_rate")) {
        attention_decay_rate_ = config.at("attention_decay_rate");
    }
}

void HierarchicalKnowledgeOrganizer::reset() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    nodes_.clear();
    level_index_.clear();
    
    total_queries_ = 0;
    total_query_time_us_ = 0;
    max_observed_depth_ = 0;
}

float HierarchicalKnowledgeOrganizer::calculate_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b
) const {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }
    
    // Cosine similarity
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
    
    return denominator > 0.0f ? dot_product / denominator : 0.0f;
}

void HierarchicalKnowledgeOrganizer::update_level_index(
    const std::string& node_id,
    uint32_t old_level,
    uint32_t new_level
) {
    // Remove from old level
    if (level_index_.count(old_level)) {
        auto& old_nodes = level_index_[old_level];
        old_nodes.erase(
            std::remove(old_nodes.begin(), old_nodes.end(), node_id),
            old_nodes.end()
        );
    }
    
    // Add to new level
    if (!level_index_.count(new_level)) {
        level_index_[new_level] = {};
    }
    level_index_[new_level].push_back(node_id);
}

void HierarchicalKnowledgeOrganizer::rebalance_subtree(const std::string& root_id) {
    if (!nodes_.count(root_id)) return;
    
    auto& root = nodes_[root_id];
    
    if (root.children_ids.size() <= max_children_per_node_) {
        return;  // Already balanced
    }
    
    // Create intermediate nodes to reduce children count
    std::vector<std::string> children = root.children_ids;
    root.children_ids.clear();
    
    uint32_t groups = (children.size() + max_children_per_node_ - 1) / max_children_per_node_;
    
    for (uint32_t g = 0; g < groups; ++g) {
        // Create intermediate node
        HierarchicalNode intermediate;
        intermediate.node_id = root_id + "_inter_" + std::to_string(g);
        intermediate.level = root.level + 1;
        intermediate.parent_id = root_id;
        intermediate.importance = root.importance * 0.5f;
        
        // Assign children to intermediate
        size_t start = g * max_children_per_node_;
        size_t end = std::min(start + max_children_per_node_, children.size());
        
        for (size_t i = start; i < end; ++i) {
            intermediate.children_ids.push_back(children[i]);
            if (nodes_.count(children[i])) {
                nodes_[children[i]].parent_id = intermediate.node_id;
                nodes_[children[i]].level = intermediate.level + 1;
            }
        }
        
        add_node(intermediate);
        root.children_ids.push_back(intermediate.node_id);
    }
}

std::vector<std::string> HierarchicalKnowledgeOrganizer::collect_subtree(
    const std::string& root_id,
    uint32_t max_depth
) const {
    std::vector<std::string> result;
    
    if (!nodes_.count(root_id)) {
        return result;
    }
    
    std::queue<std::pair<std::string, uint32_t>> to_visit;
    to_visit.push({root_id, 0});
    
    while (!to_visit.empty()) {
        auto [current_id, depth] = to_visit.front();
        to_visit.pop();
        
        result.push_back(current_id);
        
        if (depth < max_depth && nodes_.count(current_id)) {
            for (const auto& child_id : nodes_.at(current_id).children_ids) {
                to_visit.push({child_id, depth + 1});
            }
        }
    }
    
    return result;
}

} // namespace rros

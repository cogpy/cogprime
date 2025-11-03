#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>

namespace rros {

/**
 * @brief Hierarchical node in knowledge structure
 */
struct HierarchicalNode {
    std::string node_id;                           // Unique node identifier
    uint32_t level;                                // Hierarchy level (0 = root)
    std::string parent_id;                         // Parent node ID
    std::vector<std::string> children_ids;         // Child node IDs
    std::vector<float> data;                       // Node data/embedding
    float importance;                              // Node importance score
    std::unordered_map<std::string, float> attributes;  // Node attributes
    mutable uint64_t access_count;                 // Access frequency (mutable for stats)
    mutable std::chrono::high_resolution_clock::time_point last_access;  // Last access time (mutable)
};

/**
 * @brief Query result from hierarchical search
 */
struct HierarchicalQuery {
    std::vector<float> query_vector;               // Query embedding
    uint32_t max_results;                          // Maximum results to return
    uint32_t max_depth;                            // Maximum depth to search
    float similarity_threshold;                     // Minimum similarity
    bool include_ancestors;                        // Include parent nodes
    bool include_descendants;                      // Include child nodes
};

/**
 * @brief Search result from hierarchical query
 */
struct SearchResult {
    std::string node_id;
    float similarity_score;
    uint32_t depth;
    std::vector<float> data;
};

/**
 * @brief B-tree node for efficient indexing
 */
template<typename KeyType, typename ValueType>
struct BTreeNode {
    static constexpr size_t ORDER = 16;  // B-tree order
    
    std::vector<KeyType> keys;
    std::vector<ValueType> values;
    std::vector<std::shared_ptr<BTreeNode>> children;
    bool is_leaf;
    
    BTreeNode() : is_leaf(true) {}
};

/**
 * @brief Efficient B-tree for O(log n) operations
 */
template<typename KeyType, typename ValueType>
class BTree {
public:
    BTree() : root_(std::make_shared<BTreeNode<KeyType, ValueType>>()) {}
    
    /**
     * @brief Insert key-value pair
     * @param key Key to insert
     * @param value Value to insert
     */
    void insert(const KeyType& key, const ValueType& value);
    
    /**
     * @brief Search for a key
     * @param key Key to search for
     * @return Optional value if found
     */
    std::pair<bool, ValueType> search(const KeyType& key) const;
    
    /**
     * @brief Remove a key
     * @param key Key to remove
     * @return True if removed
     */
    bool remove(const KeyType& key);
    
    /**
     * @brief Range query
     * @param start Start key
     * @param end End key
     * @return Vector of matching values
     */
    std::vector<ValueType> range_query(const KeyType& start, const KeyType& end) const;
    
    /**
     * @brief Get tree height (for performance monitoring)
     * @return Height of tree
     */
    uint32_t get_height() const;

private:
    std::shared_ptr<BTreeNode<KeyType, ValueType>> root_;
    mutable std::mutex tree_mutex_;
    
    void split_child(std::shared_ptr<BTreeNode<KeyType, ValueType>> parent, 
                    size_t index);
    void insert_non_full(std::shared_ptr<BTreeNode<KeyType, ValueType>> node,
                        const KeyType& key, const ValueType& value);
};

/**
 * @brief Hierarchical Knowledge Organization System
 * 
 * Implements multi-level hierarchical knowledge structure with:
 * - O(log n) access patterns via B-tree indexing
 * - Hierarchical attention allocation
 * - Efficient ancestor/descendant queries
 * - Automatic organization and balancing
 * - Support for 10x+ knowledge base scaling
 */
class HierarchicalKnowledgeOrganizer {
public:
    /**
     * @brief Constructor initializes hierarchical system
     * @param config Configuration parameters
     */
    explicit HierarchicalKnowledgeOrganizer(
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~HierarchicalKnowledgeOrganizer();
    
    // Node Management
    
    /**
     * @brief Add node to hierarchy
     * @param node Node to add
     * @return True if added successfully
     */
    bool add_node(const HierarchicalNode& node);
    
    /**
     * @brief Remove node from hierarchy
     * @param node_id Node identifier
     * @return True if removed
     */
    bool remove_node(const std::string& node_id);
    
    /**
     * @brief Get node by ID
     * @param node_id Node identifier
     * @return Node if found
     */
    std::pair<bool, HierarchicalNode> get_node(const std::string& node_id) const;
    
    /**
     * @brief Update node data
     * @param node_id Node identifier
     * @param data New data
     */
    void update_node_data(const std::string& node_id, const std::vector<float>& data);
    
    /**
     * @brief Set node parent (reorganize hierarchy)
     * @param node_id Node identifier
     * @param parent_id New parent identifier
     */
    void set_parent(const std::string& node_id, const std::string& parent_id);
    
    // Hierarchical Queries (O(log n))
    
    /**
     * @brief Search hierarchy with query vector
     * @param query Query parameters
     * @return Search results
     */
    std::vector<SearchResult> search(const HierarchicalQuery& query) const;
    
    /**
     * @brief Get all ancestors of a node
     * @param node_id Node identifier
     * @return Vector of ancestor nodes (bottom-up)
     */
    std::vector<HierarchicalNode> get_ancestors(const std::string& node_id) const;
    
    /**
     * @brief Get all descendants of a node
     * @param node_id Node identifier
     * @param max_depth Maximum depth to traverse
     * @return Vector of descendant nodes
     */
    std::vector<HierarchicalNode> get_descendants(
        const std::string& node_id, 
        uint32_t max_depth = UINT32_MAX
    ) const;
    
    /**
     * @brief Get siblings of a node
     * @param node_id Node identifier
     * @return Vector of sibling nodes
     */
    std::vector<HierarchicalNode> get_siblings(const std::string& node_id) const;
    
    /**
     * @brief Find k-nearest neighbors in hierarchy
     * @param query_vector Query vector
     * @param k Number of neighbors
     * @param start_node_id Optional starting node
     * @return K nearest nodes
     */
    std::vector<SearchResult> knn_search(
        const std::vector<float>& query_vector,
        uint32_t k,
        const std::string& start_node_id = ""
    ) const;
    
    // Hierarchical Organization
    
    /**
     * @brief Automatically organize nodes into hierarchy
     * @param nodes Nodes to organize
     * @param max_children Maximum children per node
     * @return Root node ID
     */
    std::string auto_organize(
        const std::vector<HierarchicalNode>& nodes,
        uint32_t max_children = 10
    );
    
    /**
     * @brief Balance hierarchy for optimal performance
     * @return Number of reorganizations performed
     */
    uint32_t balance_hierarchy();
    
    /**
     * @brief Merge similar nodes in hierarchy
     * @param similarity_threshold Minimum similarity for merging
     * @return Number of nodes merged
     */
    uint32_t merge_similar_nodes(float similarity_threshold = 0.95f);
    
    /**
     * @brief Prune infrequently accessed nodes
     * @param access_threshold Minimum access count
     * @return Number of nodes pruned
     */
    uint32_t prune_infrequent_nodes(uint64_t access_threshold = 10);
    
    // Hierarchical Attention
    
    /**
     * @brief Allocate attention across hierarchy levels
     * @param total_attention Total attention budget
     * @return Attention allocation per level
     */
    std::unordered_map<uint32_t, float> allocate_hierarchical_attention(
        float total_attention
    ) const;
    
    /**
     * @brief Update node importance based on access patterns
     * @param node_id Node identifier
     * @param delta Importance delta
     */
    void update_importance(const std::string& node_id, float delta);
    
    /**
     * @brief Propagate importance up the hierarchy
     * @param node_id Starting node
     * @param decay Importance decay per level
     */
    void propagate_importance(const std::string& node_id, float decay = 0.9f);
    
    // Performance & Statistics
    
    /**
     * @brief Get hierarchy statistics
     * @return Statistics map
     */
    std::unordered_map<std::string, float> get_stats() const;
    
    /**
     * @brief Get average query time
     * @return Average time in microseconds
     */
    float get_avg_query_time() const;
    
    /**
     * @brief Get hierarchy depth
     * @return Maximum depth
     */
    uint32_t get_max_depth() const;
    
    /**
     * @brief Get node count at each level
     * @return Map of level to node count
     */
    std::unordered_map<uint32_t, uint32_t> get_level_distribution() const;
    
    /**
     * @brief Verify O(log n) performance
     * @return True if performance meets target
     */
    bool verify_performance() const;
    
    // Configuration
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset organizer state
     */
    void reset();

private:
    // Data structures
    std::unordered_map<std::string, HierarchicalNode> nodes_;
    BTree<std::string, std::string> node_index_;   // Fast node lookup
    std::unordered_map<uint32_t, std::vector<std::string>> level_index_;  // Nodes per level
    
    // Configuration
    uint32_t max_hierarchy_depth_;
    uint32_t max_children_per_node_;
    float attention_decay_rate_;
    
    // Synchronization
    mutable std::mutex nodes_mutex_;
    mutable std::mutex index_mutex_;
    
    // Performance tracking
    mutable std::atomic<uint64_t> total_queries_{0};
    mutable std::atomic<uint64_t> total_query_time_us_{0};
    std::atomic<uint32_t> max_observed_depth_{0};
    
    // Helper methods
    float calculate_similarity(const std::vector<float>& a, 
                              const std::vector<float>& b) const;
    void update_level_index(const std::string& node_id, uint32_t old_level, uint32_t new_level);
    void rebalance_subtree(const std::string& root_id);
    std::vector<std::string> collect_subtree(const std::string& root_id, uint32_t max_depth) const;
};

// Template implementations for BTree

template<typename KeyType, typename ValueType>
void BTree<KeyType, ValueType>::insert(const KeyType& key, const ValueType& value) {
    std::lock_guard<std::mutex> lock(tree_mutex_);
    
    if (root_->keys.size() >= BTreeNode<KeyType, ValueType>::ORDER - 1) {
        auto new_root = std::make_shared<BTreeNode<KeyType, ValueType>>();
        new_root->is_leaf = false;
        new_root->children.push_back(root_);
        split_child(new_root, 0);
        root_ = new_root;
    }
    
    insert_non_full(root_, key, value);
}

template<typename KeyType, typename ValueType>
std::pair<bool, ValueType> BTree<KeyType, ValueType>::search(const KeyType& key) const {
    std::lock_guard<std::mutex> lock(tree_mutex_);
    
    auto current = root_;
    
    while (current) {
        size_t i = 0;
        while (i < current->keys.size() && key > current->keys[i]) {
            ++i;
        }
        
        if (i < current->keys.size() && key == current->keys[i]) {
            return {true, current->values[i]};
        }
        
        if (current->is_leaf) {
            return {false, ValueType{}};
        }
        
        current = current->children[i];
    }
    
    return {false, ValueType{}};
}

template<typename KeyType, typename ValueType>
uint32_t BTree<KeyType, ValueType>::get_height() const {
    std::lock_guard<std::mutex> lock(tree_mutex_);
    
    uint32_t height = 0;
    auto current = root_;
    
    while (current && !current->is_leaf) {
        ++height;
        if (!current->children.empty()) {
            current = current->children[0];
        } else {
            break;
        }
    }
    
    return height;
}

template<typename KeyType, typename ValueType>
void BTree<KeyType, ValueType>::split_child(
    std::shared_ptr<BTreeNode<KeyType, ValueType>> parent,
    size_t index
) {
    auto child = parent->children[index];
    auto new_child = std::make_shared<BTreeNode<KeyType, ValueType>>();
    new_child->is_leaf = child->is_leaf;
    
    size_t mid = BTreeNode<KeyType, ValueType>::ORDER / 2;
    
    // Move half of keys/values to new child
    // Note: Using vector insert for clarity. For production, consider
    // pre-allocating capacity or using a different container (e.g., std::deque)
    // to avoid O(n) element shifting during splits.
    new_child->keys.assign(child->keys.begin() + mid, child->keys.end());
    new_child->values.assign(child->values.begin() + mid, child->values.end());
    
    if (!child->is_leaf) {
        new_child->children.assign(child->children.begin() + mid, child->children.end());
        child->children.resize(mid);
    }
    
    // Insert middle key into parent
    parent->keys.insert(parent->keys.begin() + index, child->keys[mid - 1]);
    parent->values.insert(parent->values.begin() + index, child->values[mid - 1]);
    parent->children.insert(parent->children.begin() + index + 1, new_child);
    
    // Shrink original child
    child->keys.resize(mid - 1);
    child->values.resize(mid - 1);
}

template<typename KeyType, typename ValueType>
void BTree<KeyType, ValueType>::insert_non_full(
    std::shared_ptr<BTreeNode<KeyType, ValueType>> node,
    const KeyType& key,
    const ValueType& value
) {
    int i = static_cast<int>(node->keys.size()) - 1;
    
    if (node->is_leaf) {
        node->keys.push_back(key);
        node->values.push_back(value);
        
        while (i >= 0 && key < node->keys[i]) {
            node->keys[i + 1] = node->keys[i];
            node->values[i + 1] = node->values[i];
            --i;
        }
        node->keys[i + 1] = key;
        node->values[i + 1] = value;
    } else {
        while (i >= 0 && key < node->keys[i]) {
            --i;
        }
        ++i;
        
        if (node->children[i]->keys.size() >= BTreeNode<KeyType, ValueType>::ORDER - 1) {
            split_child(node, i);
            if (key > node->keys[i]) {
                ++i;
            }
        }
        
        insert_non_full(node->children[i], key, value);
    }
}

} // namespace rros

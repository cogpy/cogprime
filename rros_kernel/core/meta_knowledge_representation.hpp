#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <deque>
#include <chrono>

namespace rros {

/**
 * @brief Meta-knowledge entry
 */
struct MetaKnowledgeEntry {
    std::string id;
    std::string type;               // Type of meta-knowledge
    std::vector<float> embedding;   // Vector representation
    std::string symbolic_form;      // Symbolic/textual representation
    float confidence;               // Confidence in knowledge
    float utility;                  // Utility score
    std::chrono::high_resolution_clock::time_point creation_time;
    std::chrono::high_resolution_clock::time_point last_access;
    uint64_t access_count;
    
    MetaKnowledgeEntry();
};

/**
 * @brief Knowledge graph node
 */
struct KnowledgeNode {
    std::string node_id;
    std::string node_type;
    std::vector<float> features;
    std::unordered_map<std::string, float> properties;
    std::vector<std::string> connected_nodes;  // IDs of connected nodes
    
    KnowledgeNode();
    explicit KnowledgeNode(const std::string& id);
};

/**
 * @brief Knowledge retrieval query
 */
struct KnowledgeQuery {
    std::string query_type;
    std::vector<float> query_vector;
    std::unordered_map<std::string, float> constraints;
    size_t max_results;
    
    KnowledgeQuery();
};

/**
 * @brief Meta-Knowledge Representation System
 * 
 * Represents and manages meta-cognitive knowledge including:
 * - Knowledge about cognitive processes
 * - Understanding of strategies and their effectiveness
 * - Insights about learning and adaptation
 * - Representations of self-understanding
 * 
 * Uses hybrid representation combining:
 * - Vector embeddings for similarity and retrieval
 * - Symbolic forms for reasoning and explanation
 * - Graph structures for relationships and dependencies
 */
class MetaKnowledgeRepresentation {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit MetaKnowledgeRepresentation(
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Destructor
     */
    ~MetaKnowledgeRepresentation();
    
    /**
     * @brief Store meta-knowledge
     * @param knowledge Knowledge to store
     * @param context Context information
     * @return Knowledge entry ID
     */
    std::string store_knowledge(
        const std::unordered_map<std::string, float>& knowledge,
        const std::string& context
    );
    
    /**
     * @brief Retrieve meta-knowledge based on query
     * @param query Query specification
     * @return Retrieved knowledge entries
     */
    std::vector<MetaKnowledgeEntry> retrieve_knowledge(
        const KnowledgeQuery& query
    );
    
    /**
     * @brief Update existing knowledge
     * @param entry_id ID of entry to update
     * @param updates Update data
     */
    void update_knowledge(
        const std::string& entry_id,
        const std::unordered_map<std::string, float>& updates
    );
    
    /**
     * @brief Create knowledge graph node
     * @param node Node to create
     * @return Node ID
     */
    std::string create_node(const KnowledgeNode& node);
    
    /**
     * @brief Create relationship between nodes
     * @param source_id Source node ID
     * @param target_id Target node ID
     * @param relation_type Type of relationship
     * @param strength Relationship strength
     */
    void create_relationship(
        const std::string& source_id,
        const std::string& target_id,
        const std::string& relation_type,
        float strength = 1.0f
    );
    
    /**
     * @brief Query knowledge graph
     * @param start_node Starting node ID
     * @param max_depth Maximum traversal depth
     * @return Connected subgraph
     */
    std::vector<KnowledgeNode> query_graph(
        const std::string& start_node,
        uint32_t max_depth = 2
    );
    
    /**
     * @brief Consolidate similar knowledge entries
     * @param similarity_threshold Threshold for consolidation
     * @return Number of entries consolidated
     */
    size_t consolidate_knowledge(float similarity_threshold = 0.85f);
    
    /**
     * @brief Generate knowledge summary
     * @param knowledge_type Type of knowledge to summarize
     * @return Summary representation
     */
    std::unordered_map<std::string, float> generate_summary(
        const std::string& knowledge_type
    );
    
    /**
     * @brief Compute knowledge coherence
     * @return Coherence score [0.0, 1.0]
     */
    float compute_coherence() const;
    
    /**
     * @brief Detect knowledge gaps
     * @param domain Domain to analyze
     * @return Identified gaps with severity scores
     */
    std::unordered_map<std::string, float> detect_gaps(
        const std::string& domain
    );
    
    /**
     * @brief Export knowledge in symbolic form
     * @param entry_ids IDs of entries to export (empty = all)
     * @return Symbolic representation
     */
    std::vector<std::string> export_symbolic(
        const std::vector<std::string>& entry_ids = {}
    );
    
    /**
     * @brief Get representation statistics
     * @return Current statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Reset representation system
     */
    void reset();

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Knowledge storage
    std::unordered_map<std::string, MetaKnowledgeEntry> knowledge_store_;
    
    // Knowledge graph
    std::unordered_map<std::string, KnowledgeNode> knowledge_graph_;
    
    // Relationship storage
    struct Relationship {
        std::string source;
        std::string target;
        std::string type;
        float strength;
    };
    std::vector<Relationship> relationships_;
    
    // Index for fast retrieval
    struct VectorIndex {
        std::vector<std::pair<std::string, std::vector<float>>> entries;
        
        std::vector<std::pair<std::string, float>> search(
            const std::vector<float>& query,
            size_t k
        ) const;
    } vector_index_;
    
    // Statistics
    uint64_t total_entries_;
    uint64_t total_retrievals_;
    uint64_t total_consolidations_;
    
    // Internal methods
    std::vector<float> encode_knowledge(
        const std::unordered_map<std::string, float>& knowledge
    );
    
    std::string generate_entry_id();
    
    float compute_similarity(
        const std::vector<float>& v1,
        const std::vector<float>& v2
    ) const;
    
    void update_vector_index(
        const std::string& entry_id,
        const std::vector<float>& embedding
    );
    
    void merge_entries(
        const std::string& target_id,
        const std::string& source_id
    );
    
    std::string encode_symbolic(
        const std::unordered_map<std::string, float>& knowledge,
        const std::string& context
    );
    
    void prune_old_entries();
};

} // namespace rros

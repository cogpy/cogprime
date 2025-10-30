#pragma once

#include "unified_representation.hpp"
#include "cross_modal_attention.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>

namespace rros {

// Forward declarations
struct Domain;
struct LearningTask;

/**
 * @brief Abstract concept mapping for generalization
 */
struct AbstractMapping {
    std::string source_domain;
    std::string target_domain;
    std::unordered_map<std::string, std::string> concept_mappings;
    float mapping_confidence;
    
    AbstractMapping();
};

/**
 * @brief Knowledge graph node
 */
struct KnowledgeNode {
    std::string node_id;
    UnifiedConcept concept;
    std::vector<std::string> edges;  // Connected node IDs
    std::unordered_map<std::string, float> edge_weights;
    
    KnowledgeNode();
};

/**
 * @brief Cross-Domain Integration Framework
 * 
 * Comprehensive framework for integrating cognitive processing
 * across different domains and modalities. Provides unified
 * representations, cross-modal reasoning, and knowledge transfer.
 */
class CrossDomainIntegrator {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit CrossDomainIntegrator(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor
     */
    ~CrossDomainIntegrator();
    
    /**
     * @brief Register a domain for integration
     * @param domain Domain to register
     */
    void register_domain(const Domain& domain);
    
    /**
     * @brief Create abstract concept mapping between domains
     * @param source_domain Source domain
     * @param target_domain Target domain
     * @return Abstract mapping
     */
    AbstractMapping create_abstract_mapping(
        const std::string& source_domain,
        const std::string& target_domain
    );
    
    /**
     * @brief Perform cross-domain reasoning
     * @param concepts Concepts from multiple domains
     * @param query Reasoning query
     * @return Reasoning result
     */
    UnifiedConcept cross_domain_reasoning(
        const std::vector<UnifiedConcept>& concepts,
        const std::vector<float>& query
    );
    
    /**
     * @brief Integrate knowledge across domains
     * @param domain_concepts Concepts indexed by domain
     * @return Integrated knowledge representation
     */
    UnifiedConcept integrate_cross_domain_knowledge(
        const std::unordered_map<std::string, std::vector<UnifiedConcept>>& domain_concepts
    );
    
    /**
     * @brief Translate knowledge from domain-specific to general
     * @param domain_knowledge Domain-specific knowledge
     * @param source_domain Source domain name
     * @return General knowledge representation
     */
    UnifiedConcept translate_to_general(
        const std::vector<float>& domain_knowledge,
        const std::string& source_domain
    );
    
    /**
     * @brief Translate general knowledge to domain-specific
     * @param general_knowledge General knowledge
     * @param target_domain Target domain name
     * @return Domain-specific representation
     */
    std::vector<float> translate_to_domain(
        const UnifiedConcept& general_knowledge,
        const std::string& target_domain
    );
    
    /**
     * @brief Add node to knowledge graph
     * @param node Knowledge node
     */
    void add_knowledge_node(const KnowledgeNode& node);
    
    /**
     * @brief Connect nodes in knowledge graph
     * @param node1_id First node ID
     * @param node2_id Second node ID
     * @param weight Connection weight
     */
    void connect_nodes(const std::string& node1_id, const std::string& node2_id, float weight);
    
    /**
     * @brief Query knowledge graph
     * @param query Query vector
     * @param max_results Maximum number of results
     * @return Relevant knowledge nodes
     */
    std::vector<KnowledgeNode> query_knowledge_graph(
        const std::vector<float>& query,
        size_t max_results = 10
    );
    
    /**
     * @brief Validate cross-domain knowledge consistency
     * @param concepts Concepts to validate
     * @return Consistency score [0.0, 1.0]
     */
    float validate_consistency(const std::vector<UnifiedConcept>& concepts);
    
    /**
     * @brief Adapt domain to new context
     * @param source_domain Source domain
     * @param target_context Target context features
     * @return Adaptation accuracy
     */
    float adapt_domain(
        const std::string& source_domain,
        const std::vector<float>& target_context
    );
    
    /**
     * @brief Get integration metrics
     */
    std::unordered_map<std::string, float> get_metrics() const;
    
    /**
     * @brief Get statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Update configuration
     */
    void update_config(const std::unordered_map<std::string, float>& new_config);
    
    /**
     * @brief Reset to initial state
     */
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mutex_;
    
    // Component systems
    std::unique_ptr<UnifiedRepresentation> unified_repr_;
    std::unique_ptr<CrossModalAttentionSystem> attention_system_;
    
    // Helper functions
    float compute_domain_similarity(const std::string& domain1, const std::string& domain2);
    std::vector<UnifiedConcept> align_domain_concepts(
        const std::vector<UnifiedConcept>& concepts,
        const std::string& target_domain
    );
    bool check_logical_consistency(const UnifiedConcept& c1, const UnifiedConcept& c2);
};

} // namespace rros

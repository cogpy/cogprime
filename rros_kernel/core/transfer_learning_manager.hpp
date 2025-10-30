#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>

namespace rros {

// Forward declarations
struct Domain;
struct LearningTask;

/**
 * @brief Knowledge representation for transfer
 */
struct TransferableKnowledge {
    std::vector<float> feature_weights;
    std::vector<float> domain_invariant_features;
    std::unordered_map<std::string, float> learned_patterns;
    float confidence;
    
    TransferableKnowledge();
};

/**
 * @brief Transfer Learning Manager for cross-domain knowledge transfer
 * 
 * Implements sophisticated transfer learning mechanisms that enable
 * knowledge transfer between domains while avoiding negative transfer.
 * Uses domain adaptation and feature alignment techniques.
 */
class TransferLearningManager {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit TransferLearningManager(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor
     */
    ~TransferLearningManager();
    
    /**
     * @brief Transfer knowledge from source to target domain
     * @param source_domain Source domain with existing knowledge
     * @param target_domain Target domain for transfer
     * @param task Task in target domain
     * @return Transfer effectiveness [0.0, 1.0]
     */
    float transfer(
        const Domain& source_domain,
        const Domain& target_domain,
        const LearningTask& task
    );
    
    /**
     * @brief Extract transferable knowledge from domain
     * @param domain Source domain
     * @param tasks Tasks in the domain
     * @return Transferable knowledge representation
     */
    TransferableKnowledge extract_knowledge(
        const Domain& domain,
        const std::vector<LearningTask>& tasks
    );
    
    /**
     * @brief Adapt knowledge to target domain
     * @param knowledge Source knowledge
     * @param target_domain Target domain
     * @return Adapted knowledge
     */
    TransferableKnowledge adapt_knowledge(
        const TransferableKnowledge& knowledge,
        const Domain& target_domain
    );
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset state
     */
    void reset();

private:
    std::unordered_map<std::string, float> config_;
    std::unordered_map<std::string, TransferableKnowledge> knowledge_base_;
    mutable std::mutex manager_mutex_;
    
    float align_feature_spaces(
        const std::vector<float>& source_features,
        const std::vector<float>& target_features
    ) const;
    
    std::vector<float> find_domain_invariant_features(
        const Domain& source,
        const Domain& target
    ) const;
};

} // namespace rros

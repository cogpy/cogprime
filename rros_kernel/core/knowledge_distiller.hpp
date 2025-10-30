#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>

namespace rros {

/**
 * @brief Knowledge Distiller for model compression
 * 
 * Implements knowledge distillation techniques to compress large
 * models into smaller ones while preserving performance.
 */
class KnowledgeDistiller {
public:
    explicit KnowledgeDistiller(const std::unordered_map<std::string, float>& config = {});
    ~KnowledgeDistiller();
    
    /**
     * @brief Distill knowledge from source to target model
     * @param source_model Source model parameters
     * @param target_size Target model size
     * @param preserve_performance Minimum performance to preserve
     * @return Distilled model parameters
     */
    std::vector<float> distill(
        const std::vector<float>& source_model,
        size_t target_size,
        float preserve_performance
    );
    
    void update_config(const std::unordered_map<std::string, float>& config);
    void reset();

private:
    std::unordered_map<std::string, float> config_;
    mutable std::mutex distiller_mutex_;
    
    std::vector<float> compute_importance_scores(
        const std::vector<float>& model
    ) const;
    
    std::vector<float> select_important_parameters(
        const std::vector<float>& model,
        const std::vector<float>& importance_scores,
        size_t target_size
    ) const;
};

} // namespace rros

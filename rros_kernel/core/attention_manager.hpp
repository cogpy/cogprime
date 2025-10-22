#pragma once

#include <vector>
#include <unordered_map>
#include <string>

namespace rros {

/**
 * @brief Attention allocation and management system
 * 
 * Implements attention allocation based on relevance and current cognitive state.
 * Manages attention budget and focus dynamics according to cognitive principles.
 */
class AttentionManager {
public:
    explicit AttentionManager(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Update attention allocation based on salience map
     * @param salience_map Current salience values
     * @return Attention weights for each element
     */
    std::vector<float> update(const std::vector<float>& salience_map);
    
    /**
     * @brief Allocate attention across multiple targets
     * @param relevance_scores Relevance scores for each target
     * @return Attention allocation weights
     */
    std::vector<float> allocate(const std::vector<float>& relevance_scores);
    
    /**
     * @brief Update configuration parameters
     * @param config New configuration values
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset attention manager to initial state
     */
    void reset();
    
    /**
     * @brief Get current attention budget utilization
     * @return Fraction of attention budget currently allocated [0.0, 1.0]
     */
    float get_utilization() const;

private:
    std::unordered_map<std::string, float> config_;
    float attention_budget_;
    float decay_rate_;
    std::vector<float> current_allocation_;
    
    // Internal methods
    void apply_decay();
    void normalize_allocation(std::vector<float>& allocation);
    float compute_competition(const std::vector<float>& scores);
};

} // namespace rros
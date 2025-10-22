#pragma once

#include <vector>
#include <unordered_map>
#include <string>

namespace rros {

/**
 * @brief Result of relevance processing
 */
struct RelevanceResult {
    std::vector<float> salience_map;     // Salience values for input elements
    float max_salience;                  // Maximum salience detected
    float average_salience;              // Average salience across input
    std::vector<size_t> relevant_indices; // Indices of highly relevant elements
    float confidence;                    // Confidence in relevance assessment
};

/**
 * @brief Core relevance realization engine implementing Vervaeke's framework
 * 
 * Processes input through multiple relevance modes to determine salience
 * and relevance according to current context and goals.
 */
class RelevanceEngine {
public:
    explicit RelevanceEngine(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Process input data to determine relevance
     * @param input Input data to process
     * @param context Current contextual information
     * @return Relevance processing result
     */
    RelevanceResult process(
        const std::vector<float>& input,
        const std::unordered_map<std::string, float>& context = {}
    );
    
    /**
     * @brief Update engine configuration
     * @param config New configuration parameters
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset engine to initial state
     */
    void reset();
    
    /**
     * @brief Get current relevance thresholds
     * @return Current threshold values for relevance detection
     */
    std::unordered_map<std::string, float> get_thresholds() const;

private:
    std::unordered_map<std::string, float> config_;
    std::vector<float> relevance_weights_;
    float relevance_threshold_;
    
    // Internal processing methods
    std::vector<float> compute_salience(const std::vector<float>& input);
    float evaluate_contextual_relevance(float base_relevance, 
                                       const std::unordered_map<std::string, float>& context);
    void normalize_salience(std::vector<float>& salience);
};

} // namespace rros
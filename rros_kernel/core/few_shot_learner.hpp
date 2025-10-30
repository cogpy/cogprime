#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>

namespace rros {

struct LearningTask;

/**
 * @brief Few-Shot Learner for learning from minimal examples
 * 
 * Implements meta-learning techniques that enable rapid learning
 * from very few examples using prototype-based learning and
 * metric learning approaches.
 */
class FewShotLearner {
public:
    explicit FewShotLearner(const std::unordered_map<std::string, float>& config = {});
    ~FewShotLearner();
    
    /**
     * @brief Learn from few examples
     * @param task Learning task with minimal examples
     * @param num_shots Number of examples per class
     * @return Learning accuracy [0.0, 1.0]
     */
    float learn(const LearningTask& task, uint32_t num_shots);
    
    /**
     * @brief Update configuration
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset state
     */
    void reset();

private:
    std::unordered_map<std::string, float> config_;
    std::vector<std::vector<float>> prototypes_;
    mutable std::mutex learner_mutex_;
    
    std::vector<std::vector<float>> compute_prototypes(
        const std::vector<std::vector<float>>& examples,
        const std::vector<float>& labels,
        uint32_t num_shots
    ) const;
    
    float compute_similarity(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) const;
};

} // namespace rros

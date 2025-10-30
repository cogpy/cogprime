#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>

namespace rros {

struct LearningTask;
struct LearningStrategy;

/**
 * @brief Adaptive Learning Strategy for dynamic strategy selection
 * 
 * Selects and adapts learning strategies based on task characteristics
 * and performance feedback, implementing meta-learning for optimization.
 */
class AdaptiveLearningStrategy {
public:
    explicit AdaptiveLearningStrategy(const std::unordered_map<std::string, float>& config = {});
    ~AdaptiveLearningStrategy();
    
    /**
     * @brief Select optimal learning strategy
     * @param task Current learning task
     * @param performance_feedback Recent performance metrics
     * @param difficulty Task difficulty estimate
     * @return Selected learning strategy
     */
    LearningStrategy select_strategy(
        const LearningTask& task,
        const std::unordered_map<std::string, float>& performance_feedback,
        float difficulty
    );
    
    void update_config(const std::unordered_map<std::string, float>& config);
    void reset();

private:
    std::unordered_map<std::string, float> config_;
    std::vector<LearningStrategy> available_strategies_;
    std::unordered_map<std::string, float> strategy_performance_;
    mutable std::mutex strategy_mutex_;
    
    void initialize_strategies();
    float evaluate_strategy_fit(
        const LearningStrategy& strategy,
        const LearningTask& task,
        float difficulty
    ) const;
};

} // namespace rros

#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>

namespace rros {

struct LearningTask;

/**
 * @brief Curriculum Optimizer for optimal learning sequences
 * 
 * Automatically optimizes the sequence of learning tasks to
 * maximize learning efficiency using difficulty progression.
 */
class CurriculumOptimizer {
public:
    explicit CurriculumOptimizer(const std::unordered_map<std::string, float>& config = {});
    ~CurriculumOptimizer();
    
    /**
     * @brief Optimize curriculum learning sequence
     * @param tasks Available learning tasks
     * @param learner_state Current learner capabilities
     * @return Optimal task sequence (task IDs)
     */
    std::vector<std::string> optimize(
        const std::vector<LearningTask>& tasks,
        const std::unordered_map<std::string, float>& learner_state
    );
    
    /**
     * @brief Evaluate curriculum quality
     * @param task_sequence Sequence of task IDs
     * @param tasks All available tasks
     * @return Quality score [0.0, 1.0]
     */
    float evaluate_curriculum(
        const std::vector<std::string>& task_sequence,
        const std::vector<LearningTask>& tasks
    ) const;
    
    void update_config(const std::unordered_map<std::string, float>& config);
    void reset();

private:
    std::unordered_map<std::string, float> config_;
    mutable std::mutex optimizer_mutex_;
    
    float compute_task_difficulty(const LearningTask& task) const;
    float compute_prerequisite_satisfaction(
        const LearningTask& task,
        const std::vector<std::string>& completed_tasks
    ) const;
};

} // namespace rros

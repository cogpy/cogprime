#include "curriculum_optimizer.hpp"
#include "meta_learning_engine.hpp"
#include <algorithm>
#include <cmath>

namespace rros {

CurriculumOptimizer::CurriculumOptimizer(
    const std::unordered_map<std::string, float>& config)
    : config_(config) {
    
    if (config_.find("difficulty_step") == config_.end()) {
        config_["difficulty_step"] = 0.2f;
    }
}

CurriculumOptimizer::~CurriculumOptimizer() = default;

std::vector<std::string> CurriculumOptimizer::optimize(
    const std::vector<LearningTask>& tasks,
    const std::unordered_map<std::string, float>& learner_state) {
    
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    
    if (tasks.empty()) {
        return {};
    }
    
    // Get learner's current capability level
    float capability = 0.3f; // Default beginner level
    auto cap_it = learner_state.find("capability_level");
    if (cap_it != learner_state.end()) {
        capability = cap_it->second;
    }
    
    // Sort tasks by difficulty and prerequisite satisfaction
    std::vector<std::pair<float, std::string>> scored_tasks;
    
    for (const auto& task : tasks) {
        float difficulty = compute_task_difficulty(task);
        
        // Score based on match to current capability
        float difficulty_match = 1.0f - std::abs(difficulty - capability);
        
        // Prefer slightly harder tasks (zone of proximal development)
        if (difficulty > capability && difficulty <= capability + config_["difficulty_step"]) {
            difficulty_match += 0.3f;
        }
        
        scored_tasks.emplace_back(difficulty_match, task.task_id);
    }
    
    // Sort by score (higher is better)
    std::sort(scored_tasks.begin(), scored_tasks.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract task IDs in optimal order
    std::vector<std::string> sequence;
    for (const auto& [score, task_id] : scored_tasks) {
        sequence.push_back(task_id);
    }
    
    return sequence;
}

float CurriculumOptimizer::evaluate_curriculum(
    const std::vector<std::string>& task_sequence,
    const std::vector<LearningTask>& tasks) const {
    
    if (task_sequence.empty() || tasks.empty()) {
        return 0.0f;
    }
    
    // Build task lookup
    std::unordered_map<std::string, LearningTask> task_map;
    for (const auto& task : tasks) {
        task_map[task.task_id] = task;
    }
    
    // Evaluate difficulty progression
    float quality = 0.0f;
    float prev_difficulty = 0.0f;
    
    for (const auto& task_id : task_sequence) {
        auto it = task_map.find(task_id);
        if (it == task_map.end()) {
            continue;
        }
        
        float difficulty = compute_task_difficulty(it->second);
        
        // Reward gradual difficulty increase
        if (difficulty >= prev_difficulty && difficulty <= prev_difficulty + 0.3f) {
            quality += 1.0f;
        } else if (difficulty < prev_difficulty) {
            quality += 0.3f; // Penalty for going backwards
        }
        
        prev_difficulty = difficulty;
    }
    
    return quality / static_cast<float>(task_sequence.size());
}

void CurriculumOptimizer::update_config(
    const std::unordered_map<std::string, float>& config) {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
}

void CurriculumOptimizer::reset() {
    std::lock_guard<std::mutex> lock(optimizer_mutex_);
    // Nothing to reset currently
}

float CurriculumOptimizer::compute_task_difficulty(const LearningTask& task) const {
    // Use task's difficulty if available
    if (task.difficulty > 0.0f) {
        return task.difficulty;
    }
    
    // Estimate from characteristics
    float difficulty = 0.5f;
    
    auto complexity_it = task.task_characteristics.find("complexity");
    if (complexity_it != task.task_characteristics.end()) {
        difficulty = 0.7f * difficulty + 0.3f * complexity_it->second;
    }
    
    // Factor in number of training examples (fewer = harder)
    if (!task.training_examples.empty()) {
        float data_factor = 1.0f - std::min(1.0f, 
            static_cast<float>(task.training_examples.size()) / 100.0f);
        difficulty = 0.6f * difficulty + 0.4f * data_factor;
    }
    
    return std::max(0.0f, std::min(1.0f, difficulty));
}

float CurriculumOptimizer::compute_prerequisite_satisfaction(
    const LearningTask& task,
    const std::vector<std::string>& completed_tasks) const {
    
    // Check if task has prerequisites in characteristics
    auto prereq_it = task.task_characteristics.find("prerequisites");
    if (prereq_it == task.task_characteristics.end()) {
        return 1.0f; // No prerequisites
    }
    
    // Simplified: assume satisfaction based on number of completed tasks
    float satisfaction = std::min(1.0f, 
        static_cast<float>(completed_tasks.size()) * 0.1f);
    
    return satisfaction;
}

} // namespace rros

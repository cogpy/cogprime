#include "enhanced_goal_hierarchy.hpp"
#include <algorithm>
#include <cmath>

namespace rros {

EnhancedGoalHierarchy::EnhancedGoalHierarchy() {}

bool EnhancedGoalHierarchy::add_goal(const Goal& goal) {
    // Check if goal already exists
    if (goals_.find(goal.name) != goals_.end()) {
        return false;
    }
    
    // Add to goals map
    goals_[goal.name] = goal;
    
    // Update parent's subgoal list if parent exists
    if (!goal.parent_goal.empty() && goals_.find(goal.parent_goal) != goals_.end()) {
        goals_[goal.parent_goal].subgoals.push_back(goal.name);
    }
    
    // Add to active goals
    active_goals_.push_back(goal.name);
    
    return true;
}

bool EnhancedGoalHierarchy::update_goal_progress(const std::string& goal_name, float progress) {
    auto it = goals_.find(goal_name);
    if (it == goals_.end()) {
        return false;
    }
    
    it->second.completion_status = std::min(1.0f, std::max(0.0f, progress));
    
    // Mark as completed if progress reaches 1.0
    if (it->second.completion_status >= 1.0f) {
        complete_goal(goal_name);
    }
    
    // Propagate to parent
    propagate_progress_to_parent(goal_name);
    
    return true;
}

void EnhancedGoalHierarchy::complete_goal(const std::string& goal_name) {
    auto it = goals_.find(goal_name);
    if (it == goals_.end()) {
        return;
    }
    
    // Remove from active goals
    auto active_it = std::find(active_goals_.begin(), active_goals_.end(), goal_name);
    if (active_it != active_goals_.end()) {
        active_goals_.erase(active_it);
        completed_goals_.push_back(goal_name);
    }
    
    // Mark as completed
    it->second.completion_status = 1.0f;
    
    // Propagate to parent
    propagate_progress_to_parent(goal_name);
}

std::vector<std::string> EnhancedGoalHierarchy::get_sorted_active_goals() const {
    std::vector<std::pair<float, std::string>> prioritized;
    prioritized.reserve(active_goals_.size());
    
    for (const auto& goal_name : active_goals_) {
        float priority = calculate_dynamic_priority(goal_name);
        prioritized.push_back({priority, goal_name});
    }
    
    // Sort by priority (descending)
    std::sort(prioritized.begin(), prioritized.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<std::string> result;
    result.reserve(prioritized.size());
    for (const auto& p : prioritized) {
        result.push_back(p.second);
    }
    
    return result;
}

std::vector<std::string> EnhancedGoalHierarchy::get_next_required_actions(size_t top_k) const {
    auto sorted_goals = get_sorted_active_goals();
    
    std::unordered_map<std::string, bool> unique_actions;
    std::vector<std::string> result;
    
    size_t goals_to_consider = std::min(top_k, sorted_goals.size());
    for (size_t i = 0; i < goals_to_consider; ++i) {
        const auto& goal = goals_.at(sorted_goals[i]);
        for (const auto& action : goal.required_actions) {
            if (unique_actions.find(action) == unique_actions.end()) {
                unique_actions[action] = true;
                result.push_back(action);
            }
        }
    }
    
    return result;
}

float EnhancedGoalHierarchy::calculate_dynamic_priority(const std::string& goal_name) const {
    auto it = goals_.find(goal_name);
    if (it == goals_.end()) {
        return 0.0f;
    }
    
    const Goal& goal = it->second;
    float dynamic_priority = goal.priority;
    
    // Factor in urgency (deadline proximity)
    float urgency = calculate_urgency(goal);
    dynamic_priority += urgency * URGENCY_SCALING_FACTOR;  // Urgency can significantly boost priority
    
    // Factor in parent goal importance
    if (!goal.parent_goal.empty()) {
        auto parent_it = goals_.find(goal.parent_goal);
        if (parent_it != goals_.end()) {
            dynamic_priority += parent_it->second.priority * 0.3f;
        }
    }
    
    // Penalize for low completion (encourage finishing started tasks)
    if (goal.completion_status > 0.0f && goal.completion_status < 1.0f) {
        dynamic_priority += 0.5f;  // Bonus for partially completed goals
    }
    
    return dynamic_priority;
}

const Goal& EnhancedGoalHierarchy::get_goal(const std::string& goal_name) const {
    return goals_.at(goal_name);
}

std::unordered_map<std::string, float> EnhancedGoalHierarchy::get_planning_stats() const {
    std::unordered_map<std::string, float> stats;
    
    stats["total_goals"] = static_cast<float>(goals_.size());
    stats["active_goals"] = static_cast<float>(active_goals_.size());
    stats["completed_goals"] = static_cast<float>(completed_goals_.size());
    
    if (!goals_.empty()) {
        float avg_priority = 0.0f;
        float avg_completion = 0.0f;
        size_t deadlined_goals = 0;
        
        for (const auto& [name, goal] : goals_) {
            avg_priority += goal.priority;
            avg_completion += goal.completion_status;
            if (goal.deadline.has_value()) {
                deadlined_goals++;
            }
        }
        
        stats["avg_priority"] = avg_priority / goals_.size();
        stats["avg_completion"] = avg_completion / goals_.size();
        stats["deadlined_goals"] = static_cast<float>(deadlined_goals);
    }
    
    // Calculate max hierarchy depth
    size_t max_depth = 0;
    for (const auto& [name, _] : goals_) {
        max_depth = std::max(max_depth, get_goal_depth(name));
    }
    stats["max_hierarchy_depth"] = static_cast<float>(max_depth);
    
    return stats;
}

bool EnhancedGoalHierarchy::is_deadline_approaching(
    const std::string& goal_name, float threshold_seconds) const {
    
    auto it = goals_.find(goal_name);
    if (it == goals_.end() || !it->second.deadline.has_value()) {
        return false;
    }
    
    auto now = std::chrono::system_clock::now();
    auto deadline = it->second.deadline.value();
    auto time_remaining = std::chrono::duration_cast<std::chrono::seconds>(deadline - now).count();
    
    return time_remaining > 0 && time_remaining < threshold_seconds;
}

void EnhancedGoalHierarchy::clear() {
    goals_.clear();
    active_goals_.clear();
    completed_goals_.clear();
}

// Private helper functions

float EnhancedGoalHierarchy::calculate_urgency(const Goal& goal) const {
    if (!goal.deadline.has_value()) {
        return 0.0f;
    }
    
    auto now = std::chrono::system_clock::now();
    auto deadline = goal.deadline.value();
    
    // Calculate time remaining in seconds
    auto time_remaining = std::chrono::duration_cast<std::chrono::seconds>(deadline - now).count();
    
    if (time_remaining <= 0) {
        return 1.0f;  // Maximum urgency if deadline passed
    }
    
    // Urgency increases as deadline approaches
    // Using inverse function: urgency = k / (1 + time_remaining)
    float urgency = 2.0f / (1.0f + static_cast<float>(time_remaining));
    
    return std::min(1.0f, urgency);
}

size_t EnhancedGoalHierarchy::get_goal_depth(const std::string& goal_name) const {
    auto it = goals_.find(goal_name);
    if (it == goals_.end()) {
        return 0;
    }
    
    size_t depth = 0;
    std::string current = goal_name;
    
    // Traverse up the hierarchy with cycle detection
    while (!goals_.at(current).parent_goal.empty()) {
        depth++;
        current = goals_.at(current).parent_goal;
        
        // Prevent infinite loops from circular dependencies
        if (depth > MAX_HIERARCHY_DEPTH) {
            break;
        }
    }
    
    return depth;
}

void EnhancedGoalHierarchy::propagate_progress_to_parent(const std::string& goal_name) {
    auto it = goals_.find(goal_name);
    if (it == goals_.end() || it->second.parent_goal.empty()) {
        return;
    }
    
    const std::string& parent_name = it->second.parent_goal;
    auto parent_it = goals_.find(parent_name);
    if (parent_it == goals_.end()) {
        return;
    }
    
    // Calculate parent progress based on subgoals
    const auto& parent = parent_it->second;
    if (parent.subgoals.empty()) {
        return;
    }
    
    float total_progress = 0.0f;
    for (const auto& subgoal_name : parent.subgoals) {
        auto subgoal_it = goals_.find(subgoal_name);
        if (subgoal_it != goals_.end()) {
            total_progress += subgoal_it->second.completion_status;
        }
    }
    
    float new_parent_progress = total_progress / parent.subgoals.size();
    parent_it->second.completion_status = new_parent_progress;
    
    // Check if parent should be marked complete
    if (new_parent_progress >= 1.0f) {
        complete_goal(parent_name);
    }
}

// DynamicResourceAllocator implementation

DynamicResourceAllocator::DynamicResourceAllocator(float total_resources)
    : total_resources_(total_resources),
      available_resources_(total_resources) {}

bool DynamicResourceAllocator::allocate(const std::string& action_name, float amount) {
    if (amount <= available_resources_) {
        allocations_[action_name] = amount;
        available_resources_ -= amount;
        allocation_history_.push_back(allocations_);
        return true;
    }
    return false;
}

void DynamicResourceAllocator::release(const std::string& action_name) {
    auto it = allocations_.find(action_name);
    if (it != allocations_.end()) {
        available_resources_ += it->second;
        allocations_.erase(it);
    }
}

float DynamicResourceAllocator::get_utilization() const {
    if (total_resources_ <= 0.0f) {
        return 0.0f;
    }
    return 1.0f - (available_resources_ / total_resources_);
}

std::unordered_map<std::string, float> DynamicResourceAllocator::optimize_allocation(
    const std::unordered_map<std::string, float>& action_priorities) const {
    
    std::unordered_map<std::string, float> optimal_allocation;
    
    if (action_priorities.empty()) {
        return optimal_allocation;
    }
    
    // Calculate total priority
    float total_priority = 0.0f;
    for (const auto& [action, priority] : action_priorities) {
        total_priority += priority;
    }
    
    if (total_priority <= 0.0f) {
        return optimal_allocation;
    }
    
    // Allocate proportionally to priorities
    for (const auto& [action, priority] : action_priorities) {
        float allocation = (priority / total_priority) * available_resources_;
        optimal_allocation[action] = allocation;
    }
    
    return optimal_allocation;
}

void DynamicResourceAllocator::reset() {
    available_resources_ = total_resources_;
    allocations_.clear();
    allocation_history_.clear();
}

} // namespace rros

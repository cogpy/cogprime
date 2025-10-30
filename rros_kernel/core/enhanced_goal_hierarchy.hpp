#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <optional>

namespace rros {

/**
 * @brief Goal structure with hierarchical relationships and temporal constraints
 */
struct Goal {
    std::string name;
    std::string description;
    float priority;                                    // Base priority [0, 1]
    std::optional<std::chrono::system_clock::time_point> deadline;
    std::string parent_goal;                           // Empty if root goal
    std::vector<std::string> subgoals;
    float completion_status;                           // [0, 1]
    std::vector<std::string> required_actions;
    std::chrono::system_clock::time_point creation_time;
    
    // Enhanced planning fields
    float estimated_cost;                              // Resource cost estimate
    float urgency_factor;                              // Dynamic urgency [0, 1]
    std::unordered_map<std::string, float> dependencies; // Goal name -> strength
};

/**
 * @brief Enhanced Goal Hierarchy Manager
 * 
 * Implements multi-level goal hierarchies with temporal constraints:
 * - Hierarchical goal decomposition
 * - Deadline-aware prioritization
 * - Dynamic urgency calculation
 * - Dependency tracking
 * - Action extraction and planning
 * 
 * Handles complex planning scenarios with real-time constraints
 */
class EnhancedGoalHierarchy {
public:
    EnhancedGoalHierarchy();
    
    /**
     * @brief Add a goal to the hierarchy
     * @param goal Goal to add
     * @return True if successfully added
     */
    bool add_goal(const Goal& goal);
    
    /**
     * @brief Update goal progress
     * @param goal_name Name of the goal
     * @param progress Progress value [0, 1]
     * @return True if update successful
     */
    bool update_goal_progress(const std::string& goal_name, float progress);
    
    /**
     * @brief Mark goal as completed
     * @param goal_name Name of the goal
     */
    void complete_goal(const std::string& goal_name);
    
    /**
     * @brief Get active goals sorted by priority and urgency
     * @return Vector of goal names sorted by computed priority
     */
    std::vector<std::string> get_sorted_active_goals() const;
    
    /**
     * @brief Get next required actions from top priority goals
     * @param top_k Number of top goals to consider
     * @return Vector of unique required actions
     */
    std::vector<std::string> get_next_required_actions(size_t top_k = 3) const;
    
    /**
     * @brief Calculate dynamic priority for a goal
     * 
     * Considers:
     * - Base priority
     * - Deadline proximity
     * - Parent goal importance
     * - Completion progress
     * 
     * @param goal_name Name of the goal
     * @return Computed dynamic priority
     */
    float calculate_dynamic_priority(const std::string& goal_name) const;

    
    /**
     * @brief Get goal by name
     * @param goal_name Name of the goal
     * @return Const reference to goal
     */
    const Goal& get_goal(const std::string& goal_name) const;
    
    /**
     * @brief Get all active goal names
     * @return Vector of active goal names
     */
    std::vector<std::string> get_active_goals() const { return active_goals_; }
    
    /**
     * @brief Get all completed goal names
     * @return Vector of completed goal names
     */
    std::vector<std::string> get_completed_goals() const { return completed_goals_; }
    
    /**
     * @brief Get planning statistics
     * @return Map of statistics
     */
    std::unordered_map<std::string, float> get_planning_stats() const;
    
    /**
     * @brief Check if deadline is approaching for a goal
     * @param goal_name Name of the goal
     * @param threshold_seconds Time threshold in seconds
     * @return True if deadline within threshold
     */
    bool is_deadline_approaching(const std::string& goal_name, float threshold_seconds = 3600.0f) const;
    
    /**
     * @brief Clear all goals
     */
    void clear();

private:
    static constexpr float URGENCY_SCALING_FACTOR = 2.0f;
    static constexpr size_t MAX_HIERARCHY_DEPTH = 100;
    
    std::unordered_map<std::string, Goal> goals_;
    std::vector<std::string> active_goals_;
    std::vector<std::string> completed_goals_;
    
    /**
     * @brief Calculate urgency based on deadline
     */
    float calculate_urgency(const Goal& goal) const;
    
    /**
     * @brief Get hierarchy depth of a goal
     */
    size_t get_goal_depth(const std::string& goal_name) const;
    
    /**
     * @brief Propagate progress to parent goals
     */
    void propagate_progress_to_parent(const std::string& goal_name);
};

/**
 * @brief Dynamic Resource Allocator for action execution
 * 
 * Manages resource allocation with optimization
 */
class DynamicResourceAllocator {
public:
    explicit DynamicResourceAllocator(float total_resources = 100.0f);
    
    /**
     * @brief Allocate resources for an action
     * @param action_name Name of the action
     * @param amount Amount to allocate
     * @return True if allocation successful
     */
    bool allocate(const std::string& action_name, float amount);
    
    /**
     * @brief Release allocated resources
     * @param action_name Name of the action
     */
    void release(const std::string& action_name);
    
    /**
     * @brief Get current utilization rate
     * @return Utilization [0, 1]
     */
    float get_utilization() const;
    
    /**
     * @brief Optimize resource allocation based on priorities
     * @param action_priorities Map of action names to priorities
     * @return Optimal allocation map
     */
    std::unordered_map<std::string, float> optimize_allocation(
        const std::unordered_map<std::string, float>& action_priorities) const;
    
    /**
     * @brief Get available resources
     * @return Available resource amount
     */
    float get_available_resources() const { return available_resources_; }
    
    /**
     * @brief Reset allocator
     */
    void reset();

private:
    float total_resources_;
    float available_resources_;
    std::unordered_map<std::string, float> allocations_;
    std::vector<std::unordered_map<std::string, float>> allocation_history_;
};

} // namespace rros

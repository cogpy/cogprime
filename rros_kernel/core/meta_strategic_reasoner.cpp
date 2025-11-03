#include "meta_strategic_reasoner.hpp"
#include "strategy_effectiveness_evaluator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace rros {

// StrategicPlan implementation
StrategicPlan::StrategicPlan()
    : confidence(0.0f), priority(0.5f) {}

// StrategicContext implementation
StrategicContext::StrategicContext() {}

// StrategyPerformanceMetrics implementation
StrategyPerformanceMetrics::StrategyPerformanceMetrics()
    : effectiveness_score(0.0f)
    , adaptation_rate(0.0f)
    , generalization_ability(0.0f)
    , robustness(0.0f)
    , efficiency(0.0f)
    , execution_count(0)
    , last_execution(std::chrono::high_resolution_clock::now()) {}

// MetaStrategicReasoner implementation
MetaStrategicReasoner::MetaStrategicReasoner(
    const std::unordered_map<std::string, float>& config)
    : config_(config)
    , max_experience_history_(1000)
    , total_strategic_decisions_(0)
    , successful_adaptations_(0) {
    
    // Set default configuration
    if (config_.find("exploration_rate") == config_.end()) {
        config_["exploration_rate"] = 0.2f;
    }
    if (config_.find("learning_rate") == config_.end()) {
        config_["learning_rate"] = 0.1f;
    }
    if (config_.find("planning_horizon") == config_.end()) {
        config_["planning_horizon"] = 10.0f;
    }
    if (config_.find("effectiveness_threshold") == config_.end()) {
        config_["effectiveness_threshold"] = 0.6f;
    }
    
    // Initialize planning state
    planning_state_.exploration_rate = config_["exploration_rate"];
    planning_state_.exploitation_rate = 1.0f - config_["exploration_rate"];
    planning_state_.total_plans_generated = 0;
    planning_state_.successful_plans = 0;
    
    initialize_subsystems();
}

MetaStrategicReasoner::~MetaStrategicReasoner() {}

void MetaStrategicReasoner::initialize_subsystems() {
    // Initialize effectiveness evaluator
    effectiveness_evaluator_ = std::make_unique<StrategyEffectivenessEvaluator>(config_);
    
    // StrategyTransferEngine and StrategicKnowledgeBase are optional
    // and not yet implemented, so we leave them as nullptr
}

StrategicPlan MetaStrategicReasoner::generate_strategic_plan(
    const StrategicContext& context,
    float planning_horizon) {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    StrategicPlan plan;
    plan.plan_id = "plan_" + std::to_string(planning_state_.total_plans_generated);
    
    // Generate strategy sequence based on context
    size_t max_length = static_cast<size_t>(planning_horizon / 2.0f);
    plan.strategy_sequence = generate_strategy_sequence(context, max_length);
    
    // Estimate expected outcomes based on historical data
    for (const auto& strategy : plan.strategy_sequence) {
        auto it = strategy_metrics_.find(strategy);
        if (it != strategy_metrics_.end()) {
            plan.expected_outcomes[strategy] = it->second.effectiveness_score;
        } else {
            plan.expected_outcomes[strategy] = 0.5f; // Default expectation
        }
    }
    
    // Estimate resource requirements
    plan.resource_requirements["time"] = static_cast<float>(plan.strategy_sequence.size());
    plan.resource_requirements["memory"] = static_cast<float>(plan.strategy_sequence.size()) * 0.1f;
    plan.resource_requirements["computation"] = static_cast<float>(plan.strategy_sequence.size()) * 0.5f;
    
    // Compute plan confidence based on historical success
    plan.confidence = compute_plan_alignment(plan, context);
    
    // Set priority based on goal alignment
    float goal_alignment = 0.0f;
    if (!context.goals.empty()) {
        for (const auto& [goal, value] : context.goals) {
            goal_alignment += value;
        }
        goal_alignment /= static_cast<float>(context.goals.size());
    }
    plan.priority = goal_alignment;
    
    planning_state_.total_plans_generated++;
    
    return plan;
}

StrategyPerformanceMetrics MetaStrategicReasoner::evaluate_plan_effectiveness(
    const StrategicPlan& plan,
    const StrategicContext& context) {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    StrategyPerformanceMetrics metrics;
    
    // Aggregate metrics from individual strategies in the plan
    float total_effectiveness = 0.0f;
    float total_robustness = 0.0f;
    float total_efficiency = 0.0f;
    size_t count = 0;
    
    for (const auto& strategy : plan.strategy_sequence) {
        auto it = strategy_metrics_.find(strategy);
        if (it != strategy_metrics_.end()) {
            total_effectiveness += it->second.effectiveness_score;
            total_robustness += it->second.robustness;
            total_efficiency += it->second.efficiency;
            count++;
        }
    }
    
    if (count > 0) {
        metrics.effectiveness_score = total_effectiveness / static_cast<float>(count);
        metrics.robustness = total_robustness / static_cast<float>(count);
        metrics.efficiency = total_efficiency / static_cast<float>(count);
    } else {
        metrics.effectiveness_score = 0.5f;
        metrics.robustness = 0.5f;
        metrics.efficiency = 0.5f;
    }
    
    // Compute adaptation rate based on plan diversity
    metrics.adaptation_rate = std::min(1.0f, 
        static_cast<float>(plan.strategy_sequence.size()) / 10.0f);
    
    // Estimate generalization from context similarity to past experiences
    auto similar_exp = find_similar_experiences(context, 5);
    if (!similar_exp.empty()) {
        float total_gen = 0.0f;
        for (const auto& exp : similar_exp) {
            total_gen += exp.success_score;
        }
        metrics.generalization_ability = total_gen / static_cast<float>(similar_exp.size());
    } else {
        metrics.generalization_ability = 0.5f;
    }
    
    metrics.execution_count = 1;
    metrics.last_execution = std::chrono::high_resolution_clock::now();
    
    return metrics;
}

StrategicPlan MetaStrategicReasoner::adapt_strategic_approach(
    const StrategicPlan& current_plan,
    const std::unordered_map<std::string, float>& performance_feedback,
    const StrategicContext& context) {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    StrategicPlan adapted_plan = current_plan;
    adapted_plan.plan_id = "adapted_" + current_plan.plan_id;
    
    // Compute average performance from feedback
    float avg_performance = 0.0f;
    if (!performance_feedback.empty()) {
        for (const auto& [key, value] : performance_feedback) {
            avg_performance += value;
        }
        avg_performance /= static_cast<float>(performance_feedback.size());
    }
    
    // If performance is below threshold, adapt the strategy sequence
    if (avg_performance < config_["effectiveness_threshold"]) {
        // Remove poorly performing strategies
        std::vector<std::string> new_sequence;
        for (const auto& strategy : current_plan.strategy_sequence) {
            auto it = strategy_metrics_.find(strategy);
            if (it != strategy_metrics_.end() && 
                it->second.effectiveness_score >= config_["effectiveness_threshold"]) {
                new_sequence.push_back(strategy);
            }
        }
        
        // Add new strategies based on context
        auto additional_strategies = generate_strategy_sequence(context, 3);
        new_sequence.insert(new_sequence.end(), 
                          additional_strategies.begin(), 
                          additional_strategies.end());
        
        adapted_plan.strategy_sequence = new_sequence;
        successful_adaptations_++;
    }
    
    // Update expected outcomes based on feedback
    for (const auto& [key, value] : performance_feedback) {
        adapted_plan.expected_outcomes[key] = value;
    }
    
    // Adjust confidence based on adaptation success
    float learning_rate = config_["learning_rate"];
    adapted_plan.confidence = current_plan.confidence * (1.0f - learning_rate) + 
                             avg_performance * learning_rate;
    
    return adapted_plan;
}

std::vector<std::string> MetaStrategicReasoner::optimize_strategy_selection(
    const std::vector<std::string>& available_strategies,
    const StrategicContext& context) {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    // Score each strategy based on context
    std::vector<std::pair<std::string, float>> scored_strategies;
    
    for (const auto& strategy : available_strategies) {
        float score = predict_strategy_performance(strategy, context);
        scored_strategies.emplace_back(strategy, score);
    }
    
    // Sort by score descending
    std::sort(scored_strategies.begin(), scored_strategies.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Select top strategies (up to 5)
    std::vector<std::string> optimized_sequence;
    size_t max_select = std::min(size_t(5), scored_strategies.size());
    for (size_t i = 0; i < max_select; ++i) {
        optimized_sequence.push_back(scored_strategies[i].first);
    }
    
    total_strategic_decisions_++;
    
    return optimized_sequence;
}

float MetaStrategicReasoner::transfer_strategic_knowledge(
    const std::string& source_domain,
    const std::string& target_domain,
    const std::vector<std::string>& strategies) {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    // Estimate transfer effectiveness based on domain similarity
    float transfer_effectiveness = estimate_transfer_effectiveness(source_domain, target_domain);
    
    // Transfer strategy metrics with reduced confidence
    for (const auto& strategy : strategies) {
        auto it = strategy_metrics_.find(strategy);
        if (it != strategy_metrics_.end()) {
            std::string target_strategy = target_domain + "::" + strategy;
            strategy_metrics_[target_strategy] = it->second;
            // Reduce confidence due to domain transfer
            strategy_metrics_[target_strategy].effectiveness_score *= transfer_effectiveness;
            strategy_metrics_[target_strategy].execution_count = 0;
        }
    }
    
    return transfer_effectiveness;
}

void MetaStrategicReasoner::learn_from_experience(
    const StrategicPlan& plan,
    const std::unordered_map<std::string, float>& actual_outcomes,
    const StrategicContext& context) {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    // Compute success score
    float success_score = 0.0f;
    if (!actual_outcomes.empty()) {
        for (const auto& [key, value] : actual_outcomes) {
            success_score += value;
        }
        success_score /= static_cast<float>(actual_outcomes.size());
    }
    
    // Update strategy metrics for each strategy in the plan
    for (const auto& strategy : plan.strategy_sequence) {
        update_strategy_metrics(strategy, success_score, context);
    }
    
    // Store experience
    StrategicExperience experience;
    experience.plan = plan;
    experience.context = context;
    experience.outcomes = actual_outcomes;
    experience.success_score = success_score;
    experience.timestamp = std::chrono::high_resolution_clock::now();
    
    experience_history_.push_back(experience);
    
    // Prune if needed
    if (experience_history_.size() > max_experience_history_) {
        prune_experience_history();
    }
    
    // Update planning state
    if (success_score >= config_["effectiveness_threshold"]) {
        planning_state_.successful_plans++;
    }
}

float MetaStrategicReasoner::predict_strategy_performance(
    const std::string& strategy_name,
    const StrategicContext& context) {
    
    // Check if we have metrics for this strategy
    auto it = strategy_metrics_.find(strategy_name);
    if (it != strategy_metrics_.end()) {
        // Use historical effectiveness as baseline
        float base_performance = it->second.effectiveness_score;
        
        // Adjust based on context similarity to past successes
        auto similar_exp = find_similar_experiences(context, 3);
        if (!similar_exp.empty()) {
            float context_adjustment = 0.0f;
            for (const auto& exp : similar_exp) {
                // Check if this strategy was in the successful plan
                auto& seq = exp.plan.strategy_sequence;
                if (std::find(seq.begin(), seq.end(), strategy_name) != seq.end()) {
                    context_adjustment += exp.success_score;
                }
            }
            context_adjustment /= static_cast<float>(similar_exp.size());
            
            // Blend historical and contextual predictions
            return 0.7f * base_performance + 0.3f * context_adjustment;
        }
        
        return base_performance;
    }
    
    // No historical data - return neutral prediction
    return 0.5f;
}

std::unordered_map<std::string, float> MetaStrategicReasoner::get_strategic_insights(
    const std::string& domain) {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    std::unordered_map<std::string, float> insights;
    
    // Overall success rate
    if (planning_state_.total_plans_generated > 0) {
        insights["overall_success_rate"] = 
            static_cast<float>(planning_state_.successful_plans) / 
            static_cast<float>(planning_state_.total_plans_generated);
    }
    
    // Average strategy effectiveness
    if (!strategy_metrics_.empty()) {
        float total_eff = 0.0f;
        for (const auto& [name, metrics] : strategy_metrics_) {
            if (domain.empty() || name.find(domain) == 0) {
                total_eff += metrics.effectiveness_score;
            }
        }
        insights["average_effectiveness"] = total_eff / static_cast<float>(strategy_metrics_.size());
    }
    
    // Adaptation success rate
    if (total_strategic_decisions_ > 0) {
        insights["adaptation_success_rate"] = 
            static_cast<float>(successful_adaptations_) / 
            static_cast<float>(total_strategic_decisions_);
    }
    
    // Experience-based insights
    if (!experience_history_.empty()) {
        float avg_success = 0.0f;
        for (const auto& exp : experience_history_) {
            avg_success += exp.success_score;
        }
        insights["experiential_success_rate"] = 
            avg_success / static_cast<float>(experience_history_.size());
    }
    
    insights["total_experiences"] = static_cast<float>(experience_history_.size());
    insights["total_strategies_tracked"] = static_cast<float>(strategy_metrics_.size());
    
    return insights;
}

StrategyPerformanceMetrics MetaStrategicReasoner::get_strategy_metrics(
    const std::string& strategy_name) const {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    auto it = strategy_metrics_.find(strategy_name);
    if (it != strategy_metrics_.end()) {
        return it->second;
    }
    
    return StrategyPerformanceMetrics();
}

void MetaStrategicReasoner::update_config(
    const std::unordered_map<std::string, float>& config) {
    
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
    
    // Update planning state
    planning_state_.exploration_rate = config_["exploration_rate"];
    planning_state_.exploitation_rate = 1.0f - config_["exploration_rate"];
}

std::unordered_map<std::string, float> MetaStrategicReasoner::get_statistics() const {
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    std::unordered_map<std::string, float> stats;
    stats["total_strategic_decisions"] = static_cast<float>(total_strategic_decisions_);
    stats["successful_adaptations"] = static_cast<float>(successful_adaptations_);
    stats["total_plans_generated"] = static_cast<float>(planning_state_.total_plans_generated);
    stats["successful_plans"] = static_cast<float>(planning_state_.successful_plans);
    stats["experience_count"] = static_cast<float>(experience_history_.size());
    stats["tracked_strategies"] = static_cast<float>(strategy_metrics_.size());
    
    return stats;
}

void MetaStrategicReasoner::reset() {
    std::lock_guard<std::mutex> lock(reasoner_mutex_);
    
    strategy_metrics_.clear();
    experience_history_.clear();
    
    planning_state_.total_plans_generated = 0;
    planning_state_.successful_plans = 0;
    
    total_strategic_decisions_ = 0;
    successful_adaptations_ = 0;
}

// Private methods

std::vector<std::string> MetaStrategicReasoner::generate_strategy_sequence(
    const StrategicContext& context,
    size_t max_length) {
    
    std::vector<std::string> sequence;
    
    // Generate sequence based on context demands
    std::vector<std::string> candidate_strategies = {
        "analytical", "creative", "systematic", "adaptive", 
        "exploratory", "exploitative", "collaborative", "focused"
    };
    
    // Explore vs exploit decision
    if (should_explore()) {
        // Random exploration
        for (size_t i = 0; i < max_length && i < candidate_strategies.size(); ++i) {
            sequence.push_back(candidate_strategies[i]);
        }
    } else {
        // Exploit best known strategies
        std::vector<std::pair<std::string, float>> scored;
        for (const auto& strategy : candidate_strategies) {
            float score = predict_strategy_performance(strategy, context);
            scored.emplace_back(strategy, score);
        }
        
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (size_t i = 0; i < max_length && i < scored.size(); ++i) {
            sequence.push_back(scored[i].first);
        }
    }
    
    return sequence;
}

float MetaStrategicReasoner::compute_plan_alignment(
    const StrategicPlan& plan,
    const StrategicContext& context) const {
    
    // Simple alignment based on expected outcomes vs goals
    float alignment = 0.0f;
    size_t count = 0;
    
    for (const auto& [strategy, expected_outcome] : plan.expected_outcomes) {
        for (const auto& [goal, goal_value] : context.goals) {
            // Check if outcome aligns with goal
            float diff = std::abs(expected_outcome - goal_value);
            alignment += (1.0f - diff);
            count++;
        }
    }
    
    if (count > 0) {
        alignment /= static_cast<float>(count);
    }
    
    return std::max(0.0f, std::min(1.0f, alignment));
}

float MetaStrategicReasoner::compute_context_similarity(
    const StrategicContext& ctx1,
    const StrategicContext& ctx2) const {
    
    // Compute similarity based on state and goals overlap
    float similarity = 0.0f;
    size_t count = 0;
    
    // State similarity
    for (const auto& [key, value1] : ctx1.current_state) {
        auto it = ctx2.current_state.find(key);
        if (it != ctx2.current_state.end()) {
            float diff = std::abs(value1 - it->second);
            similarity += (1.0f - diff);
            count++;
        }
    }
    
    // Goal similarity
    for (const auto& [key, value1] : ctx1.goals) {
        auto it = ctx2.goals.find(key);
        if (it != ctx2.goals.end()) {
            float diff = std::abs(value1 - it->second);
            similarity += (1.0f - diff);
            count++;
        }
    }
    
    if (count > 0) {
        similarity /= static_cast<float>(count);
    }
    
    return similarity;
}

void MetaStrategicReasoner::update_strategy_metrics(
    const std::string& strategy_name,
    float performance,
    const StrategicContext& context) {
    
    auto& metrics = strategy_metrics_[strategy_name];
    
    // Update using exponential moving average
    float alpha = config_["learning_rate"];
    
    if (metrics.execution_count == 0) {
        metrics.effectiveness_score = performance;
    } else {
        metrics.effectiveness_score = (1.0f - alpha) * metrics.effectiveness_score + 
                                     alpha * performance;
    }
    
    // Update other metrics
    metrics.robustness = 0.8f * metrics.robustness + 0.2f * performance;
    metrics.efficiency = 0.9f;  // Simplified efficiency metric
    metrics.adaptation_rate = alpha;
    metrics.generalization_ability = 0.7f;  // Simplified generalization
    
    metrics.execution_count++;
    metrics.last_execution = std::chrono::high_resolution_clock::now();
}

std::vector<MetaStrategicReasoner::StrategicExperience> 
MetaStrategicReasoner::find_similar_experiences(
    const StrategicContext& context,
    size_t top_k) const {
    
    std::vector<std::pair<StrategicExperience, float>> scored;
    
    for (const auto& exp : experience_history_) {
        float similarity = compute_context_similarity(context, exp.context);
        scored.emplace_back(exp, similarity);
    }
    
    // Sort by similarity descending
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top k
    std::vector<StrategicExperience> result;
    size_t num_results = std::min(top_k, scored.size());
    for (size_t i = 0; i < num_results; ++i) {
        result.push_back(scored[i].first);
    }
    
    return result;
}

float MetaStrategicReasoner::estimate_transfer_effectiveness(
    const std::string& source_domain,
    const std::string& target_domain) const {
    
    // Simple domain similarity heuristic
    // In practice, this would use more sophisticated domain analysis
    
    if (source_domain == target_domain) {
        return 1.0f;
    }
    
    // Compute string similarity as a proxy
    size_t common = 0;
    size_t min_len = std::min(source_domain.length(), target_domain.length());
    for (size_t i = 0; i < min_len; ++i) {
        if (source_domain[i] == target_domain[i]) {
            common++;
        }
    }
    
    float similarity = static_cast<float>(common) / 
                      static_cast<float>(std::max(source_domain.length(), 
                                                  target_domain.length()));
    
    return similarity * 0.7f;  // Cap transfer effectiveness
}

bool MetaStrategicReasoner::should_explore() const {
    // Epsilon-greedy exploration using modern C++ random
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_val = dist(gen);
    return random_val < planning_state_.exploration_rate;
}

void MetaStrategicReasoner::prune_experience_history() {
    // Keep only the most recent experiences up to max size
    if (experience_history_.size() > max_experience_history_) {
        size_t to_remove = experience_history_.size() - max_experience_history_;
        experience_history_.erase(experience_history_.begin(), 
                                 experience_history_.begin() + to_remove);
    }
}

} // namespace rros

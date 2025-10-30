#include "adaptive_learning_strategy.hpp"
#include "meta_learning_engine.hpp"
#include <algorithm>
#include <cmath>

namespace rros {

AdaptiveLearningStrategy::AdaptiveLearningStrategy(
    const std::unordered_map<std::string, float>& config)
    : config_(config) {
    
    initialize_strategies();
}

AdaptiveLearningStrategy::~AdaptiveLearningStrategy() = default;

void AdaptiveLearningStrategy::initialize_strategies() {
    // Initialize common learning strategies
    
    // SGD Strategy
    LearningStrategy sgd;
    sgd.strategy_name = "sgd";
    sgd.strategy_type = "gradient_descent";
    sgd.learning_rate = 0.01f;
    sgd.momentum = 0.0f;
    sgd.hyperparameters["weight_decay"] = 0.0001f;
    available_strategies_.push_back(sgd);
    
    // Momentum Strategy
    LearningStrategy momentum;
    momentum.strategy_name = "momentum";
    momentum.strategy_type = "gradient_descent";
    momentum.learning_rate = 0.01f;
    momentum.momentum = 0.9f;
    momentum.hyperparameters["weight_decay"] = 0.0001f;
    available_strategies_.push_back(momentum);
    
    // Adam-like Strategy
    LearningStrategy adam;
    adam.strategy_name = "adam";
    adam.strategy_type = "adaptive";
    adam.learning_rate = 0.001f;
    adam.momentum = 0.9f;
    adam.hyperparameters["beta2"] = 0.999f;
    adam.hyperparameters["epsilon"] = 1e-8f;
    available_strategies_.push_back(adam);
    
    // Aggressive Strategy for easy tasks
    LearningStrategy aggressive;
    aggressive.strategy_name = "aggressive";
    aggressive.strategy_type = "gradient_descent";
    aggressive.learning_rate = 0.1f;
    aggressive.momentum = 0.5f;
    available_strategies_.push_back(aggressive);
    
    // Conservative Strategy for difficult tasks
    LearningStrategy conservative;
    conservative.strategy_name = "conservative";
    conservative.strategy_type = "adaptive";
    conservative.learning_rate = 0.0001f;
    conservative.momentum = 0.95f;
    available_strategies_.push_back(conservative);
}

LearningStrategy AdaptiveLearningStrategy::select_strategy(
    const LearningTask& task,
    const std::unordered_map<std::string, float>& performance_feedback,
    float difficulty) {
    
    std::lock_guard<std::mutex> lock(strategy_mutex_);
    
    // Evaluate fit of each strategy
    float best_fit = -1.0f;
    LearningStrategy best_strategy = available_strategies_[0];
    
    for (const auto& strategy : available_strategies_) {
        float fit = evaluate_strategy_fit(strategy, task, difficulty);
        
        // Adjust fit based on historical performance
        auto perf_it = strategy_performance_.find(strategy.strategy_name);
        if (perf_it != strategy_performance_.end()) {
            fit = 0.7f * fit + 0.3f * perf_it->second;
        }
        
        if (fit > best_fit) {
            best_fit = fit;
            best_strategy = strategy;
        }
    }
    
    // Update performance if feedback available
    auto feedback_it = performance_feedback.find("accuracy");
    if (feedback_it != performance_feedback.end()) {
        strategy_performance_[best_strategy.strategy_name] = feedback_it->second;
    }
    
    best_strategy.performance_score = best_fit;
    return best_strategy;
}

void AdaptiveLearningStrategy::update_config(
    const std::unordered_map<std::string, float>& config) {
    std::lock_guard<std::mutex> lock(strategy_mutex_);
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
}

void AdaptiveLearningStrategy::reset() {
    std::lock_guard<std::mutex> lock(strategy_mutex_);
    strategy_performance_.clear();
}

float AdaptiveLearningStrategy::evaluate_strategy_fit(
    const LearningStrategy& strategy,
    const LearningTask& task,
    float difficulty) const {
    
    float fit = 0.5f;
    
    // Difficulty-based selection
    if (difficulty < 0.3f) {
        // Easy task - prefer aggressive strategies
        if (strategy.strategy_name == "aggressive" || strategy.learning_rate > 0.01f) {
            fit += 0.3f;
        }
    } else if (difficulty > 0.7f) {
        // Hard task - prefer conservative strategies
        if (strategy.strategy_name == "conservative" || 
            strategy.strategy_type == "adaptive") {
            fit += 0.3f;
        }
    } else {
        // Medium task - prefer balanced strategies
        if (strategy.strategy_name == "momentum" || strategy.strategy_name == "adam") {
            fit += 0.3f;
        }
    }
    
    // Task characteristics consideration
    auto complexity_it = task.task_characteristics.find("complexity");
    if (complexity_it != task.task_characteristics.end()) {
        if (complexity_it->second > 0.7f && strategy.strategy_type == "adaptive") {
            fit += 0.2f;
        }
    }
    
    return std::min(1.0f, fit);
}

} // namespace rros

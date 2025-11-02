#include "cognitive_strategy_selector.hpp"
#include "meta_cognitive_synthesis.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

// StrategyEvaluation implementation
StrategyEvaluation::StrategyEvaluation()
    : expected_effectiveness(0.5f)
    , confidence(0.5f)
    , cost(0.5f)
    , risk(0.5f)
    , alignment(0.5f)
{}

float StrategyEvaluation::utility_score() const {
    return expected_effectiveness * alignment - cost - risk * 0.5f;
}

// SelectionContext implementation
SelectionContext::SelectionContext() {}

// StrategyPerformance implementation
CognitiveStrategySelector::StrategyPerformance::StrategyPerformance()
    : average_effectiveness(0.5f)
    , variance(0.0f)
    , selection_count(0)
{}

void CognitiveStrategySelector::StrategyPerformance::update(
    float outcome,
    const SelectionContext& ctx
) {
    outcomes.push_back(outcome);
    contexts.push_back(ctx);
    selection_count++;
    
    // Update running average
    float sum = std::accumulate(outcomes.begin(), outcomes.end(), 0.0f);
    average_effectiveness = sum / outcomes.size();
    
    // Update variance
    variance = 0.0f;
    for (float val : outcomes) {
        float diff = val - average_effectiveness;
        variance += diff * diff;
    }
    variance /= outcomes.size();
}

// CognitiveStrategySelector implementation
CognitiveStrategySelector::CognitiveStrategySelector(
    const std::unordered_map<std::string, float>& config
)
    : config_(config)
    , total_selections_(0)
    , successful_selections_(0)
{
    optimization_state_.learning_rate = 
        config.count("learning_rate") ? config.at("learning_rate") : 0.05f;
    optimization_state_.exploration_rate = 
        config.count("exploration_rate") ? config.at("exploration_rate") : 0.1f;
    optimization_state_.total_selections = 0;
}

CognitiveStrategySelector::~CognitiveStrategySelector() {}

std::pair<MetaCognitiveStrategy, float> CognitiveStrategySelector::select_strategy(
    const SelectionContext& context,
    const std::vector<MetaCognitiveStrategy>& available_strategies
) {
    if (available_strategies.empty()) {
        return {MetaCognitiveStrategy(), 0.0f};
    }
    
    total_selections_++;
    optimization_state_.total_selections++;
    
    // Exploration vs exploitation
    if (should_explore()) {
        // Random selection for exploration
        size_t idx = total_selections_ % available_strategies.size();
        return {available_strategies[idx], 0.5f};
    }
    
    // Exploitation: select best strategy
    float best_score = -1000.0f;
    MetaCognitiveStrategy best_strategy;
    float best_confidence = 0.0f;
    
    for (const auto& strategy : available_strategies) {
        auto eval = evaluate_strategy(strategy, context);
        float score = eval.utility_score();
        
        if (score > best_score) {
            best_score = score;
            best_strategy = strategy;
            best_confidence = eval.confidence;
        }
    }
    
    return {best_strategy, best_confidence};
}

StrategyEvaluation CognitiveStrategySelector::evaluate_strategy(
    const MetaCognitiveStrategy& strategy,
    const SelectionContext& context
) {
    StrategyEvaluation eval;
    
    eval.expected_effectiveness = compute_expected_effectiveness(strategy, context);
    eval.alignment = compute_alignment_score(strategy, context);
    eval.cost = 0.3f;  // Simplified cost model
    eval.risk = 1.0f - strategy.effectiveness;  // Higher effectiveness = lower risk
    eval.confidence = strategy.effectiveness;
    
    return eval;
}

MetaCognitiveStrategy CognitiveStrategySelector::optimize_strategy(
    const MetaCognitiveStrategy& strategy,
    const SelectionContext& context,
    const std::vector<float>& performance_history
) {
    MetaCognitiveStrategy optimized = strategy;
    
    if (performance_history.empty()) {
        return optimized;
    }
    
    // Compute performance gradient
    std::vector<float> gradient;
    for (size_t i = 1; i < performance_history.size(); i++) {
        gradient.push_back(performance_history[i] - performance_history[i-1]);
    }
    
    optimized = apply_parameter_optimization(strategy, gradient);
    
    return optimized;
}

void CognitiveStrategySelector::update_effectiveness(
    const std::string& strategy_name,
    float outcome,
    const SelectionContext& context
) {
    if (performance_tracking_.find(strategy_name) == performance_tracking_.end()) {
        performance_tracking_[strategy_name] = StrategyPerformance();
    }
    
    performance_tracking_[strategy_name].update(outcome, context);
    
    if (outcome > 0.6f) {
        successful_selections_++;
    }
    
    // Update learned mappings
    MetaCognitiveStrategy temp_strategy(strategy_name);
    update_learned_mappings(context, temp_strategy, outcome);
}

void CognitiveStrategySelector::learn_from_experiences(
    const std::vector<std::tuple<SelectionContext, MetaCognitiveStrategy, float>>& experiences
) {
    for (const auto& [ctx, strategy, outcome] : experiences) {
        update_effectiveness(strategy.name, outcome, ctx);
    }
}

std::vector<float> CognitiveStrategySelector::get_effectiveness_history(
    const std::string& strategy_name
) const {
    auto it = performance_tracking_.find(strategy_name);
    if (it != performance_tracking_.end()) {
        return it->second.outcomes;
    }
    return {};
}

std::vector<std::pair<std::string, float>> CognitiveStrategySelector::recommend_strategies(
    const SelectionContext& context,
    size_t num_recommendations
) {
    std::vector<std::pair<std::string, float>> recommendations;
    
    // Sort strategies by expected effectiveness
    std::vector<std::pair<std::string, float>> candidates;
    for (const auto& [name, perf] : performance_tracking_) {
        candidates.push_back({name, perf.average_effectiveness});
    }
    
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    size_t count = std::min(num_recommendations, candidates.size());
    for (size_t i = 0; i < count; i++) {
        recommendations.push_back(candidates[i]);
    }
    
    return recommendations;
}

float CognitiveStrategySelector::detect_conflicts(
    const std::vector<MetaCognitiveStrategy>& strategies
) {
    if (strategies.size() < 2) {
        return 0.0f;
    }
    
    float total_conflict = 0.0f;
    int comparisons = 0;
    
    // Check for conflicting prerequisites
    for (size_t i = 0; i < strategies.size(); i++) {
        for (size_t j = i + 1; j < strategies.size(); j++) {
            // Simplified conflict detection
            if (strategies[i].name == strategies[j].name) {
                total_conflict += 1.0f;
            }
            comparisons++;
        }
    }
    
    return comparisons > 0 ? total_conflict / comparisons : 0.0f;
}

std::unordered_map<std::string, float> CognitiveStrategySelector::get_statistics() const {
    std::unordered_map<std::string, float> stats;
    stats["total_selections"] = static_cast<float>(total_selections_);
    stats["successful_selections"] = static_cast<float>(successful_selections_);
    stats["tracked_strategies"] = static_cast<float>(performance_tracking_.size());
    stats["learned_mappings"] = static_cast<float>(learned_mappings_.size());
    stats["exploration_rate"] = optimization_state_.exploration_rate;
    
    return stats;
}

void CognitiveStrategySelector::reset() {
    performance_tracking_.clear();
    learned_mappings_.clear();
    total_selections_ = 0;
    successful_selections_ = 0;
    optimization_state_.total_selections = 0;
}

// Private methods

float CognitiveStrategySelector::compute_context_similarity(
    const SelectionContext& ctx1,
    const SelectionContext& ctx2
) const {
    auto features1 = extract_context_features(ctx1);
    auto features2 = extract_context_features(ctx2);
    
    if (features1.empty() || features2.empty()) {
        return 0.0f;
    }
    
    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    size_t min_size = std::min(features1.size(), features2.size());
    for (size_t i = 0; i < min_size; i++) {
        dot += features1[i] * features2[i];
        norm1 += features1[i] * features1[i];
        norm2 += features2[i] * features2[i];
    }
    
    if (norm1 > 0 && norm2 > 0) {
        return dot / (std::sqrt(norm1) * std::sqrt(norm2));
    }
    
    return 0.0f;
}

float CognitiveStrategySelector::compute_expected_effectiveness(
    const MetaCognitiveStrategy& strategy,
    const SelectionContext& context
) const {
    auto it = performance_tracking_.find(strategy.name);
    if (it != performance_tracking_.end()) {
        return it->second.average_effectiveness;
    }
    
    return strategy.effectiveness;
}

float CognitiveStrategySelector::compute_alignment_score(
    const MetaCognitiveStrategy& strategy,
    const SelectionContext& context
) const {
    float alignment = strategy.applicability;
    
    // Check goal alignment
    for (const auto& [key, value] : context.goals) {
        if (strategy.parameters.find(key) != strategy.parameters.end()) {
            alignment += 0.1f;
        }
    }
    
    return std::min(1.0f, alignment);
}

std::vector<float> CognitiveStrategySelector::extract_context_features(
    const SelectionContext& context
) const {
    std::vector<float> features = context.state_vector;
    
    for (const auto& [key, value] : context.cognitive_state) {
        features.push_back(value);
    }
    
    return features;
}

void CognitiveStrategySelector::update_learned_mappings(
    const SelectionContext& context,
    const MetaCognitiveStrategy& strategy,
    float outcome
) {
    ContextStrategyMapping mapping;
    mapping.context_features = extract_context_features(context);
    mapping.preferred_strategy = strategy.name;
    mapping.preference_strength = outcome;
    
    learned_mappings_.push_back(mapping);
    
    // Limit size
    if (learned_mappings_.size() > 1000) {
        learned_mappings_.erase(learned_mappings_.begin(), 
                               learned_mappings_.begin() + 100);
    }
}

bool CognitiveStrategySelector::should_explore() const {
    float random_val = static_cast<float>(rand()) / RAND_MAX;
    return random_val < optimization_state_.exploration_rate;
}

MetaCognitiveStrategy CognitiveStrategySelector::apply_parameter_optimization(
    const MetaCognitiveStrategy& strategy,
    const std::vector<float>& performance_gradient
) const {
    MetaCognitiveStrategy optimized = strategy;
    
    if (performance_gradient.empty()) {
        return optimized;
    }
    
    // Compute average gradient
    float avg_gradient = std::accumulate(
        performance_gradient.begin(), 
        performance_gradient.end(), 
        0.0f
    ) / performance_gradient.size();
    
    // Update effectiveness estimate
    optimized.effectiveness += optimization_state_.learning_rate * avg_gradient;
    optimized.effectiveness = std::max(0.0f, std::min(1.0f, optimized.effectiveness));
    
    return optimized;
}

} // namespace rros

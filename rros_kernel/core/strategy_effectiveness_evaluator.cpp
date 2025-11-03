#include "strategy_effectiveness_evaluator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {

// EvaluationCriteria implementation
EvaluationCriteria::EvaluationCriteria()
    : accuracy_weight(0.3f)
    , speed_weight(0.2f)
    , robustness_weight(0.2f)
    , generalization_weight(0.2f)
    , efficiency_weight(0.1f) {}

// EffectivenessResults implementation
EffectivenessResults::EffectivenessResults()
    : overall_effectiveness(0.0f)
    , accuracy_score(0.0f)
    , speed_score(0.0f)
    , robustness_score(0.0f)
    , generalization_score(0.0f)
    , efficiency_score(0.0f)
    , confidence(0.0f) {}

// PredictionModel implementation
PredictionModel::PredictionModel()
    : model_type("linear")
    , prediction_accuracy(0.0f) {}

// PerformanceTracker implementation
StrategyEffectivenessEvaluator::PerformanceTracker::PerformanceTracker()
    : running_mean(0.0f)
    , running_variance(0.0f)
    , sample_count(0) {}

void StrategyEffectivenessEvaluator::PerformanceTracker::update(
    float value, 
    const std::vector<float>& context) {
    
    history.push_back(value);
    context_history.push_back(context);
    
    // Update running statistics
    sample_count++;
    float delta = value - running_mean;
    running_mean += delta / static_cast<float>(sample_count);
    float delta2 = value - running_mean;
    running_variance += delta * delta2;
    
    // Limit history size
    if (history.size() > 1000) {
        history.erase(history.begin());
        context_history.erase(context_history.begin());
    }
}

// StrategyEffectivenessEvaluator implementation
StrategyEffectivenessEvaluator::StrategyEffectivenessEvaluator(
    const std::unordered_map<std::string, float>& config)
    : config_(config)
    , total_evaluations_(0)
    , total_predictions_(0) {
    
    // Set default configuration
    if (config_.find("prediction_threshold") == config_.end()) {
        config_["prediction_threshold"] = 0.6f;
    }
    if (config_.find("degradation_threshold") == config_.end()) {
        config_["degradation_threshold"] = 0.3f;
    }
    if (config_.find("min_samples_for_prediction") == config_.end()) {
        config_["min_samples_for_prediction"] = 5.0f;
    }
}

StrategyEffectivenessEvaluator::~StrategyEffectivenessEvaluator() {}

EffectivenessResults StrategyEffectivenessEvaluator::evaluate_effectiveness(
    const std::string& strategy_name,
    const std::vector<float>& context_data,
    const std::vector<float>& performance_history,
    const EvaluationCriteria& criteria) {
    
    EffectivenessResults results;
    
    // Compute individual components
    results.accuracy_score = compute_accuracy_score(performance_history);
    results.speed_score = compute_speed_score(context_data);
    results.robustness_score = compute_robustness_score(performance_history);
    results.generalization_score = compute_generalization_score(strategy_name, context_data);
    results.efficiency_score = compute_efficiency_score(context_data);
    
    // Compute overall effectiveness
    results.overall_effectiveness = 
        criteria.accuracy_weight * results.accuracy_score +
        criteria.speed_weight * results.speed_score +
        criteria.robustness_weight * results.robustness_score +
        criteria.generalization_weight * results.generalization_score +
        criteria.efficiency_weight * results.efficiency_score;
    
    // Compute confidence based on sample size
    float sample_confidence = std::min(1.0f, 
        static_cast<float>(performance_history.size()) / 20.0f);
    
    // Adjust confidence based on variance
    float variance = compute_robustness_score(performance_history);
    float variance_confidence = variance;  // Higher robustness = higher confidence
    
    results.confidence = 0.7f * sample_confidence + 0.3f * variance_confidence;
    
    // Store detailed metrics
    results.detailed_metrics["sample_count"] = static_cast<float>(performance_history.size());
    results.detailed_metrics["mean_performance"] = results.accuracy_score;
    results.detailed_metrics["variance"] = 1.0f - results.robustness_score;
    
    total_evaluations_++;
    
    return results;
}

float StrategyEffectivenessEvaluator::predict_performance(
    const std::string& strategy_name,
    const std::vector<float>& context_data) {
    
    auto model_it = prediction_models_.find(strategy_name);
    if (model_it == prediction_models_.end()) {
        // No model exists - return neutral prediction
        return 0.5f;
    }
    
    float prediction = apply_prediction_model(model_it->second, context_data);
    total_predictions_++;
    
    return prediction;
}

std::vector<std::pair<std::string, float>> StrategyEffectivenessEvaluator::compare_strategies(
    const std::vector<std::string>& strategies,
    const std::vector<float>& context_data) {
    
    std::vector<std::pair<std::string, float>> ranked;
    
    for (const auto& strategy : strategies) {
        float predicted_perf = predict_performance(strategy, context_data);
        ranked.emplace_back(strategy, predicted_perf);
    }
    
    // Sort by predicted performance descending
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return ranked;
}

void StrategyEffectivenessEvaluator::update_prediction_model(
    const std::string& strategy_name,
    const std::vector<float>& context_data,
    float actual_performance) {
    
    // Update performance tracker
    auto& tracker = performance_trackers_[strategy_name];
    tracker.update(actual_performance, context_data);
    
    // Retrain model if we have enough samples
    if (tracker.sample_count >= static_cast<uint64_t>(config_["min_samples_for_prediction"])) {
        auto& model = prediction_models_[strategy_name];
        train_prediction_model(model, tracker.context_history, tracker.history);
    }
}

float StrategyEffectivenessEvaluator::detect_degradation(
    const std::string& strategy_name,
    const std::vector<float>& recent_performance) {
    
    if (recent_performance.size() < 2) {
        return 0.0f;  // Not enough data
    }
    
    auto tracker_it = performance_trackers_.find(strategy_name);
    if (tracker_it == performance_trackers_.end()) {
        return 0.0f;  // No historical data
    }
    
    // Compute recent mean
    float recent_mean = std::accumulate(recent_performance.begin(), 
                                       recent_performance.end(), 0.0f) / 
                       static_cast<float>(recent_performance.size());
    
    // Compare to historical mean
    float historical_mean = tracker_it->second.running_mean;
    
    // Degradation is the negative difference from historical performance
    float degradation = std::max(0.0f, historical_mean - recent_mean);
    
    // Normalize to [0, 1]
    return std::min(1.0f, degradation);
}

float StrategyEffectivenessEvaluator::get_evaluation_confidence(
    const std::string& strategy_name) const {
    
    auto tracker_it = performance_trackers_.find(strategy_name);
    if (tracker_it == performance_trackers_.end()) {
        return 0.0f;
    }
    
    // Confidence based on sample count and variance
    float sample_confidence = std::min(1.0f, 
        static_cast<float>(tracker_it->second.sample_count) / 20.0f);
    
    float variance = tracker_it->second.sample_count > 1 ? 
        tracker_it->second.running_variance / static_cast<float>(tracker_it->second.sample_count - 1) : 
        1.0f;
    
    float variance_confidence = std::max(0.0f, 1.0f - variance);
    
    return 0.6f * sample_confidence + 0.4f * variance_confidence;
}

std::unordered_map<std::string, float> StrategyEffectivenessEvaluator::get_detailed_metrics(
    const std::string& strategy_name) const {
    
    std::unordered_map<std::string, float> metrics;
    
    auto tracker_it = performance_trackers_.find(strategy_name);
    if (tracker_it != performance_trackers_.end()) {
        const auto& tracker = tracker_it->second;
        metrics["sample_count"] = static_cast<float>(tracker.sample_count);
        metrics["running_mean"] = tracker.running_mean;
        
        if (tracker.sample_count > 1) {
            metrics["running_variance"] = tracker.running_variance / 
                static_cast<float>(tracker.sample_count - 1);
            metrics["running_std"] = std::sqrt(metrics["running_variance"]);
        }
    }
    
    auto model_it = prediction_models_.find(strategy_name);
    if (model_it != prediction_models_.end()) {
        metrics["prediction_accuracy"] = model_it->second.prediction_accuracy;
    }
    
    metrics["evaluation_confidence"] = get_evaluation_confidence(strategy_name);
    
    return metrics;
}

void StrategyEffectivenessEvaluator::reset() {
    prediction_models_.clear();
    performance_trackers_.clear();
    total_evaluations_ = 0;
    total_predictions_ = 0;
}

// Private methods

float StrategyEffectivenessEvaluator::compute_accuracy_score(
    const std::vector<float>& performance_history) const {
    
    if (performance_history.empty()) {
        return 0.5f;
    }
    
    // Return mean performance
    float sum = std::accumulate(performance_history.begin(), 
                               performance_history.end(), 0.0f);
    return sum / static_cast<float>(performance_history.size());
}

float StrategyEffectivenessEvaluator::compute_speed_score(
    const std::vector<float>& context_data) const {
    
    // Simplified speed score based on context complexity
    // In practice, would measure actual execution time
    float complexity = context_data.empty() ? 1.0f : 
        std::accumulate(context_data.begin(), context_data.end(), 0.0f) / 
        static_cast<float>(context_data.size());
    
    // Inverse of complexity (lower complexity = faster)
    return std::max(0.0f, std::min(1.0f, 1.0f - complexity * 0.5f));
}

float StrategyEffectivenessEvaluator::compute_robustness_score(
    const std::vector<float>& performance_history) const {
    
    if (performance_history.size() < 2) {
        return 0.5f;
    }
    
    // Compute variance
    float mean = std::accumulate(performance_history.begin(), 
                                performance_history.end(), 0.0f) / 
                static_cast<float>(performance_history.size());
    
    float variance = 0.0f;
    for (float value : performance_history) {
        float diff = value - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(performance_history.size());
    
    // Robustness is inverse of variance (normalized)
    float std_dev = std::sqrt(variance);
    return std::max(0.0f, std::min(1.0f, 1.0f - std_dev));
}

float StrategyEffectivenessEvaluator::compute_generalization_score(
    const std::string& strategy_name,
    const std::vector<float>& context_data) const {
    
    auto tracker_it = performance_trackers_.find(strategy_name);
    if (tracker_it == performance_trackers_.end()) {
        return 0.5f;
    }
    
    // Compute how well strategy performs across different contexts
    // Measure consistency of performance across context variations
    const auto& history = tracker_it->second.history;
    if (history.size() < 2) {
        return 0.5f;
    }
    
    // Use variance as proxy for generalization
    // Low variance across contexts = good generalization
    float robustness = compute_robustness_score(history);
    
    // Also consider sample diversity
    float diversity_bonus = std::min(1.0f, 
        static_cast<float>(tracker_it->second.context_history.size()) / 10.0f);
    
    return 0.7f * robustness + 0.3f * diversity_bonus;
}

float StrategyEffectivenessEvaluator::compute_efficiency_score(
    const std::vector<float>& context_data) const {
    
    // Simplified efficiency based on context size
    // Smaller context = more efficient processing
    if (context_data.empty()) {
        return 1.0f;
    }
    
    float size_factor = std::min(1.0f, 10.0f / static_cast<float>(context_data.size()));
    
    // Consider context complexity
    float avg_value = std::accumulate(context_data.begin(), 
                                     context_data.end(), 0.0f) / 
                     static_cast<float>(context_data.size());
    float complexity_factor = std::max(0.0f, 1.0f - avg_value);
    
    return 0.5f * size_factor + 0.5f * complexity_factor;
}

float StrategyEffectivenessEvaluator::compute_overall_effectiveness(
    const EffectivenessResults& results,
    const EvaluationCriteria& criteria) const {
    
    return criteria.accuracy_weight * results.accuracy_score +
           criteria.speed_weight * results.speed_score +
           criteria.robustness_weight * results.robustness_score +
           criteria.generalization_weight * results.generalization_score +
           criteria.efficiency_weight * results.efficiency_score;
}

float StrategyEffectivenessEvaluator::compute_prediction_confidence(
    const PredictionModel& model,
    const std::vector<float>& context_data) const {
    
    // Confidence based on model accuracy and input characteristics
    float model_confidence = model.prediction_accuracy;
    
    // Reduce confidence for out-of-distribution inputs
    // (simplified - in practice would use more sophisticated detection)
    float input_confidence = context_data.empty() ? 0.5f : 
        std::min(1.0f, 1.0f / static_cast<float>(context_data.size()));
    
    return 0.7f * model_confidence + 0.3f * input_confidence;
}

void StrategyEffectivenessEvaluator::train_prediction_model(
    PredictionModel& model,
    const std::vector<std::vector<float>>& training_contexts,
    const std::vector<float>& training_targets) {
    
    if (training_contexts.empty() || training_targets.empty()) {
        return;
    }
    
    // Simple linear regression model
    model.model_type = "linear";
    
    // Determine input dimension
    size_t input_dim = training_contexts[0].size();
    if (input_dim == 0) {
        input_dim = 1;
    }
    
    // Initialize weights if needed
    if (model.weights.size() != input_dim) {
        model.weights.resize(input_dim, 0.0f);
    }
    if (model.biases.empty()) {
        model.biases.push_back(0.0f);
    }
    
    // Simplified training: compute mean of targets as prediction
    // In practice, would use gradient descent or similar
    float target_mean = std::accumulate(training_targets.begin(), 
                                       training_targets.end(), 0.0f) / 
                       static_cast<float>(training_targets.size());
    
    model.biases[0] = target_mean;
    
    // Estimate prediction accuracy
    float error_sum = 0.0f;
    for (size_t i = 0; i < training_targets.size(); ++i) {
        float prediction = apply_prediction_model(model, training_contexts[i]);
        error_sum += std::abs(prediction - training_targets[i]);
    }
    float mean_error = error_sum / static_cast<float>(training_targets.size());
    
    model.prediction_accuracy = std::max(0.0f, 1.0f - mean_error);
}

float StrategyEffectivenessEvaluator::apply_prediction_model(
    const PredictionModel& model,
    const std::vector<float>& context_data) const {
    
    if (model.model_type == "linear") {
        // Simple linear model: y = w^T x + b
        float prediction = model.biases.empty() ? 0.0f : model.biases[0];
        
        size_t min_size = std::min(model.weights.size(), context_data.size());
        for (size_t i = 0; i < min_size; ++i) {
            prediction += model.weights[i] * context_data[i];
        }
        
        // Clamp to [0, 1]
        return std::max(0.0f, std::min(1.0f, prediction));
    }
    
    return 0.5f;  // Default prediction
}

} // namespace rros

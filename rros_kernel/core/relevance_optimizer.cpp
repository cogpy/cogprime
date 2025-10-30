#include "relevance_optimizer.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace rros {

RelevanceOptimizer::RelevanceOptimizer(const std::unordered_map<std::string, float>& config) :
    config_(config),
    base_threshold_(0.3f),
    adaptive_threshold_(0.3f),
    goal_weight_(0.35f),
    context_weight_(0.25f),
    salience_weight_(0.25f),
    temporal_weight_(0.15f),
    max_feedback_history_(1000),
    assessment_count_(0),
    total_prediction_error_(0.0f),
    average_confidence_(0.0f)
{
    // Load configuration
    if (config_.find("base_threshold") != config_.end()) {
        base_threshold_ = config_["base_threshold"];
        adaptive_threshold_ = base_threshold_;
    }
    if (config_.find("goal_weight") != config_.end()) {
        goal_weight_ = config_["goal_weight"];
    }
    if (config_.find("context_weight") != config_.end()) {
        context_weight_ = config_["context_weight"];
    }
    if (config_.find("salience_weight") != config_.end()) {
        salience_weight_ = config_["salience_weight"];
    }
    if (config_.find("temporal_weight") != config_.end()) {
        temporal_weight_ = config_["temporal_weight"];
    }
    if (config_.find("max_feedback_history") != config_.end()) {
        max_feedback_history_ = static_cast<size_t>(config_["max_feedback_history"]);
    }
    
    // Initialize learned weights
    learned_weights_ = {goal_weight_, context_weight_, salience_weight_, temporal_weight_};
}

RelevanceScore RelevanceOptimizer::assess_relevance(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context,
    const std::vector<Goal>& goals,
    const EnvironmentalContext& env_context
) {
    RelevanceScore score;
    
    // Compute base relevance from input characteristics
    score.base_score = compute_base_relevance(input);
    
    // Apply contextual weighting
    score.contextual_weight = compute_contextual_boost(input, context);
    
    // Compute goal alignment
    score.goal_alignment = compute_goal_alignment(input, goals);
    
    // Detect environmental salience
    score.salience = compute_salience_score(input, env_context);
    
    // Compute temporal relevance
    score.temporal_relevance = compute_temporal_relevance(input, env_context);
    
    // Combine scores using learned weights
    score.final_score = 
        score.base_score * (
            learned_weights_[0] * score.goal_alignment +
            learned_weights_[1] * score.contextual_weight +
            learned_weights_[2] * score.salience +
            learned_weights_[3] * score.temporal_relevance
        );
    
    // Normalize final score
    score.final_score = std::min(1.0f, std::max(0.0f, score.final_score));
    
    // Compute confidence based on agreement between components
    std::vector<float> components = {
        score.goal_alignment,
        score.contextual_weight / 2.0f, // Normalize to [0,1]
        score.salience,
        score.temporal_relevance
    };
    
    float mean_component = std::accumulate(components.begin(), components.end(), 0.0f) / components.size();
    float variance = 0.0f;
    for (float c : components) {
        float diff = c - mean_component;
        variance += diff * diff;
    }
    variance /= components.size();
    
    // Lower variance = higher agreement = higher confidence
    score.confidence = std::exp(-variance * 2.0f);
    
    // Update metrics
    ++assessment_count_;
    average_confidence_ = (average_confidence_ * (assessment_count_ - 1) + score.confidence) / assessment_count_;
    
    return score;
}

std::vector<float> RelevanceOptimizer::allocate_attention_dynamic(
    const std::vector<std::vector<float>>& items,
    const std::vector<RelevanceScore>& scores,
    float budget
) {
    if (items.empty() || items.size() != scores.size()) {
        return {};
    }
    
    std::vector<float> allocation(items.size(), 0.0f);
    
    // Extract final scores and confidences
    std::vector<float> weighted_scores(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        // Weight by confidence - more confident predictions get more attention
        weighted_scores[i] = scores[i].final_score * (0.5f + 0.5f * scores[i].confidence);
    }
    
    // Softmax allocation with temperature control
    float temperature = config_.find("attention_temperature") != config_.end() ? 
                       config_["attention_temperature"] : 2.0f;
    
    std::vector<float> exp_scores(weighted_scores.size());
    float exp_sum = 0.0f;
    
    for (size_t i = 0; i < weighted_scores.size(); ++i) {
        exp_scores[i] = std::exp(weighted_scores[i] * temperature);
        exp_sum += exp_scores[i];
    }
    
    // Allocate budget proportionally
    for (size_t i = 0; i < allocation.size(); ++i) {
        allocation[i] = (exp_scores[i] / exp_sum) * budget;
    }
    
    // Apply winner-take-more strategy for high-relevance items
    for (size_t i = 0; i < allocation.size(); ++i) {
        if (scores[i].final_score > 0.7f) {
            allocation[i] *= 1.2f; // Boost high relevance items
        }
    }
    
    // Renormalize to budget
    float total = std::accumulate(allocation.begin(), allocation.end(), 0.0f);
    if (total > 0.0f) {
        float scale = budget / total;
        for (float& a : allocation) {
            a *= scale;
        }
    }
    
    return allocation;
}

std::vector<float> RelevanceOptimizer::compute_contextual_weights(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context,
    const std::vector<std::unordered_map<std::string, float>>& history
) {
    std::vector<float> weights(input.size(), 1.0f);
    
    // Compute feature importance based on variance across history
    if (!history.empty()) {
        for (size_t i = 0; i < input.size(); ++i) {
            float feature_variance = 0.0f;
            float mean = input[i];
            
            // Compute variance with historical context
            for (const auto& hist_context : history) {
                if (hist_context.find("feature_" + std::to_string(i)) != hist_context.end()) {
                    float hist_val = hist_context.at("feature_" + std::to_string(i));
                    float diff = hist_val - mean;
                    feature_variance += diff * diff;
                }
            }
            feature_variance /= std::max(1.0f, static_cast<float>(history.size()));
            
            // Higher variance = more informative = higher weight
            weights[i] = 1.0f + std::sqrt(feature_variance);
        }
    }
    
    // Apply context-specific modulation
    if (context.find("focus_mode") != context.end()) {
        float focus = context.at("focus_mode");
        // In focus mode, increase weight variance (sharper focus)
        for (float& w : weights) {
            w = std::pow(w, 1.0f + focus);
        }
    }
    
    // Normalize weights to sum to input size (maintains average weight of 1.0)
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (sum > 0.0f) {
        float scale = static_cast<float>(weights.size()) / sum;
        for (float& w : weights) {
            w *= scale;
        }
    }
    
    return weights;
}

std::vector<float> RelevanceOptimizer::align_with_goals(
    const std::vector<float>& input,
    const std::vector<Goal>& goals
) {
    if (goals.empty()) {
        return std::vector<float>(1, 0.5f); // Neutral alignment
    }
    
    std::vector<float> alignments(goals.size());
    
    for (size_t i = 0; i < goals.size(); ++i) {
        const Goal& goal = goals[i];
        
        // Compute similarity to goal target features
        float similarity = compute_similarity(input, goal.target_features);
        
        // Weight by goal priority and urgency
        float weighted_similarity = similarity * goal.priority * (0.5f + 0.5f * goal.urgency);
        
        // Check constraint satisfaction
        float constraint_satisfaction = 1.0f;
        for (const auto& [key, value] : goal.constraints) {
            // Simple constraint check (could be extended)
            if (key.find("min_") == 0 && similarity < value) {
                constraint_satisfaction *= 0.5f;
            }
            if (key.find("max_") == 0 && similarity > value) {
                constraint_satisfaction *= 0.5f;
            }
        }
        
        alignments[i] = weighted_similarity * constraint_satisfaction;
    }
    
    return alignments;
}

std::pair<std::vector<float>, std::vector<size_t>> RelevanceOptimizer::detect_salience(
    const EnvironmentalContext& env_context
) {
    std::vector<float> salience_map(env_context.current_state.size());
    std::vector<size_t> critical_features;
    
    // Compute salience based on multiple factors
    for (size_t i = 0; i < env_context.current_state.size(); ++i) {
        float magnitude = std::abs(env_context.current_state[i]);
        
        // Change detection
        float change = 0.0f;
        if (i < env_context.recent_changes.size()) {
            change = std::abs(env_context.recent_changes[i]);
        }
        
        // Novelty contribution
        float novelty_contrib = env_context.novelty_score * magnitude;
        
        // Threat/opportunity contribution
        float importance = std::max(env_context.threat_level, env_context.opportunity_score);
        
        // Combine factors
        salience_map[i] = 
            0.3f * magnitude +
            0.4f * change +
            0.2f * novelty_contrib +
            0.1f * importance * magnitude;
    }
    
    // Normalize salience map
    float max_sal = *std::max_element(salience_map.begin(), salience_map.end());
    if (max_sal > 0.0f) {
        for (float& s : salience_map) {
            s /= max_sal;
        }
    }
    
    // Identify critical features (top 20% salience)
    float critical_threshold = 0.8f;
    for (size_t i = 0; i < salience_map.size(); ++i) {
        if (salience_map[i] > critical_threshold) {
            critical_features.push_back(i);
        }
    }
    
    return {salience_map, critical_features};
}

std::vector<size_t> RelevanceOptimizer::retrieve_relevant_memories(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& memory_pool,
    size_t top_k
) {
    if (memory_pool.empty()) {
        return {};
    }
    
    // Compute relevance score for each memory
    std::vector<std::pair<float, size_t>> scored_memories;
    scored_memories.reserve(memory_pool.size());
    
    for (size_t i = 0; i < memory_pool.size(); ++i) {
        float similarity = compute_similarity(query, memory_pool[i]);
        
        // Apply recency bias (more recent memories have slight advantage)
        float recency_boost = 1.0f + 0.1f * (static_cast<float>(i) / memory_pool.size());
        float score = similarity * recency_boost;
        
        scored_memories.emplace_back(score, i);
    }
    
    // Sort by score descending
    std::partial_sort(
        scored_memories.begin(),
        scored_memories.begin() + std::min(top_k, scored_memories.size()),
        scored_memories.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Extract indices
    std::vector<size_t> indices;
    indices.reserve(std::min(top_k, scored_memories.size()));
    for (size_t i = 0; i < std::min(top_k, scored_memories.size()); ++i) {
        indices.push_back(scored_memories[i].second);
    }
    
    return indices;
}

std::pair<std::vector<std::vector<float>>, std::vector<size_t>> 
RelevanceOptimizer::adaptive_filter(
    const std::vector<std::vector<float>>& items,
    const std::vector<RelevanceScore>& scores,
    bool adaptive_threshold
) {
    if (items.empty() || items.size() != scores.size()) {
        return {{}, {}};
    }
    
    // Update adaptive threshold if enabled
    if (adaptive_threshold) {
        update_adaptive_threshold();
    }
    
    float threshold = adaptive_threshold ? adaptive_threshold_ : base_threshold_;
    
    // Filter items above threshold
    std::vector<std::vector<float>> filtered_items;
    std::vector<size_t> filtered_indices;
    
    for (size_t i = 0; i < items.size(); ++i) {
        if (scores[i].final_score >= threshold) {
            filtered_items.push_back(items[i]);
            filtered_indices.push_back(i);
        }
    }
    
    return {filtered_items, filtered_indices};
}

void RelevanceOptimizer::provide_feedback(const RelevanceFeedback& feedback) {
    feedback_history_.push_back(feedback);
    
    // Maintain history size limit
    while (feedback_history_.size() > max_feedback_history_) {
        feedback_history_.pop_front();
    }
    
    // Update prediction error tracking
    total_prediction_error_ += std::abs(feedback.error);
}

std::unordered_map<std::string, float> RelevanceOptimizer::learn_from_feedback() {
    if (feedback_history_.empty()) {
        return {{"error_reduction", 0.0f}, {"accuracy", 0.5f}};
    }
    
    // Analyze feedback patterns
    float total_error = 0.0f;
    float recent_error = 0.0f;
    size_t recent_window = std::min(static_cast<size_t>(100), feedback_history_.size());
    
    for (size_t i = 0; i < feedback_history_.size(); ++i) {
        total_error += std::abs(feedback_history_[i].error);
        
        if (i >= feedback_history_.size() - recent_window) {
            recent_error += std::abs(feedback_history_[i].error);
        }
    }
    
    float avg_total_error = total_error / feedback_history_.size();
    float avg_recent_error = recent_error / recent_window;
    
    // Update learned weights based on feedback
    update_learned_weights();
    
    // Compute improvement metrics
    float error_reduction = (avg_total_error - avg_recent_error) / std::max(0.001f, avg_total_error);
    float accuracy = 1.0f - avg_recent_error;
    
    return {
        {"error_reduction", error_reduction},
        {"accuracy", std::max(0.0f, std::min(1.0f, accuracy))},
        {"total_feedbacks", static_cast<float>(feedback_history_.size())},
        {"average_error", avg_recent_error}
    };
}

float RelevanceOptimizer::get_current_threshold() const {
    return adaptive_threshold_;
}

std::unordered_map<std::string, float> RelevanceOptimizer::get_metrics() const {
    float avg_error = assessment_count_ > 0 ? total_prediction_error_ / assessment_count_ : 0.0f;
    
    return {
        {"assessments_performed", static_cast<float>(assessment_count_)},
        {"average_prediction_error", avg_error},
        {"average_confidence", average_confidence_},
        {"current_threshold", adaptive_threshold_},
        {"feedback_count", static_cast<float>(feedback_history_.size())},
        {"goal_weight", learned_weights_[0]},
        {"context_weight", learned_weights_[1]},
        {"salience_weight", learned_weights_[2]},
        {"temporal_weight", learned_weights_[3]}
    };
}

void RelevanceOptimizer::update_config(const std::unordered_map<std::string, float>& config) {
    for (const auto& [key, value] : config) {
        config_[key] = value;
        
        if (key == "base_threshold") {
            base_threshold_ = value;
        } else if (key == "goal_weight") {
            goal_weight_ = value;
            learned_weights_[0] = value;
        } else if (key == "context_weight") {
            context_weight_ = value;
            learned_weights_[1] = value;
        } else if (key == "salience_weight") {
            salience_weight_ = value;
            learned_weights_[2] = value;
        } else if (key == "temporal_weight") {
            temporal_weight_ = value;
            learned_weights_[3] = value;
        }
    }
}

void RelevanceOptimizer::reset() {
    adaptive_threshold_ = base_threshold_;
    assessment_count_ = 0;
    total_prediction_error_ = 0.0f;
    average_confidence_ = 0.0f;
    feedback_history_.clear();
    learned_weights_ = {goal_weight_, context_weight_, salience_weight_, temporal_weight_};
}

// Private methods

float RelevanceOptimizer::compute_base_relevance(const std::vector<float>& input) {
    if (input.empty()) return 0.0f;
    
    // Compute multiple relevance indicators
    float magnitude = 0.0f;
    for (float val : input) {
        magnitude += val * val;
    }
    magnitude = std::sqrt(magnitude / input.size());
    
    // Compute entropy (information content)
    float entropy = 0.0f;
    float sum = std::accumulate(input.begin(), input.end(), 0.0f);
    if (sum > 0.0f) {
        for (float val : input) {
            if (val > 0.0f) {
                float p = val / sum;
                entropy -= p * std::log2(p + 1e-10f);
            }
        }
    }
    entropy /= std::log2(static_cast<float>(input.size()) + 1.0f); // Normalize
    
    // Combine magnitude and entropy
    return 0.6f * magnitude + 0.4f * entropy;
}

float RelevanceOptimizer::compute_contextual_boost(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    float boost = 1.0f;
    
    // Apply various contextual factors
    if (context.find("attention_focus") != context.end()) {
        boost *= (1.0f + context.at("attention_focus") * 0.5f);
    }
    
    if (context.find("task_priority") != context.end()) {
        boost *= (1.0f + context.at("task_priority") * 0.3f);
    }
    
    if (context.find("cognitive_load") != context.end()) {
        // Higher load reduces boost (limited resources)
        float load = context.at("cognitive_load");
        boost *= (1.0f - load * 0.2f);
    }
    
    if (context.find("novelty") != context.end()) {
        boost *= (1.0f + context.at("novelty") * 0.4f);
    }
    
    return std::max(0.1f, std::min(2.0f, boost));
}

float RelevanceOptimizer::compute_goal_alignment(
    const std::vector<float>& input,
    const std::vector<Goal>& goals
) {
    if (goals.empty()) {
        return 0.5f; // Neutral alignment
    }
    
    std::vector<float> alignments = align_with_goals(input, goals);
    
    // Compute weighted average based on goal priorities
    float total_weight = 0.0f;
    float weighted_sum = 0.0f;
    
    for (size_t i = 0; i < alignments.size(); ++i) {
        float weight = goals[i].priority * goals[i].urgency;
        weighted_sum += alignments[i] * weight;
        total_weight += weight;
    }
    
    return total_weight > 0.0f ? weighted_sum / total_weight : 0.5f;
}

float RelevanceOptimizer::compute_salience_score(
    const std::vector<float>& input,
    const EnvironmentalContext& env_context
) {
    auto [salience_map, critical_features] = detect_salience(env_context);
    
    // Compute overlap between input and salient features
    float salience_score = 0.0f;
    size_t min_size = std::min(input.size(), salience_map.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        salience_score += std::abs(input[i]) * salience_map[i];
    }
    
    salience_score /= std::max(1.0f, static_cast<float>(min_size));
    
    // Boost if input involves critical features
    float critical_boost = 1.0f + 0.3f * static_cast<float>(critical_features.size()) / 
                          std::max(1.0f, static_cast<float>(salience_map.size()));
    
    return std::min(1.0f, salience_score * critical_boost);
}

float RelevanceOptimizer::compute_temporal_relevance(
    const std::vector<float>& input,
    const EnvironmentalContext& env_context
) {
    // Time-sensitive relevance based on urgency signals
    float urgency = env_context.threat_level * 0.6f + env_context.opportunity_score * 0.4f;
    
    // Novelty decays over time - newer is more relevant
    float novelty_factor = env_context.novelty_score;
    
    // Combine urgency and novelty
    float temporal_rel = 0.7f * urgency + 0.3f * novelty_factor;
    
    return std::min(1.0f, std::max(0.0f, temporal_rel));
}

void RelevanceOptimizer::update_adaptive_threshold() {
    if (feedback_history_.empty()) {
        return;
    }
    
    // Compute optimal threshold based on feedback
    // Use recent window for adaptation
    size_t window = std::min(static_cast<size_t>(200), feedback_history_.size());
    
    float sum_predicted = 0.0f;
    float sum_actual = 0.0f;
    
    for (size_t i = feedback_history_.size() - window; i < feedback_history_.size(); ++i) {
        sum_predicted += feedback_history_[i].predicted_relevance;
        sum_actual += feedback_history_[i].actual_outcome;
    }
    
    float avg_predicted = sum_predicted / window;
    float avg_actual = sum_actual / window;
    
    // Adjust threshold based on prediction bias
    float bias = avg_predicted - avg_actual;
    
    // Gradual adaptation
    adaptive_threshold_ += bias * 0.1f;
    
    // Keep within reasonable bounds
    adaptive_threshold_ = std::max(0.1f, std::min(0.8f, adaptive_threshold_));
}

float RelevanceOptimizer::compute_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b
) const {
    if (a.empty() || b.empty()) return 0.0f;
    
    size_t min_size = std::min(a.size(), b.size());
    
    // Compute cosine similarity
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }
    
    // Return normalized similarity [0, 1]
    return std::max(0.0f, dot_product / (norm_a * norm_b));
}

void RelevanceOptimizer::update_learned_weights() {
    if (feedback_history_.size() < 50) {
        return; // Need sufficient data
    }
    
    // Simple gradient descent on weights
    // Analyze which components correlate with positive outcomes
    
    std::vector<float> component_correlations(4, 0.0f);
    size_t window = std::min(static_cast<size_t>(100), feedback_history_.size());
    
    for (size_t i = feedback_history_.size() - window; i < feedback_history_.size(); ++i) {
        const auto& fb = feedback_history_[i];
        
        // Approximate component contributions from context snapshot
        if (fb.context_snapshot.size() >= 4) {
            for (size_t j = 0; j < 4; ++j) {
                float correlation = fb.context_snapshot[j] * fb.actual_outcome;
                component_correlations[j] += correlation;
            }
        }
    }
    
    // Normalize correlations
    for (float& corr : component_correlations) {
        corr /= window;
    }
    
    // Update weights with learning rate
    float learning_rate = 0.01f;
    for (size_t i = 0; i < 4; ++i) {
        learned_weights_[i] += learning_rate * component_correlations[i];
        learned_weights_[i] = std::max(0.1f, std::min(1.0f, learned_weights_[i]));
    }
    
    // Renormalize weights to sum to 1.0
    float sum = std::accumulate(learned_weights_.begin(), learned_weights_.end(), 0.0f);
    if (sum > 0.0f) {
        for (float& w : learned_weights_) {
            w /= sum;
        }
    }
}

} // namespace rros

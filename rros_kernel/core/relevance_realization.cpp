/**
 * @file relevance_realization.cpp
 * @brief Implementation of Vervaeke's Relevance Realization Framework
 * 
 * Comprehensive implementation integrating:
 * - Multi-scale temporal processing (Episode 30)
 * - Opponent processing for attention (Episode 27)
 * - Participatory knowing (Episode 32)
 * - Salience landscaping (Episode 28)
 */

#include "relevance_realization.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace rros {

RelevanceRealizationSystem::RelevanceRealizationSystem(
    const std::unordered_map<std::string, float>& config
) : config_(config),
    max_feedback_history_(1000),
    total_assessments_(0),
    cumulative_accuracy_(0.0f),
    last_update_(std::chrono::steady_clock::now())
{
    // Initialize adaptive thresholds for each time scale
    adaptive_thresholds_[TimeScale::IMMEDIATE] = {
        0.15f,  // base_threshold - low for fast sensory processing
        0.15f,  // current_threshold
        0.05f,  // sensitivity
        0.05f,  // min_threshold
        0.40f,  // max_threshold
        50      // adaptation_window
    };
    
    adaptive_thresholds_[TimeScale::SHORT_TERM] = {
        0.25f, 0.25f, 0.08f, 0.10f, 0.50f, 100
    };
    
    adaptive_thresholds_[TimeScale::MEDIUM_TERM] = {
        0.35f, 0.35f, 0.10f, 0.15f, 0.60f, 200
    };
    
    adaptive_thresholds_[TimeScale::LONG_TERM] = {
        0.45f, 0.45f, 0.12f, 0.20f, 0.70f, 500
    };
    
    adaptive_thresholds_[TimeScale::HISTORICAL] = {
        0.55f, 0.55f, 0.15f, 0.30f, 0.80f, 1000
    };
    
    // Apply config overrides
    if (config_.find("max_feedback_history") != config_.end()) {
        max_feedback_history_ = static_cast<size_t>(config_["max_feedback_history"]);
    }
}

MultiScaleRelevance RelevanceRealizationSystem::assess_multi_scale_relevance(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context,
    const std::vector<std::vector<float>>& goals
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    MultiScaleRelevance result;
    
    // Assess relevance across time scales
    result.time_scale_scores[TimeScale::IMMEDIATE] = 
        compute_immediate_relevance(input, context);
    
    result.time_scale_scores[TimeScale::SHORT_TERM] = 
        compute_short_term_relevance(input, context);
    
    result.time_scale_scores[TimeScale::MEDIUM_TERM] = 
        compute_medium_term_relevance(input, goals);
    
    result.time_scale_scores[TimeScale::LONG_TERM] = 
        compute_long_term_relevance(input, goals);
    
    result.time_scale_scores[TimeScale::HISTORICAL] = 
        compute_historical_relevance(input);
    
    // Assess relevance across cognitive levels
    result.cognitive_scores[CognitiveLevel::SENSORY] = 
        compute_sensory_relevance(input);
    
    result.cognitive_scores[CognitiveLevel::PERCEPTUAL] = 
        compute_perceptual_relevance(input);
    
    result.cognitive_scores[CognitiveLevel::CONCEPTUAL] = 
        compute_conceptual_relevance(input);
    
    result.cognitive_scores[CognitiveLevel::GOAL_ORIENTED] = 
        compute_goal_relevance(input, goals);
    
    result.cognitive_scores[CognitiveLevel::WISDOM] = 
        compute_wisdom_relevance(input);
    
    // Integrate scores across scales
    result.integrated_relevance = integrate_multi_scale_scores(
        result.time_scale_scores,
        result.cognitive_scores
    );
    
    // Compute confidence based on consistency across scales
    float variance = 0.0f;
    float mean_score = 0.0f;
    size_t count = 0;
    
    for (const auto& [scale, score] : result.time_scale_scores) {
        mean_score += score;
        ++count;
    }
    for (const auto& [level, score] : result.cognitive_scores) {
        mean_score += score;
        ++count;
    }
    mean_score /= count;
    
    for (const auto& [scale, score] : result.time_scale_scores) {
        float diff = score - mean_score;
        variance += diff * diff;
    }
    for (const auto& [level, score] : result.cognitive_scores) {
        float diff = score - mean_score;
        variance += diff * diff;
    }
    variance /= count;
    
    // Low variance = high consistency = high confidence
    result.confidence = std::exp(-variance * 5.0f);
    
    // Identify critical features
    identify_critical_features(input, result.critical_features);
    
    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    ++total_assessments_;
    
    return result;
}

AdaptiveThreshold RelevanceRealizationSystem::get_adaptive_threshold(TimeScale scale) const {
    auto it = adaptive_thresholds_.find(scale);
    if (it != adaptive_thresholds_.end()) {
        return it->second;
    }
    // Return default if not found
    return {0.3f, 0.3f, 0.1f, 0.1f, 0.6f, 100};
}

void RelevanceRealizationSystem::update_adaptive_threshold(
    TimeScale scale,
    const std::vector<float>& performance
) {
    auto it = adaptive_thresholds_.find(scale);
    if (it != adaptive_thresholds_.end()) {
        adapt_threshold(it->second, performance);
    }
}

AttentionDirective RelevanceRealizationSystem::guide_attention(
    const MultiScaleRelevance& relevance,
    float attention_budget
) {
    AttentionDirective directive;
    
    // Weight features by integrated relevance and critical feature status
    size_t num_features = relevance.critical_features.size() > 0 ?
        *std::max_element(relevance.critical_features.begin(), 
                         relevance.critical_features.end()) + 1 : 0;
    
    directive.focus_weights.resize(num_features, 0.1f);
    
    // Boost critical features (with bounds checking)
    for (size_t idx : relevance.critical_features) {
        if (idx < directive.focus_weights.size()) {
            directive.focus_weights[idx] = 1.0f;
        }
    }
    
    // Apply time-scale weighting (immediate > short > medium > long > historical)
    float immediate_weight = relevance.time_scale_scores.at(TimeScale::IMMEDIATE);
    float short_weight = relevance.time_scale_scores.at(TimeScale::SHORT_TERM);
    
    // Modulate by temporal urgency
    float temporal_urgency = immediate_weight * 0.5f + short_weight * 0.3f;
    
    for (float& w : directive.focus_weights) {
        w *= (1.0f + temporal_urgency);
    }
    
    // Normalize to attention budget
    float sum = std::accumulate(directive.focus_weights.begin(), 
                                directive.focus_weights.end(), 0.0f);
    if (sum > 0.0f) {
        for (float& w : directive.focus_weights) {
            w = (w / sum) * attention_budget;
        }
    }
    
    // Create priority ordering
    std::vector<std::pair<size_t, float>> indexed_weights;
    for (size_t i = 0; i < directive.focus_weights.size(); ++i) {
        indexed_weights.push_back({i, directive.focus_weights[i]});
    }
    std::sort(indexed_weights.begin(), indexed_weights.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (const auto& [idx, weight] : indexed_weights) {
        directive.priority_indices.push_back(idx);
    }
    
    directive.total_allocation = std::min(attention_budget, 
                                         std::accumulate(directive.focus_weights.begin(), 
                                                        directive.focus_weights.end(), 0.0f));
    
    // Compute focus sharpness (concentration of attention)
    if (!directive.focus_weights.empty()) {
        float max_weight = *std::max_element(directive.focus_weights.begin(),
                                            directive.focus_weights.end());
        float avg_weight = directive.total_allocation / directive.focus_weights.size();
        directive.focus_sharpness = std::max(1.0f, max_weight / std::max(0.001f, avg_weight));
    } else {
        directive.focus_sharpness = 1.0f;
    }
    
    return directive;
}

MemoryGuidance RelevanceRealizationSystem::guide_memory_retrieval(
    const std::vector<float>& query,
    const MultiScaleRelevance& relevance
) {
    MemoryGuidance guidance;
    
    // Generate retrieval cues based on critical features
    guidance.retrieval_cues = query;
    
    // Boost cues for critical features
    for (size_t idx : relevance.critical_features) {
        if (idx < guidance.retrieval_cues.size()) {
            guidance.retrieval_cues[idx] *= 2.0f;
        }
    }
    
    // Determine temporal bias (recency vs relevance)
    float immediate = relevance.time_scale_scores.at(TimeScale::IMMEDIATE);
    float historical = relevance.time_scale_scores.at(TimeScale::HISTORICAL);
    
    // Higher immediate = more recency bias; higher historical = more relevance bias
    guidance.temporal_bias = immediate / std::max(0.001f, immediate + historical);
    
    // Compute associative strength from cognitive scores
    float perceptual = relevance.cognitive_scores.at(CognitiveLevel::PERCEPTUAL);
    float conceptual = relevance.cognitive_scores.at(CognitiveLevel::CONCEPTUAL);
    guidance.associative_strength = (perceptual + conceptual) / 2.0f;
    
    // Relevant episodes (simulated - would integrate with episodic memory)
    // Use integrated relevance to determine how many episodes to retrieve
    size_t num_episodes = static_cast<size_t>(
        relevance.integrated_relevance * 10.0f + 1.0f
    );
    for (size_t i = 0; i < num_episodes && i < 20; ++i) {
        guidance.relevant_episodes.push_back(i);
    }
    
    return guidance;
}

IntegrationPriority RelevanceRealizationSystem::prioritize_knowledge_integration(
    const std::vector<std::vector<float>>& knowledge_items,
    const std::vector<float>& current_knowledge,
    const MultiScaleRelevance& relevance
) {
    IntegrationPriority priority;
    
    // Assess relevance of each knowledge item
    for (const auto& item : knowledge_items) {
        if (item.empty() || current_knowledge.empty()) {
            // Empty items or knowledge get default mid-range relevance
            priority.knowledge_relevance.push_back(0.5f);
            continue;
        }
        
        // Compute similarity to current knowledge (lower = more novel)
        float similarity = 0.0f;
        size_t min_size = std::min(item.size(), current_knowledge.size());
        if (min_size == 0) {
            priority.knowledge_relevance.push_back(0.5f);
            continue;
        }
        
        for (size_t i = 0; i < min_size; ++i) {
            similarity += item[i] * current_knowledge[i];
        }
        similarity /= min_size;
        
        // Novelty score
        float novelty = 1.0f - similarity;
        
        // Relevance score combines novelty with multi-scale relevance
        float item_relevance = novelty * 0.4f + relevance.integrated_relevance * 0.6f;
        priority.knowledge_relevance.push_back(item_relevance);
    }
    
    // Create integration order
    std::vector<std::pair<size_t, float>> indexed_relevance;
    for (size_t i = 0; i < priority.knowledge_relevance.size(); ++i) {
        indexed_relevance.push_back({i, priority.knowledge_relevance[i]});
    }
    std::sort(indexed_relevance.begin(), indexed_relevance.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (const auto& [idx, rel] : indexed_relevance) {
        priority.integration_order.push_back(idx);
    }
    
    // Coherence requirement from wisdom level
    float wisdom = relevance.cognitive_scores.at(CognitiveLevel::WISDOM);
    priority.coherence_requirement = wisdom;
    
    // Novelty weight from perceptual/conceptual balance
    float perceptual = relevance.cognitive_scores.at(CognitiveLevel::PERCEPTUAL);
    float conceptual = relevance.cognitive_scores.at(CognitiveLevel::CONCEPTUAL);
    priority.novelty_weight = perceptual / std::max(0.001f, perceptual + conceptual);
    
    return priority;
}

void RelevanceRealizationSystem::provide_learning_feedback(
    const RelevanceLearningFeedback& feedback
) {
    feedback_history_.push_back(feedback);
    
    // Maintain history limit
    if (feedback_history_.size() > max_feedback_history_) {
        feedback_history_.erase(feedback_history_.begin());
    }
    
    // Update cumulative accuracy
    float abs_error = std::abs(feedback.error);
    float accuracy = 1.0f - std::min(1.0f, abs_error);
    cumulative_accuracy_ = (cumulative_accuracy_ * (total_assessments_ - 1) + accuracy) 
                          / total_assessments_;
}

std::unordered_map<std::string, float> RelevanceRealizationSystem::learn_from_feedback() {
    std::unordered_map<std::string, float> metrics;
    
    if (feedback_history_.empty()) {
        metrics["accuracy"] = 0.0f;
        metrics["error_reduction"] = 0.0f;
        metrics["feedback_count"] = 0.0f;
        return metrics;
    }
    
    // Compute recent accuracy
    float recent_accuracy = 0.0f;
    size_t recent_count = std::min(static_cast<size_t>(100), feedback_history_.size());
    for (size_t i = feedback_history_.size() - recent_count; i < feedback_history_.size(); ++i) {
        float abs_error = std::abs(feedback_history_[i].error);
        recent_accuracy += 1.0f - std::min(1.0f, abs_error);
    }
    recent_accuracy /= recent_count;
    
    metrics["accuracy"] = recent_accuracy;
    metrics["cumulative_accuracy"] = cumulative_accuracy_;
    metrics["feedback_count"] = static_cast<float>(feedback_history_.size());
    
    // Analyze error patterns per time scale
    std::unordered_map<TimeScale, std::vector<float>> scale_errors;
    for (const auto& fb : feedback_history_) {
        scale_errors[fb.time_scale].push_back(std::abs(fb.error));
    }
    
    // Adapt thresholds based on errors
    for (auto& [scale, threshold] : adaptive_thresholds_) {
        if (scale_errors.find(scale) != scale_errors.end()) {
            adapt_threshold(threshold, scale_errors[scale]);
        }
    }
    
    // Compute error reduction over time
    if (feedback_history_.size() >= 20) {
        float early_error = 0.0f;
        float late_error = 0.0f;
        size_t window = std::min(static_cast<size_t>(10), feedback_history_.size() / 2);
        
        for (size_t i = 0; i < window; ++i) {
            early_error += std::abs(feedback_history_[i].error);
        }
        for (size_t i = feedback_history_.size() - window; i < feedback_history_.size(); ++i) {
            late_error += std::abs(feedback_history_[i].error);
        }
        
        early_error /= window;
        late_error /= window;
        
        float error_reduction = (early_error - late_error) / std::max(0.001f, early_error);
        metrics["error_reduction"] = error_reduction;
    } else {
        metrics["error_reduction"] = 0.0f;
    }
    
    update_learning_from_feedback();
    
    return metrics;
}

void RelevanceRealizationSystem::propagate_relevance(const RelevancePropagation& propagation) {
    // Store propagated relevance
    propagated_relevance_[propagation.source_module] = 
        propagation.relevance_value * propagation.propagation_strength;
    
    // Could implement more sophisticated propagation (e.g., decay over time)
}

float RelevanceRealizationSystem::get_propagated_relevance(const std::string& source_module) const {
    auto it = propagated_relevance_.find(source_module);
    return (it != propagated_relevance_.end()) ? it->second : 0.0f;
}

ActionCoupling RelevanceRealizationSystem::couple_to_action(
    const std::vector<float>& state,
    const std::vector<std::vector<float>>& available_actions,
    const MultiScaleRelevance& relevance
) {
    ActionCoupling coupling;
    
    // Assess relevance of each action
    for (const auto& action : available_actions) {
        if (state.empty() || action.empty()) {
            coupling.action_relevance.push_back(0.0f);
            continue;
        }
        
        // Compute expected state-action relevance
        float action_relevance = 0.0f;
        size_t min_size = std::min(state.size(), action.size());
        if (min_size == 0) {
            coupling.action_relevance.push_back(0.0f);
            continue;
        }
        
        for (size_t i = 0; i < min_size; ++i) {
            action_relevance += state[i] * action[i];
        }
        action_relevance /= min_size;
        
        // Modulate by current relevance assessment
        action_relevance *= relevance.integrated_relevance;
        
        coupling.action_relevance.push_back(action_relevance);
    }
    
    // Find recommended action
    auto max_it = std::max_element(coupling.action_relevance.begin(), 
                                   coupling.action_relevance.end());
    if (max_it != coupling.action_relevance.end()) {
        coupling.recommended_action = std::distance(coupling.action_relevance.begin(), max_it);
    } else {
        coupling.recommended_action = 0;
    }
    
    // Compute urgency from immediate time scale
    coupling.urgency = relevance.time_scale_scores.at(TimeScale::IMMEDIATE);
    
    // Expected impact from goal-oriented cognitive level
    coupling.expected_impact = relevance.cognitive_scores.at(CognitiveLevel::GOAL_ORIENTED);
    
    // Confidence from multi-scale confidence
    coupling.confidence = relevance.confidence;
    
    return coupling;
}

void RelevanceRealizationSystem::update_config(
    const std::unordered_map<std::string, float>& config
) {
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
    
    if (config_.find("max_feedback_history") != config_.end()) {
        max_feedback_history_ = static_cast<size_t>(config_["max_feedback_history"]);
    }
}

std::unordered_map<std::string, float> RelevanceRealizationSystem::get_metrics() const {
    return {
        {"total_assessments", static_cast<float>(total_assessments_)},
        {"cumulative_accuracy", cumulative_accuracy_},
        {"feedback_history_size", static_cast<float>(feedback_history_.size())},
        {"propagated_modules", static_cast<float>(propagated_relevance_.size())}
    };
}

void RelevanceRealizationSystem::reset() {
    feedback_history_.clear();
    propagated_relevance_.clear();
    total_assessments_ = 0;
    cumulative_accuracy_ = 0.0f;
    last_update_ = std::chrono::steady_clock::now();
    
    // Reset adaptive thresholds to base values
    for (auto& [scale, threshold] : adaptive_thresholds_) {
        threshold.current_threshold = threshold.base_threshold;
    }
}

// Private implementation methods

float RelevanceRealizationSystem::compute_immediate_relevance(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Immediate relevance: salience-based, bottom-up
    if (input.empty()) return 0.0f;
    
    float max_val = *std::max_element(input.begin(), input.end());
    float avg_val = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    
    // High contrast = high immediate relevance
    float contrast = max_val - avg_val;
    
    // Context modulation (e.g., threat increases immediate relevance)
    float context_boost = 1.0f;
    if (context.find("threat_level") != context.end()) {
        context_boost += context.at("threat_level") * 0.5f;
    }
    
    return std::min(1.0f, contrast * context_boost);
}

float RelevanceRealizationSystem::compute_short_term_relevance(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Short-term relevance: working memory maintenance
    if (input.empty()) return 0.0f;
    
    float magnitude = 0.0f;
    for (float val : input) {
        magnitude += val * val;
    }
    magnitude = std::sqrt(magnitude / input.size());
    
    // Context: task relevance
    if (context.find("task_priority") != context.end()) {
        magnitude *= (1.0f + context.at("task_priority") * 0.3f);
    }
    
    return std::min(1.0f, magnitude);
}

float RelevanceRealizationSystem::compute_medium_term_relevance(
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& goals
) {
    // Medium-term relevance: alignment with active goals
    if (input.empty() || goals.empty()) return 0.0f;
    
    float max_alignment = 0.0f;
    for (const auto& goal : goals) {
        if (goal.empty()) continue;
        
        float alignment = 0.0f;
        size_t min_size = std::min(input.size(), goal.size());
        if (min_size == 0) continue;
        
        for (size_t i = 0; i < min_size; ++i) {
            alignment += input[i] * goal[i];
        }
        alignment /= min_size;
        max_alignment = std::max(max_alignment, alignment);
    }
    
    return std::min(1.0f, max_alignment);
}

float RelevanceRealizationSystem::compute_long_term_relevance(
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& goals
) {
    // Long-term relevance: strategic value for multiple goals
    if (input.empty() || goals.empty()) return 0.0f;
    
    float cumulative_alignment = 0.0f;
    size_t valid_goals = 0;
    for (const auto& goal : goals) {
        if (goal.empty()) continue;
        
        float alignment = 0.0f;
        size_t min_size = std::min(input.size(), goal.size());
        if (min_size == 0) continue;
        
        for (size_t i = 0; i < min_size; ++i) {
            alignment += input[i] * goal[i];
        }
        alignment /= min_size;
        cumulative_alignment += alignment;
        ++valid_goals;
    }
    
    return valid_goals > 0 ? std::min(1.0f, cumulative_alignment / valid_goals) : 0.0f;
}

float RelevanceRealizationSystem::compute_historical_relevance(
    const std::vector<float>& input
) {
    // Historical relevance: learning patterns from feedback history
    if (input.empty() || feedback_history_.empty()) return 0.0f;
    
    float historical_score = 0.0f;
    size_t count = 0;
    
    // Compare with historical contexts that had high actual relevance
    for (const auto& feedback : feedback_history_) {
        if (feedback.actual_relevance > 0.6f && !feedback.context_state.empty()) {
            float similarity = 0.0f;
            size_t min_size = std::min(input.size(), feedback.context_state.size());
            if (min_size == 0) continue;
            
            for (size_t i = 0; i < min_size; ++i) {
                similarity += input[i] * feedback.context_state[i];
            }
            similarity /= min_size;
            historical_score += similarity * feedback.actual_relevance;
            ++count;
        }
    }
    
    return count > 0 ? std::min(1.0f, historical_score / count) : 0.5f;
}

float RelevanceRealizationSystem::compute_sensory_relevance(const std::vector<float>& input) {
    // Sensory level: raw signal strength
    if (input.empty()) return 0.0f;
    
    float energy = 0.0f;
    for (float val : input) {
        energy += std::abs(val);
    }
    return std::min(1.0f, energy / input.size());
}

float RelevanceRealizationSystem::compute_perceptual_relevance(const std::vector<float>& input) {
    // Perceptual level: pattern distinctiveness
    if (input.empty()) return 0.0f;
    
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float variance = 0.0f;
    for (float val : input) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= input.size();
    
    // Higher variance = more distinctive pattern
    return std::min(1.0f, std::sqrt(variance) * 2.0f);
}

float RelevanceRealizationSystem::compute_conceptual_relevance(const std::vector<float>& input) {
    // Conceptual level: abstraction and categorization
    if (input.empty()) return 0.0f;
    
    // Simplified: measure information content (entropy-like)
    std::vector<float> normalized = input;
    float sum = std::accumulate(normalized.begin(), normalized.end(), 0.0f);
    if (sum > 0.0f) {
        for (float& val : normalized) {
            val /= sum;
        }
    } else {
        // Handle zero or negative sum
        return 0.0f;
    }
    
    float entropy = 0.0f;
    for (float p : normalized) {
        if (p > 0.0f) {
            entropy -= p * std::log(p);
        }
    }
    
    // Normalize entropy to [0, 1]
    float max_entropy = std::log(static_cast<float>(input.size()));
    return max_entropy > 0.0f ? std::min(1.0f, entropy / max_entropy) : 0.0f;
}

float RelevanceRealizationSystem::compute_goal_relevance(
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& goals
) {
    // Goal-oriented level: alignment with explicit goals
    return compute_medium_term_relevance(input, goals);
}

float RelevanceRealizationSystem::compute_wisdom_relevance(const std::vector<float>& input) {
    // Wisdom level: meta-cognitive coherence and meaning-making
    if (input.empty()) return 0.0f;
    
    // Measure balance and moderation (Aristotelian wisdom)
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    
    // Values close to mean = balanced = wise
    float balance_score = 0.0f;
    for (float val : input) {
        float deviation = std::abs(val - mean);
        balance_score += std::exp(-deviation * 2.0f);
    }
    balance_score /= input.size();
    
    return std::min(1.0f, balance_score);
}

void RelevanceRealizationSystem::identify_critical_features(
    const std::vector<float>& input,
    std::vector<size_t>& critical_features
) {
    critical_features.clear();
    
    if (input.empty()) return;
    
    // Features above 75th percentile are critical
    std::vector<float> sorted_input = input;
    std::sort(sorted_input.begin(), sorted_input.end());
    
    size_t percentile_75_idx = static_cast<size_t>(sorted_input.size() * 0.75f);
    float threshold = sorted_input[percentile_75_idx];
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] >= threshold) {
            critical_features.push_back(i);
        }
    }
}

void RelevanceRealizationSystem::adapt_threshold(
    AdaptiveThreshold& threshold,
    const std::vector<float>& performance
) {
    if (performance.empty()) return;
    
    // Compute recent average performance
    size_t window = std::min(threshold.adaptation_window, performance.size());
    float avg_performance = 0.0f;
    for (size_t i = performance.size() - window; i < performance.size(); ++i) {
        avg_performance += performance[i];
    }
    avg_performance /= window;
    
    // If performance is high (low error), can raise threshold (be more selective)
    // If performance is low (high error), lower threshold (be less selective)
    float target_adjustment = (0.5f - avg_performance) * threshold.sensitivity;
    
    threshold.current_threshold += target_adjustment;
    threshold.current_threshold = std::max(threshold.min_threshold,
                                          std::min(threshold.max_threshold,
                                                  threshold.current_threshold));
}

float RelevanceRealizationSystem::integrate_multi_scale_scores(
    const std::unordered_map<TimeScale, float>& time_scores,
    const std::unordered_map<CognitiveLevel, float>& cognitive_scores
) {
    // Weighted integration of multi-scale scores
    
    // Time scale weights: immediate matters most, then short, then medium, etc.
    float time_weighted = 0.0f;
    time_weighted += time_scores.at(TimeScale::IMMEDIATE) * 0.30f;
    time_weighted += time_scores.at(TimeScale::SHORT_TERM) * 0.25f;
    time_weighted += time_scores.at(TimeScale::MEDIUM_TERM) * 0.20f;
    time_weighted += time_scores.at(TimeScale::LONG_TERM) * 0.15f;
    time_weighted += time_scores.at(TimeScale::HISTORICAL) * 0.10f;
    
    // Cognitive level weights: goal-oriented and conceptual most important
    float cognitive_weighted = 0.0f;
    cognitive_weighted += cognitive_scores.at(CognitiveLevel::SENSORY) * 0.10f;
    cognitive_weighted += cognitive_scores.at(CognitiveLevel::PERCEPTUAL) * 0.15f;
    cognitive_weighted += cognitive_scores.at(CognitiveLevel::CONCEPTUAL) * 0.25f;
    cognitive_weighted += cognitive_scores.at(CognitiveLevel::GOAL_ORIENTED) * 0.35f;
    cognitive_weighted += cognitive_scores.at(CognitiveLevel::WISDOM) * 0.15f;
    
    // Combine time and cognitive dimensions
    return (time_weighted * 0.6f + cognitive_weighted * 0.4f);
}

void RelevanceRealizationSystem::update_learning_from_feedback() {
    // Additional learning updates could be implemented here
    // e.g., update internal models, adjust weighting schemes, etc.
    last_update_ = std::chrono::steady_clock::now();
}

} // namespace rros

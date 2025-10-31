#include "historical_context.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace rros {

HistoricalContextSystem::HistoricalContextSystem(
    std::shared_ptr<TemporalKnowledge> temporal_knowledge,
    std::shared_ptr<EpisodicMemorySystem> episodic_memory
) : temporal_knowledge_(temporal_knowledge),
    episodic_memory_(episodic_memory),
    baseline_decision_quality_(0.5f)
{
    learned_patterns_.reserve(1000);
    inference_rules_.reserve(500);
    decision_quality_history_.reserve(1000);
}

std::vector<HistoricalPattern> HistoricalContextSystem::detect_patterns(
    const std::vector<size_t>& event_ids,
    float min_support,
    float min_confidence
) {
    std::vector<HistoricalPattern> detected_patterns;
    
    if (event_ids.size() < 2) return detected_patterns;
    
    // Sliding window pattern detection
    for (size_t window_size = 2; window_size <= std::min(size_t(5), event_ids.size()); ++window_size) {
        for (size_t i = 0; i + window_size <= event_ids.size(); ++i) {
            std::vector<size_t> subsequence(
                event_ids.begin() + i,
                event_ids.begin() + i + window_size
            );
            
            // Check if this pattern is frequent
            size_t occurrence_count = 0;
            for (size_t j = 0; j + window_size <= event_ids.size(); ++j) {
                bool matches = true;
                for (size_t k = 0; k < window_size; ++k) {
                    // Check temporal and content similarity
                    const auto& event1 = temporal_knowledge_->get_event(event_ids[j + k]);
                    const auto& event2 = temporal_knowledge_->get_event(subsequence[k]);
                    
                    // Simple similarity check (can be enhanced)
                    float similarity = 0.0f;
                    if (!event1.state_vector.empty() && !event2.state_vector.empty()) {
                        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
                        size_t min_size = std::min(event1.state_vector.size(), 
                                                   event2.state_vector.size());
                        for (size_t m = 0; m < min_size; ++m) {
                            dot += event1.state_vector[m] * event2.state_vector[m];
                            norm1 += event1.state_vector[m] * event1.state_vector[m];
                            norm2 += event2.state_vector[m] * event2.state_vector[m];
                        }
                        if (norm1 > 0 && norm2 > 0) {
                            similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));
                        }
                    }
                    
                    if (similarity < 0.8f) {
                        matches = false;
                        break;
                    }
                }
                if (matches) occurrence_count++;
            }
            
            float support = static_cast<float>(occurrence_count) / 
                           static_cast<float>(event_ids.size() - window_size + 1);
            
            if (support >= min_support && occurrence_count >= 2) {
                HistoricalPattern pattern;
                pattern.event_sequence = subsequence;
                pattern.support = support;
                pattern.confidence = std::min(1.0f, support * 2.0f);
                pattern.occurrence_count = occurrence_count;
                pattern.pattern_type = "sequential";
                
                // Extract pattern signature
                pattern.pattern_signature = extract_pattern_features(subsequence);
                
                if (pattern.confidence >= min_confidence) {
                    detected_patterns.push_back(pattern);
                }
            }
        }
    }
    
    // Merge learned patterns
    for (auto& pattern : detected_patterns) {
        learned_patterns_.push_back(pattern);
    }
    
    return detected_patterns;
}

size_t HistoricalContextSystem::learn_temporal_rules(
    size_t event_window,
    float min_support
) {
    size_t rules_learned = 0;
    
    auto stats = temporal_knowledge_->get_statistics();
    size_t total_events = static_cast<size_t>(stats["event_count"]);
    
    if (total_events < 10) return 0;
    
    size_t window = std::min(event_window, total_events);
    
    // Analyze event sequences for if-then rules
    for (size_t i = 0; i + 3 < window; ++i) {
        try {
            const auto& event1 = temporal_knowledge_->get_event(i);
            const auto& event2 = temporal_knowledge_->get_event(i + 1);
            const auto& event3 = temporal_knowledge_->get_event(i + 2);
            
            // Create rule: if event1 and event2, then event3
            TemporalRule rule;
            rule.antecedent_events = {i, i + 1};
            rule.consequent_events = {i + 2};
            rule.support_count = 1;
            
            // Calculate typical delay
            rule.typical_delay = event3.timestamp - event2.timestamp;
            
            // Validate rule across other sequences
            size_t support_count = 0;
            size_t total_opportunities = 0;
            
            for (size_t j = 0; j + 3 < window; ++j) {
                if (j == i) continue;
                
                const auto& test1 = temporal_knowledge_->get_event(j);
                const auto& test2 = temporal_knowledge_->get_event(j + 1);
                const auto& test3 = temporal_knowledge_->get_event(j + 2);
                
                // Check if antecedent pattern matches
                bool antecedent_match = true;
                // Simplified matching - in practice, use more sophisticated similarity
                
                if (antecedent_match) {
                    total_opportunities++;
                    // Check if consequent follows
                    bool consequent_match = true; // Simplified
                    if (consequent_match) {
                        support_count++;
                    }
                }
            }
            
            if (total_opportunities > 0) {
                rule.confidence = static_cast<float>(support_count) / 
                                 static_cast<float>(total_opportunities);
                rule.support_count = support_count;
                
                float support = static_cast<float>(support_count) / 
                               static_cast<float>(window);
                
                if (support >= min_support && rule.confidence >= 0.7f) {
                    inference_rules_.push_back(rule);
                    rules_learned++;
                }
            }
        } catch (const std::exception&) {
            continue;
        }
    }
    
    return rules_learned;
}

std::vector<CausalRelation> HistoricalContextSystem::detect_causal_relations(
    uint64_t lookback_time,
    float min_correlation
) {
    std::vector<CausalRelation> causal_relations;
    
    auto stats = temporal_knowledge_->get_statistics();
    size_t total_events = static_cast<size_t>(stats["event_count"]);
    
    if (total_events < 2) return causal_relations;
    
    // Analyze pairs of events for potential causal relationships
    for (size_t i = 0; i < total_events; ++i) {
        try {
            const auto& event_i = temporal_knowledge_->get_event(i);
            
            for (size_t j = i + 1; j < total_events; ++j) {
                const auto& event_j = temporal_knowledge_->get_event(j);
                
                uint64_t time_diff = event_j.timestamp - event_i.timestamp;
                
                if (time_diff > lookback_time) break;
                
                // Estimate causal strength
                float causal_strength = estimate_causal_strength(i, j, lookback_time);
                
                if (causal_strength >= min_correlation) {
                    CausalRelation relation;
                    relation.cause_event = i;
                    relation.effect_event = j;
                    relation.strength = causal_strength;
                    relation.time_lag = time_diff;
                    relation.confidence = causal_strength;
                    
                    causal_relations.push_back(relation);
                    
                    // Add to temporal knowledge
                    temporal_knowledge_->add_causal_relation(
                        i, j, causal_strength, causal_strength
                    );
                }
            }
        } catch (const std::exception&) {
            continue;
        }
    }
    
    return causal_relations;
}

std::vector<std::pair<std::vector<float>, float>> 
HistoricalContextSystem::predict_future(
    const std::vector<float>& current_context,
    uint64_t horizon_time
) {
    std::vector<std::pair<std::vector<float>, float>> predictions;
    
    // Find similar historical patterns
    auto similar_patterns = query_patterns(current_context, 10);
    
    for (const auto& pattern : similar_patterns) {
        if (pattern.event_sequence.empty()) continue;
        
        try {
            // Use the next event in pattern as prediction
            size_t last_event_id = pattern.event_sequence.back();
            const auto& last_event = temporal_knowledge_->get_event(last_event_id);
            
            // Predict based on pattern continuation
            std::vector<float> prediction = last_event.state_vector;
            float confidence = pattern.confidence * pattern.support;
            
            predictions.emplace_back(prediction, confidence);
        } catch (const std::exception&) {
            continue;
        }
    }
    
    // Sort by confidence
    std::sort(predictions.begin(), predictions.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top predictions
    if (predictions.size() > 5) {
        predictions.resize(5);
    }
    
    return predictions;
}

std::vector<float> HistoricalContextSystem::make_historical_decision(
    const std::vector<float>& decision_context,
    const std::vector<std::vector<float>>& alternatives
) {
    std::vector<float> scores(alternatives.size(), 0.0f);
    
    // Retrieve similar past decisions from episodic memory
    auto similar_memories = episodic_memory_->retrieve_by_content(decision_context, 20);
    
    // Score each alternative based on historical outcomes
    for (size_t i = 0; i < alternatives.size(); ++i) {
        float historical_score = 0.0f;
        float total_weight = 0.0f;
        
        for (size_t memory_id : similar_memories) {
            try {
                const auto& memory = episodic_memory_->get_memory(memory_id);
                
                // Compute similarity to alternative
                float similarity = 0.0f;
                if (!memory.content.empty() && !alternatives[i].empty()) {
                    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
                    size_t min_size = std::min(memory.content.size(), alternatives[i].size());
                    for (size_t j = 0; j < min_size; ++j) {
                        dot += memory.content[j] * alternatives[i][j];
                        norm1 += memory.content[j] * memory.content[j];
                        norm2 += alternatives[i][j] * alternatives[i][j];
                    }
                    if (norm1 > 0 && norm2 > 0) {
                        similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));
                    }
                }
                
                // Weight by memory vividness and consolidation
                float weight = memory.vividness * memory.consolidation_level;
                historical_score += similarity * weight;
                total_weight += weight;
            } catch (const std::exception&) {
                continue;
            }
        }
        
        scores[i] = (total_weight > 0) ? (historical_score / total_weight) : 0.5f;
    }
    
    // Normalize scores
    float max_score = *std::max_element(scores.begin(), scores.end());
    if (max_score > 0) {
        for (float& score : scores) {
            score /= max_score;
        }
    }
    
    return scores;
}

float HistoricalContextSystem::validate_consistency() {
    float consistency_score = 1.0f;
    size_t inconsistencies = 0;
    size_t total_checks = 0;
    
    // Check causal consistency
    auto stats = temporal_knowledge_->get_statistics();
    size_t total_events = static_cast<size_t>(stats["event_count"]);
    
    for (size_t i = 0; i < total_events; ++i) {
        try {
            auto predecessors = temporal_knowledge_->get_causal_predecessors(i);
            auto successors = temporal_knowledge_->get_causal_successors(i);
            
            // Check for causal loops
            std::set<size_t> visited;
            std::vector<size_t> stack = {i};
            
            while (!stack.empty()) {
                size_t current = stack.back();
                stack.pop_back();
                
                if (visited.count(current)) {
                    inconsistencies++; // Found a loop
                    break;
                }
                visited.insert(current);
                
                auto next_successors = temporal_knowledge_->get_causal_successors(current);
                for (const auto& rel : next_successors) {
                    if (rel.effect_event == i) {
                        inconsistencies++; // Circular causality
                    }
                }
            }
            
            total_checks++;
        } catch (const std::exception&) {
            continue;
        }
    }
    
    if (total_checks > 0) {
        consistency_score = 1.0f - (static_cast<float>(inconsistencies) / 
                                     static_cast<float>(total_checks));
    }
    
    return std::max(0.0f, consistency_score);
}

std::vector<HistoricalPattern> HistoricalContextSystem::query_patterns(
    const std::vector<float>& query,
    size_t max_results
) {
    std::vector<std::pair<HistoricalPattern, float>> scored_patterns;
    
    for (const auto& pattern : learned_patterns_) {
        if (pattern.pattern_signature.empty()) continue;
        
        // Compute similarity
        float similarity = 0.0f;
        if (!query.empty()) {
            float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
            size_t min_size = std::min(query.size(), pattern.pattern_signature.size());
            for (size_t i = 0; i < min_size; ++i) {
                dot += query[i] * pattern.pattern_signature[i];
                norm1 += query[i] * query[i];
                norm2 += pattern.pattern_signature[i] * pattern.pattern_signature[i];
            }
            if (norm1 > 0 && norm2 > 0) {
                similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));
            }
        }
        
        scored_patterns.emplace_back(pattern, similarity);
    }
    
    // Sort by similarity
    std::sort(scored_patterns.begin(), scored_patterns.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract top results
    std::vector<HistoricalPattern> results;
    size_t count = std::min(max_results, scored_patterns.size());
    for (size_t i = 0; i < count; ++i) {
        results.push_back(scored_patterns[i].first);
    }
    
    return results;
}

void HistoricalContextSystem::update_from_observation(
    const std::vector<float>& observation,
    const std::vector<float>& outcome,
    float actual_quality
) {
    // Track decision quality over time
    decision_quality_history_.push_back(actual_quality);
    
    if (decision_quality_history_.size() > 100) {
        decision_quality_history_.erase(decision_quality_history_.begin());
    }
    
    // Update baseline
    if (!decision_quality_history_.empty()) {
        baseline_decision_quality_ = std::accumulate(
            decision_quality_history_.begin(),
            decision_quality_history_.end(),
            0.0f
        ) / decision_quality_history_.size();
    }
    
    // Store as episodic memory
    std::vector<float> combined = observation;
    combined.insert(combined.end(), outcome.begin(), outcome.end());
    episodic_memory_->encode_memory(combined, actual_quality);
}

std::unordered_map<std::string, float> HistoricalContextSystem::get_statistics() const {
    std::unordered_map<std::string, float> stats;
    
    stats["learned_patterns"] = static_cast<float>(learned_patterns_.size());
    stats["inference_rules"] = static_cast<float>(inference_rules_.size());
    stats["decision_quality_samples"] = static_cast<float>(decision_quality_history_.size());
    stats["baseline_decision_quality"] = baseline_decision_quality_;
    
    // Calculate improvement
    if (!decision_quality_history_.empty() && decision_quality_history_.size() > 10) {
        float recent_quality = std::accumulate(
            decision_quality_history_.end() - 10,
            decision_quality_history_.end(),
            0.0f
        ) / 10.0f;
        
        float improvement = (recent_quality - baseline_decision_quality_) / 
                           (baseline_decision_quality_ + 1e-6f);
        stats["decision_quality_improvement"] = improvement * 100.0f;
    } else {
        stats["decision_quality_improvement"] = 0.0f;
    }
    
    // Average pattern confidence
    float avg_confidence = 0.0f;
    for (const auto& pattern : learned_patterns_) {
        avg_confidence += pattern.confidence;
    }
    if (!learned_patterns_.empty()) {
        avg_confidence /= learned_patterns_.size();
    }
    stats["avg_pattern_confidence"] = avg_confidence;
    
    return stats;
}

void HistoricalContextSystem::clear() {
    learned_patterns_.clear();
    inference_rules_.clear();
    decision_quality_history_.clear();
    baseline_decision_quality_ = 0.5f;
}

float HistoricalContextSystem::compute_pattern_similarity(
    const HistoricalPattern& p1,
    const HistoricalPattern& p2
) const {
    if (p1.pattern_signature.empty() || p2.pattern_signature.empty()) {
        return 0.0f;
    }
    
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    size_t min_size = std::min(p1.pattern_signature.size(), p2.pattern_signature.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        dot += p1.pattern_signature[i] * p2.pattern_signature[i];
        norm1 += p1.pattern_signature[i] * p1.pattern_signature[i];
        norm2 += p2.pattern_signature[i] * p2.pattern_signature[i];
    }
    
    if (norm1 > 0 && norm2 > 0) {
        return dot / (std::sqrt(norm1) * std::sqrt(norm2));
    }
    
    return 0.0f;
}

float HistoricalContextSystem::compute_sequence_coherence(
    const std::vector<size_t>& event_ids
) const {
    if (event_ids.size() < 2) return 1.0f;
    
    float total_coherence = 0.0f;
    size_t pairs = 0;
    
    for (size_t i = 0; i + 1 < event_ids.size(); ++i) {
        try {
            const auto& event1 = temporal_knowledge_->get_event(event_ids[i]);
            const auto& event2 = temporal_knowledge_->get_event(event_ids[i + 1]);
            
            // Compute similarity
            float similarity = 0.0f;
            if (!event1.state_vector.empty() && !event2.state_vector.empty()) {
                float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
                size_t min_size = std::min(event1.state_vector.size(), 
                                          event2.state_vector.size());
                for (size_t j = 0; j < min_size; ++j) {
                    dot += event1.state_vector[j] * event2.state_vector[j];
                    norm1 += event1.state_vector[j] * event1.state_vector[j];
                    norm2 += event2.state_vector[j] * event2.state_vector[j];
                }
                if (norm1 > 0 && norm2 > 0) {
                    similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));
                }
            }
            
            total_coherence += similarity;
            pairs++;
        } catch (const std::exception&) {
            continue;
        }
    }
    
    return (pairs > 0) ? (total_coherence / pairs) : 0.0f;
}

float HistoricalContextSystem::estimate_causal_strength(
    size_t cause_event,
    size_t effect_event,
    uint64_t time_window
) const {
    try {
        const auto& cause = temporal_knowledge_->get_event(cause_event);
        const auto& effect = temporal_knowledge_->get_event(effect_event);
        
        // Temporal precedence check
        if (effect.timestamp <= cause.timestamp) return 0.0f;
        
        // Time lag factor
        uint64_t time_lag = effect.timestamp - cause.timestamp;
        if (time_lag > time_window) return 0.0f;
        
        float temporal_factor = 1.0f - (static_cast<float>(time_lag) / 
                                        static_cast<float>(time_window));
        
        // State correlation
        float correlation = 0.0f;
        if (!cause.state_vector.empty() && !effect.state_vector.empty()) {
            float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
            size_t min_size = std::min(cause.state_vector.size(), effect.state_vector.size());
            for (size_t i = 0; i < min_size; ++i) {
                dot += cause.state_vector[i] * effect.state_vector[i];
                norm1 += cause.state_vector[i] * cause.state_vector[i];
                norm2 += effect.state_vector[i] * effect.state_vector[i];
            }
            if (norm1 > 0 && norm2 > 0) {
                correlation = dot / (std::sqrt(norm1) * std::sqrt(norm2));
            }
        }
        
        // Combined causal strength
        return 0.5f * temporal_factor + 0.5f * std::max(0.0f, correlation);
        
    } catch (const std::exception&) {
        return 0.0f;
    }
}

std::vector<float> HistoricalContextSystem::extract_pattern_features(
    const std::vector<size_t>& event_ids
) const {
    std::vector<float> features;
    
    if (event_ids.empty()) return features;
    
    // Aggregate event features
    try {
        const auto& first_event = temporal_knowledge_->get_event(event_ids[0]);
        size_t feature_dim = first_event.state_vector.size();
        features.resize(feature_dim, 0.0f);
        
        for (size_t event_id : event_ids) {
            const auto& event = temporal_knowledge_->get_event(event_id);
            for (size_t i = 0; i < feature_dim && i < event.state_vector.size(); ++i) {
                features[i] += event.state_vector[i];
            }
        }
        
        // Average
        for (float& val : features) {
            val /= static_cast<float>(event_ids.size());
        }
    } catch (const std::exception&) {
        // Return empty on error
    }
    
    return features;
}

float HistoricalContextSystem::compute_rule_confidence(
    const TemporalRule& rule,
    const std::vector<size_t>& event_sequence
) const {
    // Simplified confidence computation
    return rule.confidence;
}

} // namespace rros

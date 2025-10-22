#include "episode_processor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

EpisodeProcessor::EpisodeProcessor(const std::unordered_map<std::string, float>& config) :
    config_(config)
{
    // Initialize all episodes as active with default weight
    for (int i = 0; i <= static_cast<int>(Episode::TILLICH_BARFIELD); ++i) {
        Episode episode = static_cast<Episode>(i);
        activations_[episode] = 1.0f; // All episodes active by default
    }
    
    initialize_processors();
}

void EpisodeProcessor::initialize_processors() {
    // Initialize episode-specific processing functions
    episode_processors_[Episode::FLOW_MYSTICISM] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_flow_mysticism(input, context);
        };
    
    episode_processors_[Episode::CONTINUOUS_COSMOS] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_continuous_cosmos(input, context);
        };
    
    episode_processors_[Episode::AXIAL_REVOLUTION] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_axial_revolution(input, context);
        };
        
    episode_processors_[Episode::PLATO_CAVE] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_plato_cave(input, context);
        };
        
    episode_processors_[Episode::ARISTOTLE_WISDOM] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_aristotle_wisdom(input, context);
        };
        
    episode_processors_[Episode::MINDFULNESS_INSIGHT] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_mindfulness_insight(input, context);
        };
        
    episode_processors_[Episode::HIGHER_ORDER_THOUGHT] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_higher_order_thought(input, context);
        };
        
    episode_processors_[Episode::SELF_DECEPTION] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_self_deception(input, context);
        };
        
    episode_processors_[Episode::EMBODIED_COGNITION] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_embodied_cognition(input, context);
        };
        
    episode_processors_[Episode::RELEVANCE_REALIZATION] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_relevance_realization(input, context);
        };
        
    episode_processors_[Episode::MYSTICAL_EXPERIENCES] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_mystical_experiences(input, context);
        };
        
    episode_processors_[Episode::COGNITIVE_REVOLUTION] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_cognitive_revolution(input, context);
        };
        
    episode_processors_[Episode::SCIENTIFIC_REVOLUTION] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_scientific_revolution(input, context);
        };
        
    episode_processors_[Episode::WISDOM_CONTEMPLATION] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_wisdom_contemplation(input, context);
        };
        
    episode_processors_[Episode::INTELLIGENCE_RATIONALITY] = 
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_intelligence_rationality(input, context);
        };
        
    // Fill in remaining episodes with default processor
    for (int i = 0; i <= static_cast<int>(Episode::TILLICH_BARFIELD); ++i) {
        Episode episode = static_cast<Episode>(i);
        if (episode_processors_.find(episode) == episode_processors_.end()) {
            episode_processors_[episode] = [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
                return this->compute_default_relevance(input, context);
            };
        }
    }
}

float EpisodeProcessor::process_episode(
    Episode episode,
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    if (activations_[episode] <= 0.0f) return 0.0f;
    
    auto it = episode_processors_.find(episode);
    if (it != episode_processors_.end()) {
        float base_result = it->second(input, context);
        return base_result * activations_[episode];
    }
    
    return 0.0f;
}

float EpisodeProcessor::compute_relevance(Episode episode, const std::vector<float>& data) {
    std::unordered_map<std::string, float> empty_context;
    return process_episode(episode, data, empty_context);
}

// Episode-specific implementations

float EpisodeProcessor::process_flow_mysticism(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Flow state detection: coherence, engagement, temporal distortion
    if (input.empty()) return 0.0f;
    
    // Measure signal coherence (low frequency variation)
    float coherence = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        coherence += std::abs(input[i] - input[i-1]);
    }
    coherence = 1.0f / (1.0f + coherence / input.size());
    
    // Engagement measure (signal strength)
    float engagement = std::sqrt(std::accumulate(input.begin(), input.end(), 0.0f,
                                               [](float sum, float val) { return sum + val * val; })) / input.size();
    
    // Temporal integration (context-dependent)
    float temporal_factor = 1.0f;
    if (context.find("time_distortion") != context.end()) {
        temporal_factor = 1.0f + context.at("time_distortion");
    }
    
    return (coherence * 0.4f + engagement * 0.4f) * temporal_factor * 0.2f;
}

float EpisodeProcessor::process_continuous_cosmos(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Shamanic consciousness: pattern recognition across scales
    if (input.empty()) return 0.0f;
    
    // Multi-scale pattern detection
    std::vector<float> scales = {1.0f, 2.0f, 4.0f, 8.0f};
    float pattern_strength = 0.0f;
    
    for (float scale : scales) {
        int step = static_cast<int>(scale);
        float scale_pattern = 0.0f;
        int count = 0;
        
        for (size_t i = 0; i + step < input.size(); i += step) {
            scale_pattern += std::abs(input[i] - input[i + step]);
            ++count;
        }
        
        if (count > 0) {
            scale_pattern /= count;
            pattern_strength += 1.0f / (1.0f + scale_pattern);
        }
    }
    
    return pattern_strength / scales.size();
}

float EpisodeProcessor::process_axial_revolution(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Transformation and breakthrough detection
    if (input.size() < 3) return 0.0f;
    
    // Detect sudden changes (revolutionary moments)
    float max_change = 0.0f;
    for (size_t i = 2; i < input.size(); ++i) {
        float trend = input[i] - input[i-1];
        float prev_trend = input[i-1] - input[i-2];
        float change_magnitude = std::abs(trend - prev_trend);
        max_change = std::max(max_change, change_magnitude);
    }
    
    // Historical context amplification
    float historical_weight = 1.0f;
    if (context.find("historical_significance") != context.end()) {
        historical_weight = 1.0f + context.at("historical_significance");
    }
    
    return std::tanh(max_change * historical_weight);
}

float EpisodeProcessor::process_plato_cave(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Reality vs appearance distinction: depth detection
    if (input.empty()) return 0.0f;
    
    // Surface vs deep pattern analysis
    float surface_variation = 0.0f;
    float deep_trend = 0.0f;
    
    // Surface: high frequency changes
    for (size_t i = 1; i < input.size(); ++i) {
        surface_variation += std::abs(input[i] - input[i-1]);
    }
    surface_variation /= (input.size() - 1);
    
    // Deep: overall trend
    if (input.size() > 1) {
        deep_trend = std::abs(input.back() - input.front()) / input.size();
    }
    
    // Insight = depth relative to surface noise
    float insight_ratio = deep_trend > 0 ? deep_trend / (surface_variation + 1e-6f) : 0.0f;
    
    return std::tanh(insight_ratio);
}

float EpisodeProcessor::process_aristotle_wisdom(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Practical wisdom: balance and moderation detection
    if (input.empty()) return 0.0f;
    
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    
    // Measure balance around mean (virtue as mean)
    float balance_score = 0.0f;
    for (float val : input) {
        float distance_from_mean = std::abs(val - mean);
        balance_score += 1.0f / (1.0f + distance_from_mean);
    }
    balance_score /= input.size();
    
    // Practical context weighting
    float practical_weight = 1.0f;
    if (context.find("practical_relevance") != context.end()) {
        practical_weight = 1.0f + context.at("practical_relevance");
    }
    
    return balance_score * practical_weight;
}

float EpisodeProcessor::process_mindfulness_insight(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Present-moment awareness and insight cultivation
    if (input.empty()) return 0.0f;
    
    // Attention stability (low variance)
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float variance = 0.0f;
    for (float val : input) {
        variance += (val - mean) * (val - mean);
    }
    variance /= input.size();
    
    float stability = 1.0f / (1.0f + variance);
    
    // Present moment focus (recency bias)
    float recency_weight = 0.0f;
    float total_weight = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float weight = static_cast<float>(i + 1) / input.size(); // Linear recency
        recency_weight += input[i] * weight;
        total_weight += weight;
    }
    float present_focus = total_weight > 0 ? recency_weight / total_weight : 0.0f;
    
    return (stability * 0.6f + std::abs(present_focus) * 0.4f);
}

float EpisodeProcessor::process_higher_order_thought(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Meta-cognitive processing and self-reflection
    if (input.size() < 2) return 0.0f;
    
    // Second-order pattern detection (patterns of patterns)
    std::vector<float> first_derivatives;
    for (size_t i = 1; i < input.size(); ++i) {
        first_derivatives.push_back(input[i] - input[i-1]);
    }
    
    float second_order_variation = 0.0f;
    for (size_t i = 1; i < first_derivatives.size(); ++i) {
        second_order_variation += std::abs(first_derivatives[i] - first_derivatives[i-1]);
    }
    
    if (first_derivatives.size() > 1) {
        second_order_variation /= (first_derivatives.size() - 1);
    }
    
    // Meta-cognitive complexity
    float complexity = 1.0f / (1.0f + second_order_variation);
    
    return complexity;
}

float EpisodeProcessor::process_self_deception(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Detection of inconsistency and self-contradictory patterns
    if (input.size() < 3) return 0.0f;
    
    // Measure internal consistency
    float inconsistency = 0.0f;
    int comparisons = 0;
    
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = i + 1; j < input.size(); ++j) {
            float distance = j - i;
            float expected_similarity = std::exp(-distance * 0.1f); // Expected decay
            float actual_similarity = 1.0f - std::abs(input[i] - input[j]);
            
            inconsistency += std::abs(expected_similarity - actual_similarity);
            ++comparisons;
        }
    }
    
    if (comparisons > 0) {
        inconsistency /= comparisons;
    }
    
    return inconsistency; // Higher values indicate more self-deception
}

float EpisodeProcessor::process_embodied_cognition(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // 4E cognition: embodied, embedded, enacted, extended
    if (input.empty()) return 0.0f;
    
    // Embodiment: dynamic interaction patterns
    float dynamics = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        dynamics += std::abs(input[i] - input[i-1]);
    }
    dynamics /= (input.size() - 1);
    
    // Environmental coupling
    float environmental_coupling = 1.0f;
    if (context.find("environmental_feedback") != context.end()) {
        environmental_coupling = 1.0f + context.at("environmental_feedback");
    }
    
    return std::tanh(dynamics * environmental_coupling);
}

float EpisodeProcessor::process_relevance_realization(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Core relevance realization: adaptive constraint satisfaction
    if (input.empty()) return 0.0f;
    
    // Multi-constraint optimization
    float constraint_satisfaction = 0.0f;
    
    // Constraint 1: Information preservation
    float information_content = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        information_content += std::abs(input[i] - input[i-1]);
    }
    information_content /= (input.size() - 1);
    
    // Constraint 2: Coherence maintenance
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float coherence = 1.0f;
    for (float val : input) {
        coherence *= (1.0f - std::abs(val - mean) / (std::abs(mean) + 1.0f));
    }
    
    // Constraint 3: Adaptive flexibility
    float flexibility = std::min(1.0f, information_content);
    
    constraint_satisfaction = (information_content * 0.4f + coherence * 0.4f + flexibility * 0.2f);
    
    return constraint_satisfaction;
}

float EpisodeProcessor::process_mystical_experiences(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Mystical experience characteristics: unity, transcendence, ineffability
    if (input.empty()) return 0.0f;
    
    // Unity measure: convergence to single value
    float variance = 0.0f;
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    for (float val : input) {
        variance += (val - mean) * (val - mean);
    }
    variance /= input.size();
    
    float unity = 1.0f / (1.0f + variance);
    
    // Transcendence: movement beyond normal ranges
    float transcendence = 0.0f;
    for (float val : input) {
        if (std::abs(val) > 1.0f) { // Beyond normal range [-1, 1]
            transcendence += std::abs(val) - 1.0f;
        }
    }
    transcendence = std::min(1.0f, transcendence / input.size());
    
    return (unity * 0.7f + transcendence * 0.3f);
}

float EpisodeProcessor::process_cognitive_revolution(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Modern cognitive science principles: information processing
    if (input.empty()) return 0.0f;
    
    // Information theoretic measures
    float entropy = 0.0f;
    std::unordered_map<int, int> value_counts;
    
    // Quantize values for entropy calculation
    for (float val : input) {
        int quantized = static_cast<int>(val * 10.0f); // 10 bins
        value_counts[quantized]++;
    }
    
    for (const auto& [value, count] : value_counts) {
        float probability = static_cast<float>(count) / input.size();
        if (probability > 0) {
            entropy -= probability * std::log2(probability);
        }
    }
    
    return entropy / 4.0f; // Normalize (max entropy ≈ 4 bits for 10 bins)
}

float EpisodeProcessor::process_scientific_revolution(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Scientific method: hypothesis testing and empirical validation
    if (input.size() < 4) return 0.0f;
    
    // Test for linear trends (hypothesis)
    size_t n = input.size();
    float sum_x = n * (n - 1) / 2.0f;
    float sum_y = std::accumulate(input.begin(), input.end(), 0.0f);
    float sum_xy = 0.0f;
    float sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0f;
    
    for (size_t i = 0; i < n; ++i) {
        sum_xy += i * input[i];
    }
    
    // Correlation coefficient (empirical validation)
    float numerator = n * sum_xy - sum_x * sum_y;
    float denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * std::accumulate(input.begin(), input.end(), 0.0f,
                                 [](float sum, float val) { return sum + val * val; }) - sum_y * sum_y));
    
    float correlation = denominator != 0 ? std::abs(numerator / denominator) : 0.0f;
    
    return correlation;
}

float EpisodeProcessor::process_wisdom_contemplation(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Contemplative wisdom: deep reflection and integration
    if (input.empty()) return 0.0f;
    
    // Depth measure: long-term integration
    float integration_depth = 0.0f;
    for (size_t window = 2; window <= input.size() && window <= 8; ++window) {
        float window_coherence = 0.0f;
        for (size_t i = 0; i <= input.size() - window; ++i) {
            float window_mean = 0.0f;
            for (size_t j = i; j < i + window; ++j) {
                window_mean += input[j];
            }
            window_mean /= window;
            
            float coherence = 0.0f;
            for (size_t j = i; j < i + window; ++j) {
                coherence += 1.0f / (1.0f + std::abs(input[j] - window_mean));
            }
            window_coherence += coherence / window;
        }
        integration_depth += window_coherence * window / input.size();
    }
    
    return std::tanh(integration_depth);
}

float EpisodeProcessor::process_intelligence_rationality(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Intelligence vs rationality: optimization vs bias detection
    if (input.size() < 3) return 0.0f;
    
    // Optimization measure: improvement over time
    float improvement = 0.0f;
    for (size_t i = 2; i < input.size(); ++i) {
        float short_term = input[i] - input[i-1];
        float long_term = input[i] - input[i-2];
        if (long_term != 0) {
            improvement += short_term / std::abs(long_term);
        }
    }
    improvement /= (input.size() - 2);
    
    // Bias detection: systematic deviations
    float systematic_bias = 0.0f;
    float running_mean = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        running_mean = (running_mean * i + input[i]) / (i + 1);
        if (i > 0) {
            systematic_bias += std::abs(input[i] - running_mean);
        }
    }
    systematic_bias /= input.size();
    
    float rationality = 1.0f / (1.0f + systematic_bias);
    
    return (std::tanh(improvement) * 0.6f + rationality * 0.4f);
}

float EpisodeProcessor::compute_default_relevance(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Default processing for episodes without specific implementations
    if (input.empty()) return 0.0f;
    
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float energy = std::sqrt(std::accumulate(input.begin(), input.end(), 0.0f,
                           [](float sum, float val) { return sum + val * val; }));
    
    return std::tanh(energy / input.size()) * 0.5f; // Lower weight for default processing
}

// Utility functions

float EpisodeProcessor::compute_similarity_distance(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) return 1.0f;
    
    float distance = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    
    return std::sqrt(distance / a.size());
}

float EpisodeProcessor::apply_nonlinear_transformation(float input, const std::string& transform_type) {
    if (transform_type == "tanh") {
        return std::tanh(input);
    } else if (transform_type == "sigmoid") {
        return 1.0f / (1.0f + std::exp(-input));
    } else if (transform_type == "relu") {
        return std::max(0.0f, input);
    } else if (transform_type == "softplus") {
        return std::log(1.0f + std::exp(input));
    }
    
    return input; // Linear (default)
}

std::vector<float> EpisodeProcessor::extract_features(const std::vector<float>& input, Episode /* episode */) {
    std::vector<float> features;
    
    if (input.empty()) return features;
    
    // Basic statistical features
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    features.push_back(mean);
    
    float variance = 0.0f;
    for (float val : input) {
        variance += (val - mean) * (val - mean);
    }
    variance /= input.size();
    features.push_back(variance);
    
    // Episode-specific features could be added here
    // For now, use general features
    
    return features;
}

float EpisodeProcessor::integrate_contextual_factors(
    float base_value,
    const std::unordered_map<std::string, float>& context
) {
    float integrated_value = base_value;
    
    // Apply contextual modulation
    for (const auto& [key, value] : context) {
        if (key == "attention_weight") {
            integrated_value *= (1.0f + value * 0.5f);
        } else if (key == "memory_activation") {
            integrated_value *= (1.0f + value * 0.3f);
        } else if (key == "goal_alignment") {
            integrated_value *= (1.0f + value * 0.7f);
        }
    }
    
    return std::min(1.0f, integrated_value); // Clamp to [0, 1]
}

void EpisodeProcessor::activate_episode(Episode episode, float strength) {
    activations_[episode] = std::max(0.0f, std::min(1.0f, strength));
}

void EpisodeProcessor::deactivate_episode(Episode episode) {
    activations_[episode] = 0.0f;
}

std::unordered_map<Episode, float> EpisodeProcessor::get_activations() const {
    return activations_;
}

void EpisodeProcessor::update_config(const std::unordered_map<std::string, float>& config) {
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
}

void EpisodeProcessor::reset() {
    // Reset all activations to default
    for (auto& [episode, activation] : activations_) {
        activation = 1.0f;
    }
}

EpisodeResult EpisodeProcessor::get_episode_result(Episode episode, const std::vector<float>& input) {
    EpisodeResult result;
    result.episode = episode;
    result.contribution = process_episode(episode, input, {});
    result.confidence = activations_[episode];
    result.features = extract_features(input, episode);
    
    // Generate episode-specific insights (placeholder implementation)
    result.insights["processing_strength"] = result.contribution;
    result.insights["activation_level"] = result.confidence;
    
    return result;
}

} // namespace rros
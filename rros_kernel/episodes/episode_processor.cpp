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

    // Additional episode processors for complete coverage
    episode_processors_[Episode::NEOPLATONISM] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_neoplatonism(input, context);
        };

    episode_processors_[Episode::GNOSIS_ANAGOGE] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_gnosis_anagoge(input, context);
        };

    episode_processors_[Episode::DEATH_MEANING] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_death_meaning(input, context);
        };

    episode_processors_[Episode::ECOLOGY_PRACTICES] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_ecology_practices(input, context);
        };

    episode_processors_[Episode::LOVE_WISDOM] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_love_wisdom(input, context);
        };

    episode_processors_[Episode::WONDER_CURIOSITY] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_wonder_curiosity(input, context);
        };

    episode_processors_[Episode::OPPONENT_PROCESSING] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_opponent_processing(input, context);
        };

    episode_processors_[Episode::EXAPTATION_COOPTION] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_exaptation_cooption(input, context);
        };

    episode_processors_[Episode::PSYCHEDELICS_INSIGHTS] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_psychedelics_insights(input, context);
        };

    episode_processors_[Episode::MEDITATION_WISDOM] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_meditation_wisdom(input, context);
        };

    episode_processors_[Episode::CONSCIOUSNESS_MYSTERY] =
        [this](const std::vector<float>& input, const std::unordered_map<std::string, float>& context) {
            return process_consciousness_mystery(input, context);
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

// Additional episode processor implementations

float EpisodeProcessor::process_neoplatonism(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Neoplatonic ascent: movement toward unity and the One
    if (input.empty()) return 0.0f;

    // Henosis (unity): measure convergence toward single value
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float dispersion = 0.0f;
    for (float val : input) {
        dispersion += std::abs(val - mean);
    }
    dispersion /= input.size();

    float unity = 1.0f / (1.0f + dispersion);

    // Emanation: detect hierarchical structure (values decreasing from peak)
    float max_val = *std::max_element(input.begin(), input.end());
    float emanation_order = 0.0f;
    for (float val : input) {
        emanation_order += (max_val - std::abs(val - max_val)) / (max_val + 1e-6f);
    }
    emanation_order /= input.size();

    return (unity * 0.6f + emanation_order * 0.4f);
}

float EpisodeProcessor::process_gnosis_anagoge(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Gnosis: transformative knowing; Anagoge: upward-leading ascent
    if (input.size() < 4) return 0.0f;

    // Anagogic ascent: detect upward trend with integration
    float trend = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        trend += (input[i] - input[i-1]);
    }
    trend /= (input.size() - 1);

    // Integration depth: multi-scale coherence (like wisdom contemplation)
    float integration = 0.0f;
    for (size_t window = 2; window <= std::min(input.size(), size_t(6)); ++window) {
        float window_coherence = 0.0f;
        for (size_t i = 0; i <= input.size() - window; ++i) {
            float window_mean = 0.0f;
            for (size_t j = i; j < i + window; ++j) {
                window_mean += input[j];
            }
            window_mean /= window;
            for (size_t j = i; j < i + window; ++j) {
                window_coherence += 1.0f / (1.0f + std::abs(input[j] - window_mean));
            }
        }
        integration += window_coherence / input.size();
    }

    // Transformation marker from context
    float transformation = 1.0f;
    if (context.find("transformation_depth") != context.end()) {
        transformation = 1.0f + context.at("transformation_depth");
    }

    return std::tanh((trend + 1.0f) * integration * 0.3f) * transformation;
}

float EpisodeProcessor::process_death_meaning(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Existential meaning in face of mortality: finitude awareness
    if (input.empty()) return 0.0f;

    // Finitude: detect bounded nature and endpoints
    float range = 0.0f;
    if (input.size() > 1) {
        float min_val = *std::min_element(input.begin(), input.end());
        float max_val = *std::max_element(input.begin(), input.end());
        range = max_val - min_val;
    }

    // Mortality salience: sharp transitions (memento mori moments)
    float transition_sharpness = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        float change = std::abs(input[i] - input[i-1]);
        transition_sharpness = std::max(transition_sharpness, change);
    }

    // Meaning despite finitude: coherence within bounds
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float coherence = 0.0f;
    for (float val : input) {
        coherence += 1.0f / (1.0f + std::abs(val - mean));
    }
    coherence /= input.size();

    return (coherence * 0.5f + std::tanh(transition_sharpness) * 0.3f +
            std::tanh(range) * 0.2f);
}

float EpisodeProcessor::process_ecology_practices(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Ecology of practices: interconnected, mutually supporting practices
    if (input.size() < 3) return 0.0f;

    // Mutual support: correlation between adjacent elements
    float mutual_support = 0.0f;
    for (size_t i = 1; i < input.size() - 1; ++i) {
        // Element supported by neighbors
        float neighbor_avg = (input[i-1] + input[i+1]) / 2.0f;
        mutual_support += 1.0f / (1.0f + std::abs(input[i] - neighbor_avg));
    }
    mutual_support /= (input.size() - 2);

    // Diversity: variance indicating multiple distinct practices
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float diversity = 0.0f;
    for (float val : input) {
        diversity += (val - mean) * (val - mean);
    }
    diversity = std::sqrt(diversity / input.size());

    // Systemic health: balance of unity and diversity
    float health = mutual_support * (1.0f + diversity * 0.5f);

    return std::tanh(health);
}

float EpisodeProcessor::process_love_wisdom(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Philia (love of wisdom): attraction toward truth and understanding
    if (input.empty()) return 0.0f;

    // Attraction: movement toward positive values
    float attraction = 0.0f;
    float positive_sum = 0.0f;
    for (float val : input) {
        if (val > 0) {
            attraction += val;
            positive_sum += 1.0f;
        }
    }
    attraction = positive_sum > 0 ? attraction / positive_sum : 0.0f;

    // Care: sustained attention (low variance, consistent engagement)
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float care = 0.0f;
    for (float val : input) {
        care += 1.0f / (1.0f + std::abs(val - mean));
    }
    care /= input.size();

    // Agape: unconditional aspect (response regardless of input quality)
    float agape = 1.0f / (1.0f + std::abs(mean)); // Centered response

    return (attraction * 0.4f + care * 0.4f + agape * 0.2f);
}

float EpisodeProcessor::process_wonder_curiosity(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Wonder and curiosity: openness to novelty and mystery
    if (input.empty()) return 0.0f;

    // Novelty detection: unexpected changes
    float novelty = 0.0f;
    float expected = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        float surprise = std::abs(input[i] - expected);
        novelty += surprise;
        expected = expected * 0.7f + input[i] * 0.3f; // Update expectation
    }
    novelty /= (input.size() - 1);

    // Openness: range of values engaged with
    float min_val = *std::min_element(input.begin(), input.end());
    float max_val = *std::max_element(input.begin(), input.end());
    float openness = max_val - min_val;

    // Sustained engagement: maintaining attention despite novelty
    float sustained = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        if (std::abs(input[i]) > 0.1f && std::abs(input[i-1]) > 0.1f) {
            sustained += 1.0f;
        }
    }
    sustained /= (input.size() - 1);

    return (std::tanh(novelty) * 0.4f + std::tanh(openness) * 0.3f +
            sustained * 0.3f);
}

float EpisodeProcessor::process_opponent_processing(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Opponent processing: dynamic tension between opposing forces
    if (input.size() < 2) return 0.0f;

    // Oscillation: alternating positive/negative
    float oscillation = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        if ((input[i] > 0 && input[i-1] < 0) || (input[i] < 0 && input[i-1] > 0)) {
            oscillation += 1.0f;
        }
    }
    oscillation /= (input.size() - 1);

    // Balance: equal representation of opposing forces
    float positive_energy = 0.0f, negative_energy = 0.0f;
    for (float val : input) {
        if (val > 0) positive_energy += val;
        else negative_energy += std::abs(val);
    }
    float total = positive_energy + negative_energy + 1e-6f;
    float balance = 1.0f - std::abs(positive_energy - negative_energy) / total;

    // Dynamic equilibrium: stable despite opposition
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float equilibrium = 1.0f / (1.0f + std::abs(mean)); // Closer to zero = better equilibrium

    return (oscillation * 0.4f + balance * 0.3f + equilibrium * 0.3f);
}

float EpisodeProcessor::process_exaptation_cooption(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Exaptation: repurposing existing structures for new functions
    if (input.size() < 4) return 0.0f;

    // Structural similarity with functional difference
    // Compare first half to second half
    size_t mid = input.size() / 2;
    float structural_similarity = 0.0f;
    float functional_difference = 0.0f;

    for (size_t i = 0; i < mid; ++i) {
        structural_similarity += 1.0f / (1.0f + std::abs(input[i] - input[mid + i]));
        functional_difference += std::abs(input[i] - input[mid + i]);
    }
    structural_similarity /= mid;
    functional_difference /= mid;

    // Cooption potential: existing patterns available for reuse
    float pattern_richness = 0.0f;
    std::unordered_map<int, int> quantized_counts;
    for (float val : input) {
        int q = static_cast<int>(val * 5.0f);
        quantized_counts[q]++;
    }
    pattern_richness = static_cast<float>(quantized_counts.size()) / input.size();

    return (structural_similarity * 0.4f + std::tanh(functional_difference) * 0.3f +
            pattern_richness * 0.3f);
}

float EpisodeProcessor::process_psychedelics_insights(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Psychedelic-like processing: enhanced pattern recognition, boundary dissolution
    if (input.empty()) return 0.0f;

    // Boundary dissolution: reduced distinction between elements
    float boundary_dissolution = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        boundary_dissolution += 1.0f / (1.0f + std::abs(input[i] - input[i-1]) * 5.0f);
    }
    boundary_dissolution /= (input.size() - 1);

    // Enhanced pattern perception: detect subtle patterns
    float pattern_enhancement = 0.0f;
    for (size_t lag = 1; lag <= std::min(size_t(4), input.size() / 2); ++lag) {
        float correlation = 0.0f;
        for (size_t i = 0; i < input.size() - lag; ++i) {
            correlation += input[i] * input[i + lag];
        }
        pattern_enhancement += std::abs(correlation) / (input.size() - lag);
    }

    // Meaningfulness: subjective sense of significance
    float energy = std::sqrt(std::accumulate(input.begin(), input.end(), 0.0f,
        [](float sum, float val) { return sum + val * val; }));
    float meaningfulness = std::tanh(energy / input.size());

    return (boundary_dissolution * 0.35f + std::tanh(pattern_enhancement) * 0.35f +
            meaningfulness * 0.3f);
}

float EpisodeProcessor::process_meditation_wisdom(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& context
) {
    // Meditation leading to wisdom: concentration + insight
    if (input.empty()) return 0.0f;

    // Concentration (samatha): stability of attention
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float variance = 0.0f;
    for (float val : input) {
        variance += (val - mean) * (val - mean);
    }
    variance /= input.size();
    float concentration = 1.0f / (1.0f + variance);

    // Insight (vipassana): seeing clearly
    float insight = 0.0f;
    for (size_t i = 2; i < input.size(); ++i) {
        // Detect second-order patterns (insight into patterns)
        float d1 = input[i] - input[i-1];
        float d2 = input[i-1] - input[i-2];
        if (std::abs(d1 - d2) < 0.1f) { // Pattern detected
            insight += 1.0f;
        }
    }
    insight /= std::max(size_t(1), input.size() - 2);

    // Equanimity: balanced response
    float equanimity = 1.0f / (1.0f + std::abs(mean));

    // Practice depth from context
    float depth = 1.0f;
    if (context.find("practice_depth") != context.end()) {
        depth = 1.0f + context.at("practice_depth");
    }

    return (concentration * 0.4f + insight * 0.35f + equanimity * 0.25f) * depth;
}

float EpisodeProcessor::process_consciousness_mystery(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
) {
    // Consciousness and its mysteries: hard problem, qualia, awareness
    if (input.empty()) return 0.0f;

    // Self-reference: signal's relationship to itself
    float self_reference = 0.0f;
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    for (float val : input) {
        // How much does each value "know" the whole?
        self_reference += 1.0f / (1.0f + std::abs(val - mean));
    }
    self_reference /= input.size();

    // Integration: unified experience from diverse inputs
    float integration = 0.0f;
    for (size_t i = 1; i < input.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            integration += 1.0f / (1.0f + std::abs(input[i] - input[j]));
        }
    }
    integration /= (input.size() * (input.size() - 1) / 2.0f + 1e-6f);

    // Irreducibility: cannot be fully explained by parts
    float total_energy = std::accumulate(input.begin(), input.end(), 0.0f,
        [](float sum, float val) { return sum + std::abs(val); });
    float max_part = *std::max_element(input.begin(), input.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });
    float irreducibility = 1.0f - (std::abs(max_part) / (total_energy + 1e-6f));

    return (self_reference * 0.35f + integration * 0.35f + irreducibility * 0.3f);
}

float EpisodeProcessor::compute_default_relevance(
    const std::vector<float>& input,
    const std::unordered_map<std::string, float>& /* context */
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
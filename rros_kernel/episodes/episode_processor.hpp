#pragma once

#include "../core/rros_kernel.hpp"
#include <unordered_map>
#include <functional>

namespace rros {

/**
 * @brief Episode-specific processing result
 */
struct EpisodeResult {
    Episode episode;                    // Episode identifier
    float contribution;                 // Contribution to overall processing
    float confidence;                   // Confidence in episode processing
    std::vector<float> features;        // Episode-specific feature extraction
    std::unordered_map<std::string, float> insights; // Generated insights
};

/**
 * @brief Episode processor integrating all 50 Relevance Realization episodes
 * 
 * Each episode contributes unique cognitive processing capabilities based on
 * Vervaeke's framework. This processor coordinates episode-specific processing
 * and integrates results into unified cognitive output.
 */
class EpisodeProcessor {
public:
    explicit EpisodeProcessor(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Process input through specific episode
     * @param episode Target episode for processing
     * @param input Input data
     * @param context Processing context
     * @return Episode processing contribution
     */
    float process_episode(
        Episode episode,
        const std::vector<float>& input,
        const std::unordered_map<std::string, float>& context
    );
    
    /**
     * @brief Compute relevance using specific episode framework
     * @param episode Episode to use for relevance computation
     * @param data Input data
     * @return Relevance value from episode perspective
     */
    float compute_relevance(Episode episode, const std::vector<float>& data);
    
    /**
     * @brief Activate specific episode for processing
     * @param episode Episode to activate
     * @param strength Activation strength [0.0, 1.0]
     */
    void activate_episode(Episode episode, float strength);
    
    /**
     * @brief Deactivate episode processing
     * @param episode Episode to deactivate
     */
    void deactivate_episode(Episode episode);
    
    /**
     * @brief Get current episode activations
     * @return Map of episode activation levels
     */
    std::unordered_map<Episode, float> get_activations() const;
    
    /**
     * @brief Update processor configuration
     * @param config New configuration parameters
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset processor to initial state
     */
    void reset();
    
    /**
     * @brief Get detailed episode processing results
     * @param episode Target episode
     * @param input Input data
     * @return Detailed processing result
     */
    EpisodeResult get_episode_result(Episode episode, const std::vector<float>& input);

private:
    std::unordered_map<std::string, float> config_;
    std::unordered_map<Episode, float> activations_;
    
    // Episode-specific processors
    std::unordered_map<Episode, std::function<float(const std::vector<float>&, 
                                                   const std::unordered_map<std::string, float>&)>> 
                      episode_processors_;
    
    // Initialize episode-specific processing functions
    void initialize_processors();
    
    // Individual episode processors
    float process_flow_mysticism(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_continuous_cosmos(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_axial_revolution(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_plato_cave(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_aristotle_wisdom(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_mindfulness_insight(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_higher_order_thought(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_self_deception(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_embodied_cognition(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_relevance_realization(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_mystical_experiences(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_cognitive_revolution(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_scientific_revolution(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_wisdom_contemplation(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    float process_intelligence_rationality(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
    
    // Helper functions
    float compute_similarity_distance(const std::vector<float>& a, const std::vector<float>& b);
    float apply_nonlinear_transformation(float input, const std::string& transform_type);
    std::vector<float> extract_features(const std::vector<float>& input, Episode episode);
    float integrate_contextual_factors(float base_value, const std::unordered_map<std::string, float>& context);
    float compute_default_relevance(const std::vector<float>& input, const std::unordered_map<std::string, float>& context);
};

} // namespace rros
#include "rros_kernel.hpp"
#include "relevance_engine.hpp"
#include "attention_manager.hpp"
#include "memory_core.hpp"
#include "../episodes/episode_processor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

CognitiveState::CognitiveState() : 
    global_relevance(0.0f),
    attention_focus(0.0f),
    current_salience(0.0f),
    confidence(0.5f),
    last_update(std::chrono::high_resolution_clock::now())
{
    // Initialize mode activations
    for (int i = 0; i <= static_cast<int>(CognitiveMode::META_COGNITIVE); ++i) {
        mode_activations[static_cast<CognitiveMode>(i)] = 0.0f;
    }
    
    // Initialize episode influences
    for (int i = 0; i <= static_cast<int>(Episode::TILLICH_BARFIELD); ++i) {
        episode_influences[static_cast<Episode>(i)] = 0.0f;
    }
}

RROSKernel::RROSKernel(const std::unordered_map<std::string, float>& config) :
    config_(config),
    start_time_(std::chrono::high_resolution_clock::now())
{
    // Set default configuration values
    if (config_.find("relevance_threshold") == config_.end()) {
        config_["relevance_threshold"] = 0.3f;
    }
    if (config_.find("attention_decay") == config_.end()) {
        config_["attention_decay"] = 0.95f;
    }
    if (config_.find("memory_capacity") == config_.end()) {
        config_["memory_capacity"] = 10000.0f;
    }
    if (config_.find("episode_integration_rate") == config_.end()) {
        config_["episode_integration_rate"] = 0.1f;
    }
    
    initialize_subsystems();
}

RROSKernel::~RROSKernel() = default;

void RROSKernel::initialize_subsystems() {
    relevance_engine_ = std::make_unique<RelevanceEngine>(config_);
    attention_manager_ = std::make_unique<AttentionManager>(config_);
    memory_core_ = std::make_unique<MemoryCore>(config_);
    episode_processor_ = std::make_unique<EpisodeProcessor>(config_);
}

CognitiveState RROSKernel::cognitive_cycle(
    const std::vector<float>& input_data,
    const std::unordered_map<std::string, float>& context
) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    ++cycle_count_;
    
    // 1. Process input through relevance engine
    auto relevance_result = relevance_engine_->process(input_data, context);
    
    // 2. Update attention allocation based on relevance
    auto attention_weights = attention_manager_->update(relevance_result.salience_map);
    
    // 3. Process through active episodes
    std::vector<float> episode_contributions;
    for (int i = 0; i <= static_cast<int>(Episode::TILLICH_BARFIELD); ++i) {
        Episode episode = static_cast<Episode>(i);
        float contribution = episode_processor_->process_episode(episode, input_data, context);
        episode_contributions.push_back(contribution);
        current_state_.episode_influences[episode] = contribution;
    }
    
    // 4. Integrate episode contributions
    float integrated_relevance = 0.0f;
    for (size_t i = 0; i < episode_contributions.size(); ++i) {
        integrated_relevance += episode_contributions[i] * (1.0f / episode_contributions.size());
    }
    
    // 5. Update memory with current experience
    memory_core_->store(input_data, integrated_relevance, attention_weights);
    
    // 6. Update cognitive state
    current_state_.global_relevance = integrated_relevance;
    current_state_.attention_focus = *std::max_element(attention_weights.begin(), attention_weights.end());
    current_state_.current_salience = relevance_result.max_salience;
    current_state_.confidence = compute_confidence(relevance_result, episode_contributions);
    current_state_.last_update = std::chrono::high_resolution_clock::now();
    
    // 7. Update mode activations based on episode processing
    update_mode_activations(episode_contributions);
    
    return current_state_;
}

RelevanceValue RROSKernel::realize_relevance(
    const std::vector<float>& data,
    const std::vector<Episode>& episode_mask
) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Use all episodes if no mask specified
    std::vector<Episode> active_episodes = episode_mask;
    if (active_episodes.empty()) {
        for (int i = 0; i <= static_cast<int>(Episode::TILLICH_BARFIELD); ++i) {
            active_episodes.push_back(static_cast<Episode>(i));
        }
    }
    
    // Process through selected episodes
    float total_relevance = 0.0f;
    float weight_sum = 0.0f;
    
    for (Episode episode : active_episodes) {
        float weight = get_episode_weight(episode);
        float episode_relevance = episode_processor_->compute_relevance(episode, data);
        
        total_relevance += episode_relevance * weight;
        weight_sum += weight;
    }
    
    return weight_sum > 0.0f ? total_relevance / weight_sum : 0.0f;
}

std::vector<AttentionWeight> RROSKernel::allocate_attention(
    const std::vector<std::vector<float>>& targets
) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    std::vector<AttentionWeight> weights;
    weights.reserve(targets.size());
    
    // Compute relevance for each target
    std::vector<float> relevance_scores;
    for (const auto& target : targets) {
        float relevance = realize_relevance(target);
        relevance_scores.push_back(relevance);
    }
    
    // Apply attention allocation algorithm
    return attention_manager_->allocate(relevance_scores);
}

float RROSKernel::process_episode(Episode episode, const std::vector<float>& input) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return episode_processor_->process_episode(episode, input, {});
}

void RROSKernel::update_config(const std::unordered_map<std::string, float>& config) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
    
    // Update subsystem configurations
    relevance_engine_->update_config(config_);
    attention_manager_->update_config(config_);
    memory_core_->update_config(config_);
    episode_processor_->update_config(config_);
}

void RROSKernel::reset() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    current_state_ = CognitiveState();
    cycle_count_ = 0;
    start_time_ = std::chrono::high_resolution_clock::now();
    
    // Reset subsystems
    relevance_engine_->reset();
    attention_manager_->reset();
    memory_core_->reset();
    episode_processor_->reset();
}

std::unordered_map<std::string, float> RROSKernel::get_metrics() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    
    std::unordered_map<std::string, float> metrics;
    metrics["cycles_per_second"] = static_cast<float>(cycle_count_) / (duration.count() / 1000.0f);
    metrics["average_relevance"] = current_state_.global_relevance;
    metrics["attention_efficiency"] = current_state_.attention_focus;
    metrics["confidence_level"] = current_state_.confidence;
    metrics["memory_utilization"] = memory_core_->get_utilization();
    
    // Episode activation statistics
    float total_activation = 0.0f;
    int active_episodes = 0;
    for (const auto& [episode, activation] : current_state_.episode_influences) {
        total_activation += activation;
        if (activation > 0.1f) ++active_episodes;
    }
    
    metrics["total_episode_activation"] = total_activation;
    metrics["active_episodes"] = static_cast<float>(active_episodes);
    
    return metrics;
}

void RROSKernel::activate_episode(Episode episode, float strength) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    episode_processor_->activate_episode(episode, strength);
}

void RROSKernel::deactivate_episode(Episode episode) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    episode_processor_->deactivate_episode(episode);
}

std::unordered_map<Episode, float> RROSKernel::get_episode_activations() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return episode_processor_->get_activations();
}

float RROSKernel::compute_confidence(
    const RelevanceResult& relevance_result,
    const std::vector<float>& episode_contributions
) {
    // Confidence based on consistency of episode contributions and relevance strength
    float mean_contribution = std::accumulate(episode_contributions.begin(), 
                                            episode_contributions.end(), 0.0f) / episode_contributions.size();
    
    float variance = 0.0f;
    for (float contrib : episode_contributions) {
        variance += (contrib - mean_contribution) * (contrib - mean_contribution);
    }
    variance /= episode_contributions.size();
    
    float consistency = 1.0f / (1.0f + variance);  // Higher consistency = lower variance
    float strength = relevance_result.max_salience;
    
    return (consistency * 0.6f + strength * 0.4f);
}

void RROSKernel::update_mode_activations(const std::vector<float>& episode_contributions) {
    // Map episode contributions to cognitive modes based on episode content
    
    // Reset mode activations
    for (auto& [mode, activation] : current_state_.mode_activations) {
        activation = 0.0f;
    }
    
    // Aggregate episode contributions by cognitive mode
    for (size_t i = 0; i < episode_contributions.size() && i <= static_cast<size_t>(Episode::TILLICH_BARFIELD); ++i) {
        Episode episode = static_cast<Episode>(i);
        float contribution = episode_contributions[i];
        
        // Map episodes to primary cognitive modes
        switch (episode) {
            case Episode::FLOW_MYSTICISM:
            case Episode::MINDFULNESS_INSIGHT:
                current_state_.mode_activations[CognitiveMode::SELECTIVE_ATTENTION] += contribution;
                break;
                
            case Episode::HIGHER_ORDER_THOUGHT:
            case Episode::ARISTOTLE_WISDOM:
            case Episode::AQUINAS_ARISTOTLE:
                current_state_.mode_activations[CognitiveMode::WORKING_MEMORY] += contribution;
                break;
                
            case Episode::PLATO_CAVE:
            case Episode::SCIENTIFIC_REVOLUTION:
                current_state_.mode_activations[CognitiveMode::PROBLEM_SPACE] += contribution;
                break;
                
            case Episode::EMBODIED_COGNITION:
            case Episode::MARTIAL_ARTS:
            case Episode::ECOLOGY_PRACTICES:
                current_state_.mode_activations[CognitiveMode::SIDE_EFFECTS] += contribution;
                break;
                
            case Episode::AUGUSTINE_WORLD:
            case Episode::WISDOM_CONTEMPLATION:
                current_state_.mode_activations[CognitiveMode::LONG_TERM_MEMORY] += contribution;
                break;
                
            default:
                current_state_.mode_activations[CognitiveMode::META_COGNITIVE] += contribution;
                break;
        }
    }
    
    // Normalize mode activations
    float total_activation = 0.0f;
    for (const auto& [mode, activation] : current_state_.mode_activations) {
        total_activation += activation;
    }
    
    if (total_activation > 0.0f) {
        for (auto& [mode, activation] : current_state_.mode_activations) {
            activation /= total_activation;
        }
    }
}

float RROSKernel::get_episode_weight(Episode episode) const {
    // Base weights derived from episode importance in Vervaeke's framework
    static const std::unordered_map<Episode, float> base_weights = {
        {Episode::RELEVANCE_REALIZATION, 1.0f},   // Core episode
        {Episode::MINDFULNESS_INSIGHT, 0.9f},     // Key insight mechanism
        {Episode::FLOW_MYSTICISM, 0.9f},          // Transformative experience
        {Episode::EMBODIED_COGNITION, 0.8f},      // 4E cognition foundation
        {Episode::SELF_DECEPTION, 0.8f},          // Wisdom vs foolishness
        {Episode::PLATO_CAVE, 0.7f},              // Classic insight allegory
        {Episode::ARISTOTLE_WISDOM, 0.7f},        // Practical wisdom
        {Episode::AXIAL_REVOLUTION, 0.7f},        // Historical transformation
        {Episode::SCIENTIFIC_REVOLUTION, 0.6f},   // Modern transformation
        {Episode::COGNITIVE_REVOLUTION, 0.6f},    // Contemporary framework
        // ... (additional weights for other episodes)
    };
    
    auto it = base_weights.find(episode);
    return it != base_weights.end() ? it->second : 0.5f;  // Default weight
}

} // namespace rros
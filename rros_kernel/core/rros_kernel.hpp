#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <atomic>
#include <mutex>
#include <chrono>
#include <functional>

namespace rros {

// Forward declarations
class RelevanceEngine;
class RelevanceOptimizer;
class AttentionManager;
class MemoryCore;
class EpisodeProcessor;
class MetaCognitiveMonitor;
class DecisionQualityTracker;
class ConfidenceEstimator;
class BiasDetector;
class SelfOptimizer;
class ResourceManager;
struct RelevanceResult;

/**
 * @brief Core types for the RROS Pure Cognitive Kernel
 * 
 * Integrates all 50 Episodes of Relevance Realization into a unified
 * cognitive architecture inspired by Vervaeke's framework
 */

using RelevanceValue = float;        // [0.0, 1.0] relevance strength
using AttentionWeight = float;       // [0.0, 1.0] attention allocation
using SalienceLevel = float;         // Salience in current context
using ConfidenceScore = float;       // [0.0, 1.0] confidence measure
using Timestamp = std::chrono::high_resolution_clock::time_point;

/**
 * @brief Episode Enumeration for all 50 Relevance Realization episodes
 * 
 * Each episode contributes specific cognitive capabilities to the kernel
 */
enum class Episode {
    INTRO = 0,                       // Introduction to meaning crisis
    FLOW_MYSTICISM = 1,              // Flow and mystical experience
    CONTINUOUS_COSMOS = 2,           // Continuous cosmos and shamanism
    AXIAL_REVOLUTION = 3,            // Axial age revolution
    ISRAEL_PROPHETS = 4,             // Ancient Israel and prophets
    PLATO_CAVE = 5,                  // Plato and cave allegory
    ARISTOTLE_WISDOM = 6,            // Aristotle and practical wisdom
    ARISTOTLE_WORLDVIEW = 7,         // Aristotelian worldview
    SIDDHARTHA_PATH = 8,             // Siddhartha and the middle path
    MINDFULNESS_INSIGHT = 9,         // Mindfulness and insight
    HIGHER_ORDER_THOUGHT = 10,       // Higher order thought
    SELF_DECEPTION = 11,             // Self deception and wisdom
    CHRISTIANITY_NARRATIVE = 12,     // Christianity and narrative
    NEOPLATONISM = 13,               // Neoplatonism and mysticism
    AUGUSTINE_WORLD = 14,            // Augustine and inner world
    AQUINAS_ARISTOTLE = 15,          // Aquinas and Aristotelian revival
    AVERROES_MAIMONIDES = 16,        // Averroes, Maimonides, Aquinas
    SCHOLASTICISM = 17,              // Scholasticism and scientific revolution
    LUTHER_MEANING = 18,             // Luther and meaning crisis
    EMERGENCE_OVERVIEW = 19,         // Emergence and overview
    DESCARTES_SCIENTIFIC = 20,       // Descartes and scientific revolution
    BACON_GALILEO = 21,              // Bacon, Galileo, and experimental method
    SCIENTIFIC_REVOLUTION = 22,      // Scientific revolution
    ROMANTICISM_HORROR = 23,         // Romanticism and horror
    HEGEL_HISTORY = 24,              // Hegel and historical consciousness
    SCHOPENHAUER_NIETZSCHE = 25,     // Schopenhauer and Nietzsche
    EVOLUTION_MEANING = 26,          // Evolution and meaning
    COGNITIVE_REVOLUTION = 27,       // Cognitive revolution
    EMBODIED_COGNITION = 28,         // Embodied and 4E cognition
    OPPONENT_PROCESSING = 29,        // Opponent processing
    RELEVANCE_REALIZATION = 30,      // Relevance realization
    EXAPTATION_COOPTION = 31,        // Exaptation and cognitive cooption
    SHAMANISM_INTEGRATION = 32,      // Shamanism and integration
    FLOW_METAPHOR = 33,              // Flow, metaphor, and insight
    PSYCHEDELICS_INSIGHTS = 34,      // Psychedelics and insights
    MYSTICAL_EXPERIENCES = 35,       // Mystical experiences
    GNOSIS_ANAGOGE = 36,             // Gnosis and anagoge
    MARTIAL_ARTS = 37,               // Martial arts and mindfulness
    MEDITATION_WISDOM = 38,          // Meditation and wisdom
    CONSCIOUSNESS_MYSTERY = 39,      // Consciousness and mystery
    DEATH_MEANING = 40,              // Death and meaning
    WISDOM_CONTEMPLATION = 41,       // Wisdom and contemplation
    INTELLIGENCE_RATIONALITY = 42,   // Intelligence and rationality
    ECOLOGY_PRACTICES = 43,          // Ecology of practices
    LOVE_WISDOM = 44,                // Love and wisdom
    WONDER_CURIOSITY = 45,           // Wonder and curiosity
    PHILOSOPHY_RELIGION = 46,        // Philosophy as religion
    PANPSYCHISM_EMERGENCE = 47,      // Panpsychism and emergence
    RESPONSE_MEANING = 48,           // Response to meaning crisis
    CORBIN_JUNG = 49,                // Corbin and Jung
    TILLICH_BARFIELD = 50            // Tillich and Barfield
};

/**
 * @brief Cognitive modes from Vervaeke's framework
 */
enum class CognitiveMode {
    SELECTIVE_ATTENTION,    // Bottom-up salience detection
    WORKING_MEMORY,        // Active maintenance and manipulation
    PROBLEM_SPACE,         // Goal-directed search and navigation
    SIDE_EFFECTS,          // Action consequence evaluation
    LONG_TERM_MEMORY,      // Organizational and retrieval systems
    META_COGNITIVE         // Higher-order monitoring and control
};

/**
 * @brief Cognitive state representing current system status
 */
struct CognitiveState {
    std::unordered_map<CognitiveMode, float> mode_activations;
    std::unordered_map<Episode, float> episode_influences;
    RelevanceValue global_relevance;
    AttentionWeight attention_focus;
    SalienceLevel current_salience;
    ConfidenceScore confidence;
    Timestamp last_update;
    
    CognitiveState();
};

/**
 * @brief Core RROS (Relevance Realization Operating System) Kernel
 * 
 * Integrates all 50 episodes of relevance realization into a unified
 * cognitive architecture that can process information, allocate attention,
 * manage memory, and realize relevance according to Vervaeke's framework.
 */
class RROSKernel {
public:
    /**
     * @brief Constructor initializes all cognitive subsystems
     * @param config Configuration parameters for kernel initialization
     */
    explicit RROSKernel(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~RROSKernel();
    
    // Core cognitive cycle operations
    
    /**
     * @brief Execute one cognitive cycle integrating all episodes
     * @param input_data Sensory or conceptual input
     * @param context Current contextual information
     * @return Updated cognitive state
     */
    CognitiveState cognitive_cycle(
        const std::vector<float>& input_data,
        const std::unordered_map<std::string, float>& context = {}
    );
    
    /**
     * @brief Realize relevance for given input using episode-specific processing
     * @param data Input data to evaluate
     * @param episode_mask Episodes to activate for processing
     * @return Relevance realization result
     */
    RelevanceValue realize_relevance(
        const std::vector<float>& data,
        const std::vector<Episode>& episode_mask = {}
    );
    
    /**
     * @brief Allocate attention based on current state and episode influences
     * @param targets Potential attention targets
     * @return Attention allocation weights
     */
    std::vector<AttentionWeight> allocate_attention(
        const std::vector<std::vector<float>>& targets
    );
    
    /**
     * @brief Process specific episode contributions to current cognitive state
     * @param episode Target episode to process
     * @param input Current input context
     * @return Episode-specific cognitive contribution
     */
    float process_episode(Episode episode, const std::vector<float>& input);
    
    // State management
    
    /**
     * @brief Get current cognitive state
     * @return Current state of all cognitive subsystems
     */
    const CognitiveState& get_state() const { return current_state_; }
    
    /**
     * @brief Update kernel configuration
     * @param config New configuration parameters
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset kernel to initial state
     */
    void reset();
    
    /**
     * @brief Get performance metrics
     * @return System performance and efficiency metrics
     */
    std::unordered_map<std::string, float> get_metrics() const;
    
    // Episode-specific interfaces
    
    /**
     * @brief Activate specific episode processing
     * @param episode Episode to activate
     * @param strength Activation strength [0.0, 1.0]
     */
    void activate_episode(Episode episode, float strength = 1.0f);
    
    /**
     * @brief Deactivate episode processing
     * @param episode Episode to deactivate
     */
    void deactivate_episode(Episode episode);
    
    /**
     * @brief Get episode activation levels
     * @return Current activation for each episode
     */
    std::unordered_map<Episode, float> get_episode_activations() const;
    
    // Self-reflection interfaces
    
    /**
     * @brief Get meta-cognitive introspection of current state
     * @param lookback_cycles Number of recent cycles to analyze
     * @return Introspection metrics
     */
    std::unordered_map<std::string, float> introspect_cognitive_state(uint32_t lookback_cycles = 100);
    
    /**
     * @brief Get decision quality metrics
     * @return Quality metrics for cognitive decisions
     */
    std::unordered_map<std::string, float> get_decision_quality_metrics();
    
    /**
     * @brief Detect cognitive biases in recent processing
     * @return Number of biases detected
     */
    uint32_t detect_cognitive_biases();
    
    /**
     * @brief Enable autonomous self-optimization
     * @param enabled True to enable, false to disable
     */
    void enable_self_optimization(bool enabled);
    
    // Relevance optimization interfaces
    
    /**
     * @brief Get the relevance optimizer for direct access
     * @return Reference to the relevance optimizer
     */
    RelevanceOptimizer& get_relevance_optimizer();
    
    /**
     * @brief Get optimization metrics from relevance optimizer
     * @return Optimization performance metrics
     */
    std::unordered_map<std::string, float> get_optimization_metrics() const;
    
    // Resource management interfaces
    
    /**
     * @brief Get the resource manager for direct access
     * @return Reference to the resource manager
     */
    ResourceManager& get_resource_manager();
    
    /**
     * @brief Get resource utilization metrics
     * @return Resource usage across all types
     */
    std::unordered_map<std::string, float> get_resource_metrics() const;
    
    /**
     * @brief Allocate computational resources for a cognitive operation
     * @param amount Amount of computational resources needed
     * @param priority Priority level
     * @return True if allocation succeeded
     */
    bool allocate_cognitive_resources(float amount, int priority = 2);
    
    /**
     * @brief Release computational resources
     * @param amount Amount to release
     */
    void release_cognitive_resources(float amount);

private:
    // Core subsystems
    std::unique_ptr<RelevanceEngine> relevance_engine_;
    std::unique_ptr<RelevanceOptimizer> relevance_optimizer_;
    std::unique_ptr<AttentionManager> attention_manager_;
    std::unique_ptr<MemoryCore> memory_core_;
    std::unique_ptr<EpisodeProcessor> episode_processor_;
    
    // Self-reflection subsystems
    std::unique_ptr<MetaCognitiveMonitor> meta_monitor_;
    std::unique_ptr<DecisionQualityTracker> quality_tracker_;
    std::unique_ptr<ConfidenceEstimator> confidence_estimator_;
    std::unique_ptr<BiasDetector> bias_detector_;
    std::unique_ptr<SelfOptimizer> self_optimizer_;
    
    // Resource management
    std::unique_ptr<ResourceManager> resource_manager_;
    
    // Current state
    CognitiveState current_state_;
    
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Thread safety
    mutable std::mutex state_mutex_;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point start_time_;
    std::atomic<uint64_t> cycle_count_{0};
    
    // Internal methods
    void initialize_subsystems();
    void update_state();
    float compute_global_relevance();
    void normalize_activations();
    float get_episode_weight(Episode episode) const;
    float compute_confidence(const RelevanceResult& relevance_result, 
                           const std::vector<float>& episode_contributions);
    void update_mode_activations(const std::vector<float>& episode_contributions);
};

} // namespace rros
#pragma once

/**
 * @file vervaeke_4e.hpp
 * @brief Vervaeke's 4E Cognition Framework Integration
 * 
 * Implements John Vervaeke's framework for embodied, embedded, enacted,
 * and extended cognition as tensor-based RROS kernel components.
 * 
 * Based on Episode 28 "Convergence to Relevance Realization" and the
 * broader 4E cognition literature integrating with the CogPrime architecture.
 */

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <cmath>

namespace rros {
namespace vervaeke {

/**
 * @brief Sensorimotor state representation for embodied cognition
 * 
 * Represents the current state of sensory input and motor output,
 * enabling tight coupling between perception and action.
 */
struct SensorimotorState {
    std::vector<float> sensory_input;      // Raw sensory data (vision, proprioception, etc.)
    std::vector<float> motor_output;       // Motor commands/predictions
    std::vector<float> body_schema;        // Internal body representation
    float sensorimotor_coupling;           // Strength of perception-action coupling [0,1]
    float spatial_awareness;               // Current spatial reasoning capability
};

/**
 * @brief Environmental context for embedded cognition
 * 
 * Captures the agent's situatedness in a physical and social environment,
 * enabling context-sensitive behavior adaptation.
 */
struct EnvironmentalContext {
    std::vector<float> physical_affordances;   // Environmental action possibilities
    std::vector<float> social_context;         // Social/cultural embedding
    std::vector<float> temporal_context;       // Temporal dynamics and history
    float context_stability;                   // How stable the context is [0,1]
    float environmental_coupling;              // Strength of environment interaction
};

/**
 * @brief Active perception state for enacted cognition
 * 
 * Represents how the agent actively constructs meaning through
 * exploration and interaction, not passive reception.
 */
struct EnactedState {
    std::vector<float> exploration_vector;     // Current exploration direction
    std::vector<float> action_predictions;     // Predicted action outcomes
    std::vector<float> enacted_meaning;        // Meaning constructed through action
    float exploration_intensity;               // How actively exploring [0,1]
    float enaction_coherence;                  // Coherence of enacted meaning
};

/**
 * @brief Tool use and environmental scaffolding for extended cognition
 * 
 * Represents how cognitive processes extend beyond the brain into
 * tools, symbols, and environmental structures.
 */
struct ExtendedState {
    std::vector<float> tool_representations;   // Internal models of available tools
    std::vector<float> symbolic_scaffolds;     // Symbolic/cultural resources
    std::vector<float> distributed_memory;     // Memory extended into environment
    float tool_mastery;                        // Proficiency with current tools [0,1]
    float cognitive_extension;                 // Degree of mind extension [0,1]
};

/**
 * @brief Salience landscape representation
 * 
 * Implements Vervaeke's salience landscape - the dynamic field of
 * what stands out as relevant in the current context.
 */
struct SalienceLandscape {
    std::vector<float> salience_field;         // Spatial salience distribution
    std::vector<float> relevance_gradients;    // Gradients toward relevant features
    std::vector<float> attention_peaks;        // Current attention focal points
    float landscape_coherence;                 // Overall landscape coherence
    float relevance_realization_rate;          // Rate of relevance detection
};

/**
 * @brief Perspectival knowing state
 * 
 * Represents the agent's current perspective and ability to
 * integrate multiple viewpoints (Vervaeke's perspectival knowing).
 */
struct PerspectivalState {
    std::vector<float> current_perspective;    // Active viewpoint
    std::vector<std::vector<float>> alternative_perspectives; // Other viewpoints
    std::vector<float> perspective_integration; // Integrated multi-view understanding
    float perspective_flexibility;             // Ability to shift perspectives [0,1]
    float viewpoint_richness;                  // Diversity of accessible viewpoints
};

/**
 * @brief Participatory knowing state
 * 
 * Represents reciprocal relationship with reality - how the agent
 * is shaped by and shapes its world (Vervaeke's participatory knowing).
 */
struct ParticipatoryState {
    std::vector<float> agent_world_coupling;   // Bidirectional influence
    std::vector<float> transformation_state;   // How agent is being transformed
    std::vector<float> world_shaping;          // How agent shapes world
    float participation_depth;                 // Depth of participatory engagement [0,1]
    float reciprocal_realization;              // Mutual constitution strength
};

/**
 * @brief Processing mode (transformative vs conformative)
 * 
 * Vervaeke distinguishes between conformative processing (fitting to
 * current patterns) and transformative processing (insight/restructuring).
 */
enum class ProcessingMode {
    CONFORMATIVE,      // Assimilation to existing patterns
    TRANSFORMATIVE,    // Accommodation and insight
    BALANCED           // Dynamic balance between both
};

/**
 * @brief Cognitive-emotional integration state
 * 
 * Following Vervaeke's framework, cognition and emotion are deeply
 * integrated through relevance realization and meaning-making.
 */
struct CognitiveEmotionalState {
    std::vector<float> emotional_valence;      // Emotional coloring of cognition
    std::vector<float> motivational_forces;    // Drive and goal states
    std::vector<float> meaning_coherence;      // Experienced meaningfulness
    float emotional_regulation;                // Emotional self-regulation capacity
    float cognitive_emotional_sync;            // Synchronization degree [0,1]
};

/**
 * @brief Complete 4E Cognition state
 * 
 * Unified representation of all 4E aspects plus salience,
 * perspectival/participatory knowing, and processing modes.
 */
struct FourEState {
    SensorimotorState embodied;
    EnvironmentalContext embedded;
    EnactedState enacted;
    ExtendedState extended;
    SalienceLandscape salience;
    PerspectivalState perspectival;
    ParticipatoryState participatory;
    ProcessingMode processing_mode;
    CognitiveEmotionalState cognitive_emotional;
    
    // Overall coherence metrics
    float overall_coherence;                   // System-wide coherence
    float wisdom_measure;                      // Wisdom indicator per Vervaeke
    float meaning_connectivity;                // Connection strength (anti-meaning-crisis)
};

/**
 * @brief Embodied Cognition Processor
 * 
 * Implements embodied cognition principles - tight coupling between
 * sensorimotor processes and higher cognition.
 */
class EmbodiedCognitionProcessor {
public:
    explicit EmbodiedCognitionProcessor(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Process sensorimotor integration
     * @param sensory_input Current sensory data
     * @param motor_context Current motor state/intentions
     * @return Updated sensorimotor state
     */
    SensorimotorState process_sensorimotor(
        const std::vector<float>& sensory_input,
        const std::vector<float>& motor_context
    );
    
    /**
     * @brief Update body schema based on experience
     * @param state Current state
     * @param feedback Sensorimotor feedback
     */
    void update_body_schema(
        SensorimotorState& state,
        const std::vector<float>& feedback
    );
    
    /**
     * @brief Compute spatial reasoning from embodied perspective
     * @param state Current sensorimotor state
     * @return Spatial reasoning vector
     */
    std::vector<float> compute_spatial_reasoning(const SensorimotorState& state);

private:
    std::unordered_map<std::string, float> config_;
    std::vector<float> body_schema_memory_;
};

/**
 * @brief Embedded Cognition Processor
 * 
 * Implements embedded cognition - situatedness in physical and
 * social environments enabling context-sensitive adaptation.
 */
class EmbeddedCognitionProcessor {
public:
    explicit EmbeddedCognitionProcessor(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Process environmental context awareness
     * @param physical_env Physical environment state
     * @param social_env Social environment state
     * @return Updated environmental context
     */
    EnvironmentalContext process_context_awareness(
        const std::vector<float>& physical_env,
        const std::vector<float>& social_env
    );
    
    /**
     * @brief Detect environmental affordances
     * @param context Current context
     * @return Detected affordances
     */
    std::vector<float> detect_affordances(const EnvironmentalContext& context);
    
    /**
     * @brief Adapt behavior based on context
     * @param context Current context
     * @param behavior Current behavior vector
     * @return Context-adapted behavior
     */
    std::vector<float> adapt_to_context(
        const EnvironmentalContext& context,
        const std::vector<float>& behavior
    );

private:
    std::unordered_map<std::string, float> config_;
    std::vector<EnvironmentalContext> context_history_;
};

/**
 * @brief Enacted Cognition Processor
 * 
 * Implements enacted cognition - meaning emerges through active
 * exploration and sensorimotor interaction, not passive reception.
 */
class EnactedCognitionProcessor {
public:
    explicit EnactedCognitionProcessor(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Process active perception and meaning construction
     * @param current_state Current enacted state
     * @param action_space Available actions
     * @return Updated enacted state
     */
    EnactedState process_active_perception(
        const EnactedState& current_state,
        const std::vector<float>& action_space
    );
    
    /**
     * @brief Generate exploration strategy
     * @param state Current state
     * @return Exploration direction vector
     */
    std::vector<float> generate_exploration_strategy(const EnactedState& state);
    
    /**
     * @brief Construct meaning through enaction
     * @param actions Performed actions
     * @param outcomes Observed outcomes
     * @return Enacted meaning vector
     */
    std::vector<float> construct_enacted_meaning(
        const std::vector<float>& actions,
        const std::vector<float>& outcomes
    );

private:
    std::unordered_map<std::string, float> config_;
    std::vector<std::pair<std::vector<float>, std::vector<float>>> action_outcome_history_;
};

/**
 * @brief Extended Cognition Processor
 * 
 * Implements extended cognition - cognitive processes extend beyond
 * the brain into tools, symbols, and environmental structures.
 */
class ExtendedCognitionProcessor {
public:
    explicit ExtendedCognitionProcessor(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Process tool use and cognitive extension
     * @param available_tools Available cognitive tools
     * @param task Current cognitive task
     * @return Updated extended state
     */
    ExtendedState process_cognitive_extension(
        const std::vector<float>& available_tools,
        const std::vector<float>& task
    );
    
    /**
     * @brief Integrate external symbolic scaffolds
     * @param state Current state
     * @param symbols Available symbolic resources
     */
    void integrate_symbolic_scaffolds(
        ExtendedState& state,
        const std::vector<float>& symbols
    );
    
    /**
     * @brief Distribute memory into environment
     * @param internal_memory Internal memory state
     * @param environment Environment representation
     * @return Distributed memory configuration
     */
    std::vector<float> distribute_memory(
        const std::vector<float>& internal_memory,
        const std::vector<float>& environment
    );

private:
    std::unordered_map<std::string, float> config_;
    std::vector<float> tool_proficiency_;
};

/**
 * @brief Salience Landscape Navigator
 * 
 * Implements Vervaeke's salience landscape - the dynamic field of
 * relevance that guides attention and action selection.
 */
class SalienceLandscapeNavigator {
public:
    explicit SalienceLandscapeNavigator(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Update salience landscape based on current context
     * @param context Current 4E state
     * @param goals Current goals
     * @return Updated salience landscape
     */
    SalienceLandscape update_landscape(
        const FourEState& context,
        const std::vector<float>& goals
    );
    
    /**
     * @brief Navigate toward relevant features
     * @param landscape Current landscape
     * @param position Current position in feature space
     * @return Navigation direction
     */
    std::vector<float> navigate_toward_relevance(
        const SalienceLandscape& landscape,
        const std::vector<float>& position
    );
    
    /**
     * @brief Compute relevance gradients
     * @param landscape Current landscape
     * @return Relevance gradient field
     */
    std::vector<float> compute_relevance_gradients(const SalienceLandscape& landscape);

private:
    std::unordered_map<std::string, float> config_;
    SalienceLandscape landscape_memory_;
};

/**
 * @brief Perspectival and Participatory Knowing Processor
 * 
 * Implements Vervaeke's perspectival (multiple viewpoint integration)
 * and participatory (reciprocal agent-world constitution) knowing.
 */
class PerspectivalParticipatoryProcessor {
public:
    explicit PerspectivalParticipatoryProcessor(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Process perspectival knowing - integrate multiple viewpoints
     * @param current_perspective Current viewpoint
     * @param alternative_views Other available perspectives
     * @return Updated perspectival state
     */
    PerspectivalState process_perspectival_knowing(
        const std::vector<float>& current_perspective,
        const std::vector<std::vector<float>>& alternative_views
    );
    
    /**
     * @brief Process participatory knowing - reciprocal agent-world shaping
     * @param agent_state Current agent state
     * @param world_state Current world state
     * @return Updated participatory state
     */
    ParticipatoryState process_participatory_knowing(
        const std::vector<float>& agent_state,
        const std::vector<float>& world_state
    );
    
    /**
     * @brief Shift perspective to new viewpoint
     * @param state Current perspectival state
     * @param target_perspective Target viewpoint
     * @return Success of perspective shift
     */
    bool shift_perspective(
        PerspectivalState& state,
        const std::vector<float>& target_perspective
    );

private:
    std::unordered_map<std::string, float> config_;
    std::vector<std::vector<float>> perspective_library_;
};

/**
 * @brief Transformative-Conformative Processing Manager
 * 
 * Manages the balance between conformative processing (assimilation)
 * and transformative processing (accommodation/insight).
 */
class ProcessingModeManager {
public:
    explicit ProcessingModeManager(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Determine appropriate processing mode
     * @param current_state Current 4E state
     * @param challenge Current challenge/problem
     * @return Recommended processing mode
     */
    ProcessingMode determine_processing_mode(
        const FourEState& current_state,
        const std::vector<float>& challenge
    );
    
    /**
     * @brief Execute conformative processing
     * @param input Input to process
     * @param existing_patterns Existing cognitive patterns
     * @return Conformatively processed output
     */
    std::vector<float> process_conformative(
        const std::vector<float>& input,
        const std::vector<float>& existing_patterns
    );
    
    /**
     * @brief Execute transformative processing (insight)
     * @param input Input requiring restructuring
     * @param constraints Current constraints
     * @return Transformatively processed output
     */
    std::vector<float> process_transformative(
        const std::vector<float>& input,
        const std::vector<float>& constraints
    );

private:
    std::unordered_map<std::string, float> config_;
    ProcessingMode current_mode_;
};

/**
 * @brief Cognitive-Emotional Integration Processor
 * 
 * Implements Vervaeke's integration of cognition and emotion through
 * relevance realization and meaning-making.
 */
class CognitiveEmotionalIntegrator {
public:
    explicit CognitiveEmotionalIntegrator(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Integrate cognitive and emotional processing
     * @param cognitive_state Current cognitive state
     * @param emotional_state Current emotional state
     * @return Integrated cognitive-emotional state
     */
    CognitiveEmotionalState integrate_cognition_emotion(
        const std::vector<float>& cognitive_state,
        const std::vector<float>& emotional_state
    );
    
    /**
     * @brief Compute meaning coherence
     * @param state Current cognitive-emotional state
     * @return Meaning coherence measure
     */
    float compute_meaning_coherence(const CognitiveEmotionalState& state);
    
    /**
     * @brief Enhance decision quality through emotion integration
     * @param decision_options Available options
     * @param state Cognitive-emotional state
     * @return Quality-enhanced decision ranking
     */
    std::vector<float> enhance_decision_quality(
        const std::vector<std::vector<float>>& decision_options,
        const CognitiveEmotionalState& state
    );

private:
    std::unordered_map<std::string, float> config_;
    std::vector<float> meaning_history_;
};

/**
 * @brief Unified Vervaeke 4E Cognition Framework
 * 
 * Integrates all components of Vervaeke's 4E cognition framework
 * with salience landscape navigation, perspectival/participatory
 * knowing, processing modes, and cognitive-emotional integration.
 */
class Vervaeke4EFramework {
public:
    explicit Vervaeke4EFramework(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Process complete 4E cognitive cycle
     * @param sensory_input Current sensory input
     * @param environment Current environment state
     * @param goals Current goals
     * @return Updated 4E state
     */
    FourEState process_4e_cycle(
        const std::vector<float>& sensory_input,
        const std::vector<float>& environment,
        const std::vector<float>& goals
    );
    
    /**
     * @brief Get current 4E state
     * @return Current complete state
     */
    const FourEState& get_state() const { return state_; }
    
    /**
     * @brief Compute wisdom measure (Vervaeke framework)
     * @return Current wisdom level
     */
    float compute_wisdom_measure();
    
    /**
     * @brief Assess meaning connectivity (anti-meaning-crisis)
     * @return Meaning connection strength
     */
    float assess_meaning_connectivity();
    
    /**
     * @brief Reset framework to initial state
     */
    void reset();

private:
    std::unordered_map<std::string, float> config_;
    FourEState state_;
    
    // Component processors
    std::unique_ptr<EmbodiedCognitionProcessor> embodied_processor_;
    std::unique_ptr<EmbeddedCognitionProcessor> embedded_processor_;
    std::unique_ptr<EnactedCognitionProcessor> enacted_processor_;
    std::unique_ptr<ExtendedCognitionProcessor> extended_processor_;
    std::unique_ptr<SalienceLandscapeNavigator> salience_navigator_;
    std::unique_ptr<PerspectivalParticipatoryProcessor> perspectival_participatory_;
    std::unique_ptr<ProcessingModeManager> processing_manager_;
    std::unique_ptr<CognitiveEmotionalIntegrator> cognitive_emotional_;
    
    void update_overall_coherence();
    void update_wisdom_measure();
    void update_meaning_connectivity();
};

} // namespace vervaeke
} // namespace rros

#include "vervaeke_4e.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace rros {
namespace vervaeke {

namespace {
    // Helper functions
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
        
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return (denom > 1e-8f) ? (dot / denom) : 0.0f;
    }
    
    float compute_coherence(const std::vector<float>& vec) {
        if (vec.size() < 2) return 1.0f;
        
        float mean = std::accumulate(vec.begin(), vec.end(), 0.0f) / vec.size();
        float variance = 0.0f;
        for (float v : vec) {
            float diff = v - mean;
            variance += diff * diff;
        }
        variance /= vec.size();
        
        // Low variance = high coherence
        return 1.0f / (1.0f + std::sqrt(variance));
    }
    
    std::vector<float> normalize_vector(const std::vector<float>& vec) {
        std::vector<float> result = vec;
        float norm = std::sqrt(std::accumulate(vec.begin(), vec.end(), 0.0f,
            [](float sum, float v) { return sum + v * v; }));
        
        if (norm > 1e-8f) {
            for (float& v : result) {
                v /= norm;
            }
        }
        return result;
    }
}

//==============================================================================
// EmbodiedCognitionProcessor
//==============================================================================

EmbodiedCognitionProcessor::EmbodiedCognitionProcessor(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    // Initialize body schema memory with default proprioceptive state
    body_schema_memory_ = std::vector<float>(10, 0.5f);
}

SensorimotorState EmbodiedCognitionProcessor::process_sensorimotor(
    const std::vector<float>& sensory_input,
    const std::vector<float>& motor_context
) {
    SensorimotorState state;
    state.sensory_input = sensory_input;
    state.motor_output = motor_context;
    
    // Integrate sensory and motor for body schema
    state.body_schema.resize(std::max(sensory_input.size(), motor_context.size()));
    for (size_t i = 0; i < state.body_schema.size(); ++i) {
        float sensory = (i < sensory_input.size()) ? sensory_input[i] : 0.0f;
        float motor = (i < motor_context.size()) ? motor_context[i] : 0.0f;
        state.body_schema[i] = 0.6f * sensory + 0.4f * motor; // Weighted integration
    }
    
    // Compute sensorimotor coupling strength
    state.sensorimotor_coupling = cosine_similarity(sensory_input, motor_context);
    
    // Compute spatial awareness from embodied state
    state.spatial_awareness = compute_coherence(state.body_schema);
    
    return state;
}

void EmbodiedCognitionProcessor::update_body_schema(
    SensorimotorState& state,
    const std::vector<float>& feedback
) {
    // Update body schema with learning rate
    float learning_rate = config_.count("body_schema_lr") ? 
        config_.at("body_schema_lr") : 0.1f;
    
    for (size_t i = 0; i < std::min(state.body_schema.size(), feedback.size()); ++i) {
        state.body_schema[i] = state.body_schema[i] * (1.0f - learning_rate) +
                               feedback[i] * learning_rate;
    }
    
    // Store in memory for future reference
    body_schema_memory_ = state.body_schema;
}

std::vector<float> EmbodiedCognitionProcessor::compute_spatial_reasoning(
    const SensorimotorState& state
) {
    // Spatial reasoning emerges from embodied sensorimotor coupling
    std::vector<float> spatial(state.body_schema.size());
    
    for (size_t i = 0; i < spatial.size(); ++i) {
        // Combine body schema with sensorimotor coupling strength
        spatial[i] = state.body_schema[i] * state.sensorimotor_coupling;
        
        // Add spatial coherence factor
        spatial[i] *= state.spatial_awareness;
    }
    
    return normalize_vector(spatial);
}

//==============================================================================
// EmbeddedCognitionProcessor
//==============================================================================

EmbeddedCognitionProcessor::EmbeddedCognitionProcessor(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    context_history_.reserve(100); // Keep last 100 contexts
}

EnvironmentalContext EmbeddedCognitionProcessor::process_context_awareness(
    const std::vector<float>& physical_env,
    const std::vector<float>& social_env
) {
    EnvironmentalContext context;
    
    // Detect affordances from physical environment
    context.physical_affordances = detect_affordances(
        EnvironmentalContext{physical_env, social_env, {}, 0.0f, 0.0f}
    );
    
    context.social_context = social_env;
    
    // Temporal context from history
    if (!context_history_.empty()) {
        const auto& prev = context_history_.back();
        context.temporal_context = prev.physical_affordances;
    } else {
        context.temporal_context = std::vector<float>(physical_env.size(), 0.0f);
    }
    
    // Context stability: how similar to recent history
    if (!context_history_.empty()) {
        context.context_stability = cosine_similarity(
            physical_env, 
            context_history_.back().physical_affordances
        );
    } else {
        context.context_stability = 0.5f;
    }
    
    // Environmental coupling strength
    context.environmental_coupling = 
        0.5f * compute_coherence(physical_env) +
        0.5f * compute_coherence(social_env);
    
    // Store in history
    context_history_.push_back(context);
    if (context_history_.size() > 100) {
        context_history_.erase(context_history_.begin());
    }
    
    return context;
}

std::vector<float> EmbeddedCognitionProcessor::detect_affordances(
    const EnvironmentalContext& context
) {
    // Affordances are action possibilities in the environment
    std::vector<float> affordances(context.physical_affordances.size());
    
    for (size_t i = 0; i < affordances.size(); ++i) {
        float physical = (i < context.physical_affordances.size()) ? 
            context.physical_affordances[i] : 0.0f;
        float social = (i < context.social_context.size()) ? 
            context.social_context[i] : 0.0f;
        
        // Affordances emerge from physical and social context interaction
        affordances[i] = std::sqrt(physical * physical + social * social);
        
        // Modulate by environmental coupling
        affordances[i] *= context.environmental_coupling;
    }
    
    return normalize_vector(affordances);
}

std::vector<float> EmbeddedCognitionProcessor::adapt_to_context(
    const EnvironmentalContext& context,
    const std::vector<float>& behavior
) {
    std::vector<float> adapted(behavior.size());
    
    float adaptation_strength = config_.count("adaptation_strength") ?
        config_.at("adaptation_strength") : 0.7f;
    
    // Adapt behavior based on affordances and context stability
    for (size_t i = 0; i < adapted.size(); ++i) {
        float affordance = (i < context.physical_affordances.size()) ?
            context.physical_affordances[i] : 0.5f;
        
        // Stable contexts allow stronger adaptation
        float context_factor = context.context_stability * adaptation_strength;
        
        adapted[i] = behavior[i] * (1.0f - context_factor) +
                     affordance * context_factor;
    }
    
    return adapted;
}

//==============================================================================
// EnactedCognitionProcessor
//==============================================================================

EnactedCognitionProcessor::EnactedCognitionProcessor(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    action_outcome_history_.reserve(1000);
}

EnactedState EnactedCognitionProcessor::process_active_perception(
    const EnactedState& current_state,
    const std::vector<float>& action_space
) {
    EnactedState state = current_state;
    
    // Generate exploration strategy
    state.exploration_vector = generate_exploration_strategy(state);
    
    // Predict action outcomes based on history
    state.action_predictions.resize(action_space.size());
    for (size_t i = 0; i < action_space.size(); ++i) {
        float prediction = 0.0f;
        int count = 0;
        
        // Look for similar actions in history
        for (const auto& [action, outcome] : action_outcome_history_) {
            if (i < action.size() && std::abs(action[i] - action_space[i]) < 0.1f) {
                prediction += (i < outcome.size()) ? outcome[i] : 0.0f;
                ++count;
            }
        }
        
        state.action_predictions[i] = (count > 0) ? (prediction / count) : 0.5f;
    }
    
    // Exploration intensity based on prediction uncertainty
    float uncertainty = 1.0f - compute_coherence(state.action_predictions);
    state.exploration_intensity = uncertainty;
    
    // Enaction coherence from exploration and prediction alignment
    state.enaction_coherence = cosine_similarity(
        state.exploration_vector,
        state.action_predictions
    );
    
    return state;
}

std::vector<float> EnactedCognitionProcessor::generate_exploration_strategy(
    const EnactedState& state
) {
    size_t size = state.enacted_meaning.empty() ? 10 : state.enacted_meaning.size();
    std::vector<float> strategy(size);
    
    // Exploration balances novelty seeking with coherence
    float novelty_weight = config_.count("novelty_weight") ?
        config_.at("novelty_weight") : 0.3f;
    
    static std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.5f, 0.2f);
    
    for (size_t i = 0; i < size; ++i) {
        float random_explore = dist(rng);
        float coherent_explore = (i < state.enacted_meaning.size()) ?
            state.enacted_meaning[i] : 0.5f;
        
        strategy[i] = novelty_weight * random_explore +
                     (1.0f - novelty_weight) * coherent_explore;
    }
    
    return normalize_vector(strategy);
}

std::vector<float> EnactedCognitionProcessor::construct_enacted_meaning(
    const std::vector<float>& actions,
    const std::vector<float>& outcomes
) {
    // Meaning emerges from action-outcome coupling
    std::vector<float> meaning(std::max(actions.size(), outcomes.size()));
    
    for (size_t i = 0; i < meaning.size(); ++i) {
        float action = (i < actions.size()) ? actions[i] : 0.0f;
        float outcome = (i < outcomes.size()) ? outcomes[i] : 0.0f;
        
        // Meaning is the learned association
        meaning[i] = 0.5f * action + 0.5f * outcome;
    }
    
    // Store in history
    action_outcome_history_.emplace_back(actions, outcomes);
    if (action_outcome_history_.size() > 1000) {
        action_outcome_history_.erase(action_outcome_history_.begin());
    }
    
    return meaning;
}

//==============================================================================
// ExtendedCognitionProcessor
//==============================================================================

ExtendedCognitionProcessor::ExtendedCognitionProcessor(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    tool_proficiency_ = std::vector<float>(20, 0.3f); // Default low proficiency
}

ExtendedState ExtendedCognitionProcessor::process_cognitive_extension(
    const std::vector<float>& available_tools,
    const std::vector<float>& task
) {
    ExtendedState state;
    state.tool_representations = available_tools;
    
    // Match tools to task requirements
    state.tool_mastery = 0.0f;
    for (size_t i = 0; i < std::min(available_tools.size(), task.size()); ++i) {
        float tool_task_fit = 1.0f - std::abs(available_tools[i] - task[i]);
        float proficiency = (i < tool_proficiency_.size()) ? 
            tool_proficiency_[i] : 0.3f;
        
        state.tool_mastery += tool_task_fit * proficiency;
    }
    state.tool_mastery /= std::max(available_tools.size(), task.size());
    
    // Cognitive extension degree
    state.cognitive_extension = state.tool_mastery * 
        compute_coherence(available_tools);
    
    // Initialize symbolic scaffolds and distributed memory
    state.symbolic_scaffolds.resize(available_tools.size());
    state.distributed_memory.resize(task.size());
    
    return state;
}

void ExtendedCognitionProcessor::integrate_symbolic_scaffolds(
    ExtendedState& state,
    const std::vector<float>& symbols
) {
    state.symbolic_scaffolds = symbols;
    
    // Symbolic scaffolds enhance cognitive extension
    float symbol_strength = compute_coherence(symbols);
    state.cognitive_extension = 0.7f * state.cognitive_extension +
                               0.3f * symbol_strength;
}

std::vector<float> ExtendedCognitionProcessor::distribute_memory(
    const std::vector<float>& internal_memory,
    const std::vector<float>& environment
) {
    // Distribute memory based on environmental structure
    std::vector<float> distributed(internal_memory.size());
    
    float distribution_strength = config_.count("memory_distribution") ?
        config_.at("memory_distribution") : 0.5f;
    
    for (size_t i = 0; i < distributed.size(); ++i) {
        float internal = internal_memory[i];
        float external = (i < environment.size()) ? environment[i] : 0.0f;
        
        distributed[i] = internal * (1.0f - distribution_strength) +
                        external * distribution_strength;
    }
    
    return distributed;
}

//==============================================================================
// SalienceLandscapeNavigator
//==============================================================================

SalienceLandscapeNavigator::SalienceLandscapeNavigator(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    landscape_memory_.salience_field = std::vector<float>(50, 0.0f);
    landscape_memory_.relevance_gradients = std::vector<float>(50, 0.0f);
    landscape_memory_.attention_peaks = std::vector<float>(10, 0.0f);
}

SalienceLandscape SalienceLandscapeNavigator::update_landscape(
    const FourEState& context,
    const std::vector<float>& goals
) {
    SalienceLandscape landscape;
    
    // Salience field from 4E integration
    size_t field_size = std::max({
        context.embodied.sensory_input.size(),
        context.embedded.physical_affordances.size(),
        context.enacted.enacted_meaning.size(),
        goals.size()
    });
    
    landscape.salience_field.resize(field_size, 0.0f);
    
    for (size_t i = 0; i < field_size; ++i) {
        float embodied_sal = (i < context.embodied.sensory_input.size()) ?
            context.embodied.sensory_input[i] : 0.0f;
        float embedded_sal = (i < context.embedded.physical_affordances.size()) ?
            context.embedded.physical_affordances[i] : 0.0f;
        float enacted_sal = (i < context.enacted.enacted_meaning.size()) ?
            context.enacted.enacted_meaning[i] : 0.0f;
        float goal_sal = (i < goals.size()) ? goals[i] : 0.0f;
        
        // Integrate all sources of salience
        landscape.salience_field[i] = 
            0.25f * embodied_sal +
            0.25f * embedded_sal +
            0.25f * enacted_sal +
            0.25f * goal_sal;
    }
    
    // Compute relevance gradients
    landscape.relevance_gradients = compute_relevance_gradients(landscape);
    
    // Find attention peaks (local maxima in salience field)
    landscape.attention_peaks.clear();
    for (size_t i = 1; i < landscape.salience_field.size() - 1; ++i) {
        if (landscape.salience_field[i] > landscape.salience_field[i-1] &&
            landscape.salience_field[i] > landscape.salience_field[i+1]) {
            landscape.attention_peaks.push_back(landscape.salience_field[i]);
        }
    }
    if (landscape.attention_peaks.empty()) {
        landscape.attention_peaks.push_back(0.5f);
    }
    
    // Landscape coherence
    landscape.landscape_coherence = compute_coherence(landscape.salience_field);
    
    // Relevance realization rate (how quickly finding relevant features)
    landscape.relevance_realization_rate = 
        landscape.landscape_coherence * 
        (landscape.attention_peaks.size() / 10.0f); // Normalize to ~10 peaks
    
    landscape_memory_ = landscape;
    return landscape;
}

std::vector<float> SalienceLandscapeNavigator::navigate_toward_relevance(
    const SalienceLandscape& landscape,
    const std::vector<float>& position
) {
    std::vector<float> direction(position.size(), 0.0f);
    
    // Navigate in direction of relevance gradients
    for (size_t i = 0; i < std::min(direction.size(), landscape.relevance_gradients.size()); ++i) {
        direction[i] = landscape.relevance_gradients[i];
        
        // Modulate by landscape coherence
        direction[i] *= landscape.landscape_coherence;
    }
    
    return normalize_vector(direction);
}

std::vector<float> SalienceLandscapeNavigator::compute_relevance_gradients(
    const SalienceLandscape& landscape
) {
    std::vector<float> gradients(landscape.salience_field.size(), 0.0f);
    
    // Compute discrete gradients
    for (size_t i = 1; i < landscape.salience_field.size() - 1; ++i) {
        gradients[i] = (landscape.salience_field[i+1] - landscape.salience_field[i-1]) / 2.0f;
    }
    
    // Boundary conditions
    if (!gradients.empty()) {
        gradients[0] = landscape.salience_field[1] - landscape.salience_field[0];
        gradients.back() = landscape.salience_field.back() - 
                          landscape.salience_field[landscape.salience_field.size()-2];
    }
    
    return gradients;
}

//==============================================================================
// PerspectivalParticipatoryProcessor
//==============================================================================

PerspectivalParticipatoryProcessor::PerspectivalParticipatoryProcessor(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    // Initialize with a few default perspectives
    perspective_library_.push_back(std::vector<float>(10, 0.3f));
    perspective_library_.push_back(std::vector<float>(10, 0.7f));
}

PerspectivalState PerspectivalParticipatoryProcessor::process_perspectival_knowing(
    const std::vector<float>& current_perspective,
    const std::vector<std::vector<float>>& alternative_views
) {
    PerspectivalState state;
    state.current_perspective = current_perspective;
    state.alternative_perspectives = alternative_views;
    
    // Integrate multiple perspectives
    state.perspective_integration.resize(current_perspective.size(), 0.0f);
    
    float total_weight = 1.0f; // Current perspective weight
    state.perspective_integration = current_perspective;
    
    for (const auto& alt_view : alternative_views) {
        float view_weight = 1.0f / (alternative_views.size() + 1);
        total_weight += view_weight;
        
        for (size_t i = 0; i < std::min(state.perspective_integration.size(), alt_view.size()); ++i) {
            state.perspective_integration[i] += alt_view[i] * view_weight;
        }
    }
    
    // Normalize integration
    for (float& val : state.perspective_integration) {
        val /= total_weight;
    }
    
    // Perspective flexibility: how different are alternative views
    state.perspective_flexibility = 0.0f;
    for (const auto& alt_view : alternative_views) {
        float similarity = cosine_similarity(current_perspective, alt_view);
        state.perspective_flexibility += (1.0f - similarity);
    }
    if (!alternative_views.empty()) {
        state.perspective_flexibility /= alternative_views.size();
    }
    
    // Viewpoint richness: diversity of available perspectives
    state.viewpoint_richness = std::min(1.0f, 
        static_cast<float>(alternative_views.size()) / 5.0f);
    
    return state;
}

ParticipatoryState PerspectivalParticipatoryProcessor::process_participatory_knowing(
    const std::vector<float>& agent_state,
    const std::vector<float>& world_state
) {
    ParticipatoryState state;
    
    // Bidirectional agent-world coupling
    state.agent_world_coupling.resize(std::max(agent_state.size(), world_state.size()));
    
    for (size_t i = 0; i < state.agent_world_coupling.size(); ++i) {
        float agent = (i < agent_state.size()) ? agent_state[i] : 0.0f;
        float world = (i < world_state.size()) ? world_state[i] : 0.0f;
        
        // Coupling through mutual influence
        state.agent_world_coupling[i] = std::sqrt(agent * world);
    }
    
    // Agent transformation (how world shapes agent)
    state.transformation_state.resize(agent_state.size());
    for (size_t i = 0; i < agent_state.size(); ++i) {
        float world_influence = (i < world_state.size()) ? world_state[i] : 0.5f;
        state.transformation_state[i] = 0.7f * agent_state[i] + 0.3f * world_influence;
    }
    
    // World shaping (how agent shapes world)
    state.world_shaping.resize(world_state.size());
    for (size_t i = 0; i < world_state.size(); ++i) {
        float agent_influence = (i < agent_state.size()) ? agent_state[i] : 0.5f;
        state.world_shaping[i] = 0.7f * world_state[i] + 0.3f * agent_influence;
    }
    
    // Participation depth
    state.participation_depth = cosine_similarity(agent_state, world_state);
    
    // Reciprocal realization strength
    state.reciprocal_realization = compute_coherence(state.agent_world_coupling);
    
    return state;
}

bool PerspectivalParticipatoryProcessor::shift_perspective(
    PerspectivalState& state,
    const std::vector<float>& target_perspective
) {
    float shift_threshold = config_.count("perspective_shift_threshold") ?
        config_.at("perspective_shift_threshold") : 0.3f;
    
    // Can only shift if flexibility is high enough
    if (state.perspective_flexibility < shift_threshold) {
        return false;
    }
    
    // Perform shift
    state.alternative_perspectives.push_back(state.current_perspective);
    state.current_perspective = target_perspective;
    
    // Add to library
    perspective_library_.push_back(target_perspective);
    if (perspective_library_.size() > 20) {
        perspective_library_.erase(perspective_library_.begin());
    }
    
    return true;
}

//==============================================================================
// ProcessingModeManager
//==============================================================================

ProcessingModeManager::ProcessingModeManager(
    const std::unordered_map<std::string, float>& config
) : config_(config), current_mode_(ProcessingMode::BALANCED) {}

ProcessingMode ProcessingModeManager::determine_processing_mode(
    const FourEState& current_state,
    const std::vector<float>& challenge
) {
    // Determine if conformative or transformative processing needed
    
    // High coherence suggests conformative processing works
    if (current_state.overall_coherence > 0.7f) {
        current_mode_ = ProcessingMode::CONFORMATIVE;
        return current_mode_;
    }
    
    // Low coherence or high challenge difficulty suggests need for transformation
    float challenge_difficulty = 1.0f - compute_coherence(challenge);
    
    if (current_state.overall_coherence < 0.4f || challenge_difficulty > 0.6f) {
        current_mode_ = ProcessingMode::TRANSFORMATIVE;
        return current_mode_;
    }
    
    // Default to balanced
    current_mode_ = ProcessingMode::BALANCED;
    return current_mode_;
}

std::vector<float> ProcessingModeManager::process_conformative(
    const std::vector<float>& input,
    const std::vector<float>& existing_patterns
) {
    // Assimilation: fit new input to existing patterns
    std::vector<float> output(input.size());
    
    float assimilation_strength = config_.count("assimilation_strength") ?
        config_.at("assimilation_strength") : 0.7f;
    
    for (size_t i = 0; i < output.size(); ++i) {
        float pattern = (i < existing_patterns.size()) ? 
            existing_patterns[i] : 0.5f;
        
        // Pull input toward existing pattern
        output[i] = input[i] * (1.0f - assimilation_strength) +
                    pattern * assimilation_strength;
    }
    
    return output;
}

std::vector<float> ProcessingModeManager::process_transformative(
    const std::vector<float>& input,
    const std::vector<float>& constraints
) {
    // Accommodation: restructure patterns to fit new input (insight)
    std::vector<float> output(input.size());
    
    float transformation_strength = config_.count("transformation_strength") ?
        config_.at("transformation_strength") : 0.8f;
    
    // Transformative processing allows larger deviations from constraints
    for (size_t i = 0; i < output.size(); ++i) {
        float constraint = (i < constraints.size()) ? constraints[i] : 0.5f;
        
        // Allow input to reshape the pattern space
        output[i] = input[i] * transformation_strength +
                    constraint * (1.0f - transformation_strength);
        
        // Add nonlinearity for insight-like restructuring
        output[i] = std::tanh(output[i] * 2.0f - 1.0f) * 0.5f + 0.5f;
    }
    
    return output;
}

//==============================================================================
// CognitiveEmotionalIntegrator
//==============================================================================

CognitiveEmotionalIntegrator::CognitiveEmotionalIntegrator(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    meaning_history_.reserve(100);
}

CognitiveEmotionalState CognitiveEmotionalIntegrator::integrate_cognition_emotion(
    const std::vector<float>& cognitive_state,
    const std::vector<float>& emotional_state
) {
    CognitiveEmotionalState state;
    
    // Emotional valence from emotion state
    state.emotional_valence = emotional_state;
    
    // Motivational forces from cognitive-emotional interaction
    state.motivational_forces.resize(
        std::max(cognitive_state.size(), emotional_state.size())
    );
    
    for (size_t i = 0; i < state.motivational_forces.size(); ++i) {
        float cog = (i < cognitive_state.size()) ? cognitive_state[i] : 0.0f;
        float emo = (i < emotional_state.size()) ? emotional_state[i] : 0.0f;
        
        // Motivation emerges from cognitive-emotional synergy
        state.motivational_forces[i] = std::sqrt(cog * cog + emo * emo);
    }
    
    // Meaning coherence
    state.meaning_coherence.resize(state.motivational_forces.size());
    for (size_t i = 0; i < state.meaning_coherence.size(); ++i) {
        state.meaning_coherence[i] = state.motivational_forces[i] *
            cosine_similarity(cognitive_state, emotional_state);
    }
    
    // Emotional regulation capacity
    state.emotional_regulation = 1.0f - compute_coherence(emotional_state);
    
    // Cognitive-emotional synchronization
    state.cognitive_emotional_sync = cosine_similarity(cognitive_state, emotional_state);
    
    // Store meaning for history
    float meaning_value = compute_meaning_coherence(state);
    meaning_history_.push_back(meaning_value);
    if (meaning_history_.size() > 100) {
        meaning_history_.erase(meaning_history_.begin());
    }
    
    return state;
}

float CognitiveEmotionalIntegrator::compute_meaning_coherence(
    const CognitiveEmotionalState& state
) {
    if (state.meaning_coherence.empty()) return 0.0f;
    
    // Meaning coherence is average of all meaning dimensions
    float total = std::accumulate(
        state.meaning_coherence.begin(),
        state.meaning_coherence.end(),
        0.0f
    );
    
    return total / state.meaning_coherence.size();
}

std::vector<float> CognitiveEmotionalIntegrator::enhance_decision_quality(
    const std::vector<std::vector<float>>& decision_options,
    const CognitiveEmotionalState& state
) {
    std::vector<float> quality_scores(decision_options.size());
    
    // Evaluate each option's alignment with cognitive-emotional state
    for (size_t i = 0; i < decision_options.size(); ++i) {
        const auto& option = decision_options[i];
        
        // Quality from emotional alignment
        float emotional_fit = cosine_similarity(option, state.emotional_valence);
        
        // Quality from motivational alignment
        float motivational_fit = cosine_similarity(option, state.motivational_forces);
        
        // Quality from meaning coherence
        float meaning_fit = cosine_similarity(option, state.meaning_coherence);
        
        // Combined quality score
        quality_scores[i] = 
            0.3f * emotional_fit +
            0.4f * motivational_fit +
            0.3f * meaning_fit;
        
        // Boost by cognitive-emotional sync
        quality_scores[i] *= (0.5f + 0.5f * state.cognitive_emotional_sync);
    }
    
    return quality_scores;
}

//==============================================================================
// Vervaeke4EFramework (Main Integration)
//==============================================================================

Vervaeke4EFramework::Vervaeke4EFramework(
    const std::unordered_map<std::string, float>& config
) : config_(config) {
    // Initialize all component processors
    embodied_processor_ = std::make_unique<EmbodiedCognitionProcessor>(config);
    embedded_processor_ = std::make_unique<EmbeddedCognitionProcessor>(config);
    enacted_processor_ = std::make_unique<EnactedCognitionProcessor>(config);
    extended_processor_ = std::make_unique<ExtendedCognitionProcessor>(config);
    salience_navigator_ = std::make_unique<SalienceLandscapeNavigator>(config);
    perspectival_participatory_ = std::make_unique<PerspectivalParticipatoryProcessor>(config);
    processing_manager_ = std::make_unique<ProcessingModeManager>(config);
    cognitive_emotional_ = std::make_unique<CognitiveEmotionalIntegrator>(config);
    
    // Initialize state
    reset();
}

FourEState Vervaeke4EFramework::process_4e_cycle(
    const std::vector<float>& sensory_input,
    const std::vector<float>& environment,
    const std::vector<float>& goals
) {
    // 1. Embodied processing: sensorimotor integration
    state_.embodied = embodied_processor_->process_sensorimotor(
        sensory_input,
        state_.embodied.motor_output // Use previous motor state
    );
    
    // 2. Embedded processing: context awareness
    std::vector<float> social_context(environment.size() / 2);
    std::copy(environment.begin(), 
              environment.begin() + std::min(social_context.size(), environment.size()),
              social_context.begin());
    
    state_.embedded = embedded_processor_->process_context_awareness(
        environment,
        social_context
    );
    
    // 3. Enacted processing: active perception
    std::vector<float> action_space = 
        embodied_processor_->compute_spatial_reasoning(state_.embodied);
    
    state_.enacted = enacted_processor_->process_active_perception(
        state_.enacted,
        action_space
    );
    
    // 4. Extended processing: cognitive extension
    std::vector<float> available_tools = state_.extended.tool_representations;
    if (available_tools.empty()) {
        available_tools.resize(10, 0.5f);
    }
    
    state_.extended = extended_processor_->process_cognitive_extension(
        available_tools,
        goals
    );
    
    // 5. Salience landscape navigation
    state_.salience = salience_navigator_->update_landscape(state_, goals);
    
    // 6. Perspectival knowing
    std::vector<std::vector<float>> alternative_perspectives;
    alternative_perspectives.push_back(environment);
    alternative_perspectives.push_back(goals);
    
    state_.perspectival = perspectival_participatory_->process_perspectival_knowing(
        sensory_input,
        alternative_perspectives
    );
    
    // 7. Participatory knowing
    state_.participatory = perspectival_participatory_->process_participatory_knowing(
        sensory_input,
        environment
    );
    
    // 8. Determine processing mode
    state_.processing_mode = processing_manager_->determine_processing_mode(
        state_,
        goals
    );
    
    // 9. Cognitive-emotional integration
    std::vector<float> cognitive = state_.enacted.enacted_meaning;
    std::vector<float> emotional = state_.salience.salience_field;
    
    state_.cognitive_emotional = cognitive_emotional_->integrate_cognition_emotion(
        cognitive,
        emotional
    );
    
    // 10. Update overall metrics
    update_overall_coherence();
    update_wisdom_measure();
    update_meaning_connectivity();
    
    return state_;
}

void Vervaeke4EFramework::update_overall_coherence() {
    // Overall coherence from all subsystems
    float embodied_coh = state_.embodied.sensorimotor_coupling;
    float embedded_coh = state_.embedded.environmental_coupling;
    float enacted_coh = state_.enacted.enaction_coherence;
    float extended_coh = state_.extended.cognitive_extension;
    float salience_coh = state_.salience.landscape_coherence;
    float perspectival_coh = state_.perspectival.viewpoint_richness;
    float participatory_coh = state_.participatory.reciprocal_realization;
    float cogemo_coh = state_.cognitive_emotional.cognitive_emotional_sync;
    
    state_.overall_coherence = (
        embodied_coh + embedded_coh + enacted_coh + extended_coh +
        salience_coh + perspectival_coh + participatory_coh + cogemo_coh
    ) / 8.0f;
}

void Vervaeke4EFramework::update_wisdom_measure() {
    // Wisdom per Vervaeke: ability to zero in on relevant information
    // and overcome self-deception through perspectival flexibility
    
    float relevance_realization = state_.salience.relevance_realization_rate;
    float perspectival_flexibility = state_.perspectival.perspective_flexibility;
    float participation_depth = state_.participatory.participation_depth;
    float meaning_coherence = cognitive_emotional_->compute_meaning_coherence(
        state_.cognitive_emotional
    );
    
    state_.wisdom_measure = (
        0.3f * relevance_realization +
        0.3f * perspectival_flexibility +
        0.2f * participation_depth +
        0.2f * meaning_coherence
    );
}

void Vervaeke4EFramework::update_meaning_connectivity() {
    // Meaning connectivity: connections between mind-body, mind-world, mind-mind
    
    // Mind-body connection (embodied cognition)
    float mind_body = state_.embodied.sensorimotor_coupling;
    
    // Mind-world connection (embedded + participatory)
    float mind_world = 0.5f * state_.embedded.environmental_coupling +
                       0.5f * state_.participatory.participation_depth;
    
    // Mind-mind connection (perspectival knowing + social context)
    float mind_mind = state_.perspectival.viewpoint_richness;
    
    // Self-connection (cognitive-emotional integration)
    float mind_self = state_.cognitive_emotional.cognitive_emotional_sync;
    
    state_.meaning_connectivity = (
        mind_body + mind_world + mind_mind + mind_self
    ) / 4.0f;
}

float Vervaeke4EFramework::compute_wisdom_measure() {
    update_wisdom_measure();
    return state_.wisdom_measure;
}

float Vervaeke4EFramework::assess_meaning_connectivity() {
    update_meaning_connectivity();
    return state_.meaning_connectivity;
}

void Vervaeke4EFramework::reset() {
    state_ = FourEState();
    state_.processing_mode = ProcessingMode::BALANCED;
    state_.overall_coherence = 0.5f;
    state_.wisdom_measure = 0.3f;
    state_.meaning_connectivity = 0.4f;
}

} // namespace vervaeke
} // namespace rros

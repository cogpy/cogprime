#include "../core/vervaeke_4e.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

using namespace rros::vervaeke;

// Test result tracking
int tests_passed = 0;
int tests_total = 0;

#define TEST(name) \
    ++tests_total; \
    std::cout << "🧪 Testing " #name "..." << std::endl; \
    if (test_##name())

#define PASS() \
    do { \
        ++tests_passed; \
        std::cout << "✅ Test passed" << std::endl; \
    } while(0)

#define ASSERT_NEAR(a, b, tol) \
    assert(std::abs((a) - (b)) < (tol))

#define ASSERT_TRUE(cond) \
    assert(cond)

#define ASSERT_FALSE(cond) \
    assert(!(cond))

#define ASSERT_EQ(a, b) \
    assert((a) == (b))

// Helper to create test vector
std::vector<float> make_vector(std::initializer_list<float> values) {
    return std::vector<float>(values);
}

//==============================================================================
// Test: Embodied Cognition
//==============================================================================

bool test_embodied_cognition_basic() {
    std::unordered_map<std::string, float> config;
    config["body_schema_lr"] = 0.1f;
    
    EmbodiedCognitionProcessor processor(config);
    
    auto sensory = make_vector({0.1f, 0.3f, 0.5f, 0.7f, 0.9f});
    auto motor = make_vector({0.2f, 0.4f, 0.6f, 0.8f, 1.0f});
    
    auto state = processor.process_sensorimotor(sensory, motor);
    
    // Check state properties
    ASSERT_EQ(state.sensory_input.size(), 5);
    ASSERT_EQ(state.motor_output.size(), 5);
    ASSERT_TRUE(state.sensorimotor_coupling >= 0.0f);
    ASSERT_TRUE(state.sensorimotor_coupling <= 1.0f);
    ASSERT_TRUE(state.spatial_awareness >= 0.0f);
    ASSERT_TRUE(state.spatial_awareness <= 1.0f);
    
    std::cout << "  - Sensorimotor coupling: " << state.sensorimotor_coupling << std::endl;
    std::cout << "  - Spatial awareness: " << state.spatial_awareness << std::endl;
    
    return true;
}

bool test_embodied_body_schema_update() {
    std::unordered_map<std::string, float> config;
    config["body_schema_lr"] = 0.2f;
    
    EmbodiedCognitionProcessor processor(config);
    
    auto sensory = make_vector({0.5f, 0.5f, 0.5f});
    auto motor = make_vector({0.5f, 0.5f, 0.5f});
    
    auto state = processor.process_sensorimotor(sensory, motor);
    auto initial_schema = state.body_schema;
    
    // Update with feedback
    auto feedback = make_vector({0.8f, 0.8f, 0.8f});
    processor.update_body_schema(state, feedback);
    
    // Schema should have changed toward feedback
    for (size_t i = 0; i < state.body_schema.size(); ++i) {
        ASSERT_TRUE(state.body_schema[i] > initial_schema[i]);
    }
    
    std::cout << "  - Body schema updated successfully" << std::endl;
    return true;
}

bool test_embodied_spatial_reasoning() {
    std::unordered_map<std::string, float> config;
    EmbodiedCognitionProcessor processor(config);
    
    auto sensory = make_vector({0.3f, 0.5f, 0.7f});
    auto motor = make_vector({0.4f, 0.6f, 0.8f});
    
    auto state = processor.process_sensorimotor(sensory, motor);
    auto spatial = processor.compute_spatial_reasoning(state);
    
    ASSERT_FALSE(spatial.empty());
    std::cout << "  - Spatial reasoning computed" << std::endl;
    
    return true;
}

//==============================================================================
// Test: Embedded Cognition
//==============================================================================

bool test_embedded_context_awareness() {
    std::unordered_map<std::string, float> config;
    config["adaptation_strength"] = 0.7f;
    
    EmbeddedCognitionProcessor processor(config);
    
    auto physical = make_vector({0.2f, 0.4f, 0.6f, 0.8f});
    auto social = make_vector({0.3f, 0.5f, 0.7f, 0.9f});
    
    auto context = processor.process_context_awareness(physical, social);
    
    ASSERT_FALSE(context.physical_affordances.empty());
    ASSERT_FALSE(context.social_context.empty());
    ASSERT_TRUE(context.context_stability >= 0.0f);
    ASSERT_TRUE(context.environmental_coupling >= 0.0f);
    
    std::cout << "  - Environmental coupling: " << context.environmental_coupling << std::endl;
    
    return true;
}

bool test_embedded_affordance_detection() {
    std::unordered_map<std::string, float> config;
    EmbeddedCognitionProcessor processor(config);
    
    EnvironmentalContext context;
    context.physical_affordances = make_vector({0.1f, 0.5f, 0.9f});
    context.social_context = make_vector({0.2f, 0.6f, 0.8f});
    context.environmental_coupling = 0.7f;
    
    auto affordances = processor.detect_affordances(context);
    
    ASSERT_FALSE(affordances.empty());
    std::cout << "  - Affordances detected" << std::endl;
    
    return true;
}

bool test_embedded_context_adaptation() {
    std::unordered_map<std::string, float> config;
    config["adaptation_strength"] = 0.6f;
    
    EmbeddedCognitionProcessor processor(config);
    
    EnvironmentalContext context;
    context.physical_affordances = make_vector({0.8f, 0.7f, 0.6f});
    context.context_stability = 0.8f;
    
    auto behavior = make_vector({0.2f, 0.3f, 0.4f});
    auto adapted = processor.adapt_to_context(context, behavior);
    
    ASSERT_EQ(adapted.size(), behavior.size());
    std::cout << "  - Behavior adapted to context" << std::endl;
    
    return true;
}

//==============================================================================
// Test: Enacted Cognition
//==============================================================================

bool test_enacted_active_perception() {
    std::unordered_map<std::string, float> config;
    config["novelty_weight"] = 0.3f;
    
    EnactedCognitionProcessor processor(config);
    
    EnactedState current_state;
    current_state.enacted_meaning = make_vector({0.4f, 0.5f, 0.6f});
    
    auto action_space = make_vector({0.2f, 0.4f, 0.6f, 0.8f});
    
    auto state = processor.process_active_perception(current_state, action_space);
    
    ASSERT_FALSE(state.exploration_vector.empty());
    ASSERT_FALSE(state.action_predictions.empty());
    ASSERT_TRUE(state.exploration_intensity >= 0.0f);
    ASSERT_TRUE(state.exploration_intensity <= 1.0f);
    
    std::cout << "  - Exploration intensity: " << state.exploration_intensity << std::endl;
    
    return true;
}

bool test_enacted_meaning_construction() {
    std::unordered_map<std::string, float> config;
    EnactedCognitionProcessor processor(config);
    
    auto actions = make_vector({0.3f, 0.6f, 0.9f});
    auto outcomes = make_vector({0.4f, 0.7f, 0.8f});
    
    auto meaning = processor.construct_enacted_meaning(actions, outcomes);
    
    ASSERT_FALSE(meaning.empty());
    std::cout << "  - Enacted meaning constructed" << std::endl;
    
    return true;
}

//==============================================================================
// Test: Extended Cognition
//==============================================================================

bool test_extended_cognitive_extension() {
    std::unordered_map<std::string, float> config;
    ExtendedCognitionProcessor processor(config);
    
    auto tools = make_vector({0.5f, 0.6f, 0.7f, 0.8f});
    auto task = make_vector({0.4f, 0.5f, 0.6f, 0.7f});
    
    auto state = processor.process_cognitive_extension(tools, task);
    
    ASSERT_FALSE(state.tool_representations.empty());
    ASSERT_TRUE(state.tool_mastery >= 0.0f);
    ASSERT_TRUE(state.tool_mastery <= 1.0f);
    ASSERT_TRUE(state.cognitive_extension >= 0.0f);
    
    std::cout << "  - Tool mastery: " << state.tool_mastery << std::endl;
    std::cout << "  - Cognitive extension: " << state.cognitive_extension << std::endl;
    
    return true;
}

bool test_extended_symbolic_scaffolds() {
    std::unordered_map<std::string, float> config;
    ExtendedCognitionProcessor processor(config);
    
    auto tools = make_vector({0.5f, 0.5f});
    auto task = make_vector({0.5f, 0.5f});
    auto state = processor.process_cognitive_extension(tools, task);
    
    auto symbols = make_vector({0.7f, 0.8f, 0.9f});
    processor.integrate_symbolic_scaffolds(state, symbols);
    
    ASSERT_EQ(state.symbolic_scaffolds.size(), symbols.size());
    std::cout << "  - Symbolic scaffolds integrated" << std::endl;
    
    return true;
}

bool test_extended_memory_distribution() {
    std::unordered_map<std::string, float> config;
    config["memory_distribution"] = 0.5f;
    
    ExtendedCognitionProcessor processor(config);
    
    auto internal_mem = make_vector({0.3f, 0.5f, 0.7f});
    auto environment = make_vector({0.6f, 0.4f, 0.2f});
    
    auto distributed = processor.distribute_memory(internal_mem, environment);
    
    ASSERT_EQ(distributed.size(), internal_mem.size());
    std::cout << "  - Memory distributed into environment" << std::endl;
    
    return true;
}

//==============================================================================
// Test: Salience Landscape
//==============================================================================

bool test_salience_landscape_update() {
    std::unordered_map<std::string, float> config;
    SalienceLandscapeNavigator navigator(config);
    
    FourEState context;
    context.embodied.sensory_input = make_vector({0.2f, 0.4f, 0.6f});
    context.embedded.physical_affordances = make_vector({0.3f, 0.5f, 0.7f});
    context.enacted.enacted_meaning = make_vector({0.4f, 0.6f, 0.8f});
    
    auto goals = make_vector({0.5f, 0.7f, 0.9f});
    
    auto landscape = navigator.update_landscape(context, goals);
    
    ASSERT_FALSE(landscape.salience_field.empty());
    ASSERT_FALSE(landscape.relevance_gradients.empty());
    ASSERT_TRUE(landscape.landscape_coherence >= 0.0f);
    ASSERT_TRUE(landscape.relevance_realization_rate >= 0.0f);
    
    std::cout << "  - Landscape coherence: " << landscape.landscape_coherence << std::endl;
    std::cout << "  - Relevance realization rate: " << landscape.relevance_realization_rate << std::endl;
    
    return true;
}

bool test_salience_navigation() {
    std::unordered_map<std::string, float> config;
    SalienceLandscapeNavigator navigator(config);
    
    SalienceLandscape landscape;
    landscape.salience_field = make_vector({0.1f, 0.5f, 0.9f, 0.3f, 0.7f});
    landscape.relevance_gradients = make_vector({0.2f, 0.4f, 0.6f, 0.8f, 1.0f});
    landscape.landscape_coherence = 0.7f;
    
    auto position = make_vector({0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
    auto direction = navigator.navigate_toward_relevance(landscape, position);
    
    ASSERT_FALSE(direction.empty());
    std::cout << "  - Navigation direction computed" << std::endl;
    
    return true;
}

//==============================================================================
// Test: Perspectival and Participatory Knowing
//==============================================================================

bool test_perspectival_knowing() {
    std::unordered_map<std::string, float> config;
    PerspectivalParticipatoryProcessor processor(config);
    
    auto current = make_vector({0.3f, 0.5f, 0.7f});
    std::vector<std::vector<float>> alternatives = {
        make_vector({0.4f, 0.6f, 0.8f}),
        make_vector({0.2f, 0.4f, 0.6f})
    };
    
    auto state = processor.process_perspectival_knowing(current, alternatives);
    
    ASSERT_FALSE(state.current_perspective.empty());
    ASSERT_FALSE(state.perspective_integration.empty());
    ASSERT_TRUE(state.perspective_flexibility >= 0.0f);
    ASSERT_TRUE(state.viewpoint_richness >= 0.0f);
    
    std::cout << "  - Perspective flexibility: " << state.perspective_flexibility << std::endl;
    std::cout << "  - Viewpoint richness: " << state.viewpoint_richness << std::endl;
    
    return true;
}

bool test_participatory_knowing() {
    std::unordered_map<std::string, float> config;
    PerspectivalParticipatoryProcessor processor(config);
    
    auto agent = make_vector({0.4f, 0.5f, 0.6f});
    auto world = make_vector({0.6f, 0.5f, 0.4f});
    
    auto state = processor.process_participatory_knowing(agent, world);
    
    ASSERT_FALSE(state.agent_world_coupling.empty());
    ASSERT_FALSE(state.transformation_state.empty());
    ASSERT_FALSE(state.world_shaping.empty());
    ASSERT_TRUE(state.participation_depth >= 0.0f);
    ASSERT_TRUE(state.reciprocal_realization >= 0.0f);
    
    std::cout << "  - Participation depth: " << state.participation_depth << std::endl;
    std::cout << "  - Reciprocal realization: " << state.reciprocal_realization << std::endl;
    
    return true;
}

bool test_perspective_shift() {
    std::unordered_map<std::string, float> config;
    config["perspective_shift_threshold"] = 0.3f;
    
    PerspectivalParticipatoryProcessor processor(config);
    
    auto current = make_vector({0.3f, 0.5f, 0.7f});
    std::vector<std::vector<float>> alternatives = {
        make_vector({0.8f, 0.2f, 0.5f}),  // Very different
        make_vector({0.1f, 0.9f, 0.3f})
    };
    
    auto state = processor.process_perspectival_knowing(current, alternatives);
    
    auto target = make_vector({0.7f, 0.3f, 0.9f});
    bool shifted = processor.shift_perspective(state, target);
    
    if (shifted) {
        std::cout << "  - Perspective shift successful" << std::endl;
    } else {
        std::cout << "  - Perspective shift blocked (low flexibility)" << std::endl;
    }
    
    return true;
}

//==============================================================================
// Test: Processing Modes
//==============================================================================

bool test_processing_mode_determination() {
    std::unordered_map<std::string, float> config;
    ProcessingModeManager manager(config);
    
    FourEState high_coherence_state;
    high_coherence_state.overall_coherence = 0.8f;
    
    auto easy_challenge = make_vector({0.5f, 0.5f, 0.5f});
    auto mode = manager.determine_processing_mode(high_coherence_state, easy_challenge);
    
    ASSERT_EQ(mode, ProcessingMode::CONFORMATIVE);
    std::cout << "  - High coherence → Conformative mode" << std::endl;
    
    FourEState low_coherence_state;
    low_coherence_state.overall_coherence = 0.3f;
    
    auto hard_challenge = make_vector({0.1f, 0.9f, 0.2f, 0.8f});
    mode = manager.determine_processing_mode(low_coherence_state, hard_challenge);
    
    ASSERT_EQ(mode, ProcessingMode::TRANSFORMATIVE);
    std::cout << "  - Low coherence → Transformative mode" << std::endl;
    
    return true;
}

bool test_conformative_processing() {
    std::unordered_map<std::string, float> config;
    config["assimilation_strength"] = 0.7f;
    
    ProcessingModeManager manager(config);
    
    auto input = make_vector({0.3f, 0.7f, 0.5f});
    auto patterns = make_vector({0.5f, 0.5f, 0.5f});
    
    auto output = manager.process_conformative(input, patterns);
    
    ASSERT_EQ(output.size(), input.size());
    
    // Output should be pulled toward patterns
    for (size_t i = 0; i < output.size(); ++i) {
        float distance_to_pattern = std::abs(output[i] - patterns[i]);
        float input_distance = std::abs(input[i] - patterns[i]);
        ASSERT_TRUE(distance_to_pattern <= input_distance);
    }
    
    std::cout << "  - Conformative processing successful" << std::endl;
    return true;
}

bool test_transformative_processing() {
    std::unordered_map<std::string, float> config;
    config["transformation_strength"] = 0.8f;
    
    ProcessingModeManager manager(config);
    
    auto input = make_vector({0.2f, 0.8f, 0.3f});
    auto constraints = make_vector({0.5f, 0.5f, 0.5f});
    
    auto output = manager.process_transformative(input, constraints);
    
    ASSERT_EQ(output.size(), input.size());
    std::cout << "  - Transformative processing successful" << std::endl;
    
    return true;
}

//==============================================================================
// Test: Cognitive-Emotional Integration
//==============================================================================

bool test_cognitive_emotional_integration() {
    std::unordered_map<std::string, float> config;
    CognitiveEmotionalIntegrator integrator(config);
    
    auto cognitive = make_vector({0.4f, 0.6f, 0.8f});
    auto emotional = make_vector({0.5f, 0.7f, 0.6f});
    
    auto state = integrator.integrate_cognition_emotion(cognitive, emotional);
    
    ASSERT_FALSE(state.emotional_valence.empty());
    ASSERT_FALSE(state.motivational_forces.empty());
    ASSERT_FALSE(state.meaning_coherence.empty());
    ASSERT_TRUE(state.emotional_regulation >= 0.0f);
    ASSERT_TRUE(state.cognitive_emotional_sync >= 0.0f);
    ASSERT_TRUE(state.cognitive_emotional_sync <= 1.0f);
    
    std::cout << "  - Cognitive-emotional sync: " << state.cognitive_emotional_sync << std::endl;
    
    return true;
}

bool test_meaning_coherence_computation() {
    std::unordered_map<std::string, float> config;
    CognitiveEmotionalIntegrator integrator(config);
    
    auto cognitive = make_vector({0.5f, 0.5f, 0.5f});
    auto emotional = make_vector({0.5f, 0.5f, 0.5f});
    
    auto state = integrator.integrate_cognition_emotion(cognitive, emotional);
    float meaning = integrator.compute_meaning_coherence(state);
    
    ASSERT_TRUE(meaning >= 0.0f);
    ASSERT_TRUE(meaning <= 1.0f);
    
    std::cout << "  - Meaning coherence: " << meaning << std::endl;
    
    return true;
}

bool test_decision_quality_enhancement() {
    std::unordered_map<std::string, float> config;
    CognitiveEmotionalIntegrator integrator(config);
    
    auto cognitive = make_vector({0.6f, 0.7f, 0.8f});
    auto emotional = make_vector({0.5f, 0.6f, 0.7f});
    
    auto state = integrator.integrate_cognition_emotion(cognitive, emotional);
    
    std::vector<std::vector<float>> options = {
        make_vector({0.5f, 0.6f, 0.7f}),
        make_vector({0.3f, 0.4f, 0.5f}),
        make_vector({0.8f, 0.9f, 0.7f})
    };
    
    auto quality_scores = integrator.enhance_decision_quality(options, state);
    
    ASSERT_EQ(quality_scores.size(), 3);
    std::cout << "  - Decision quality scores computed" << std::endl;
    
    return true;
}

//==============================================================================
// Test: Full 4E Framework Integration
//==============================================================================

bool test_4e_framework_initialization() {
    std::unordered_map<std::string, float> config;
    Vervaeke4EFramework framework(config);
    
    const auto& state = framework.get_state();
    
    ASSERT_EQ(state.processing_mode, ProcessingMode::BALANCED);
    ASSERT_TRUE(state.overall_coherence >= 0.0f);
    ASSERT_TRUE(state.wisdom_measure >= 0.0f);
    ASSERT_TRUE(state.meaning_connectivity >= 0.0f);
    
    std::cout << "  - Framework initialized successfully" << std::endl;
    
    return true;
}

bool test_4e_complete_cycle() {
    std::unordered_map<std::string, float> config;
    Vervaeke4EFramework framework(config);
    
    auto sensory = make_vector({0.2f, 0.4f, 0.6f, 0.8f});
    auto environment = make_vector({0.3f, 0.5f, 0.7f, 0.9f});
    auto goals = make_vector({0.4f, 0.6f, 0.8f, 1.0f});
    
    auto state = framework.process_4e_cycle(sensory, environment, goals);
    
    // Verify all components updated
    ASSERT_FALSE(state.embodied.sensory_input.empty());
    ASSERT_FALSE(state.embedded.physical_affordances.empty());
    ASSERT_FALSE(state.enacted.exploration_vector.empty());
    ASSERT_FALSE(state.extended.tool_representations.empty());
    ASSERT_FALSE(state.salience.salience_field.empty());
    ASSERT_FALSE(state.perspectival.current_perspective.empty());
    ASSERT_FALSE(state.participatory.agent_world_coupling.empty());
    ASSERT_FALSE(state.cognitive_emotional.emotional_valence.empty());
    
    std::cout << "  - Overall coherence: " << state.overall_coherence << std::endl;
    std::cout << "  - Wisdom measure: " << state.wisdom_measure << std::endl;
    std::cout << "  - Meaning connectivity: " << state.meaning_connectivity << std::endl;
    
    return true;
}

bool test_4e_wisdom_computation() {
    std::unordered_map<std::string, float> config;
    Vervaeke4EFramework framework(config);
    
    auto sensory = make_vector({0.5f, 0.6f, 0.7f});
    auto environment = make_vector({0.6f, 0.7f, 0.8f});
    auto goals = make_vector({0.7f, 0.8f, 0.9f});
    
    framework.process_4e_cycle(sensory, environment, goals);
    
    float wisdom = framework.compute_wisdom_measure();
    
    ASSERT_TRUE(wisdom >= 0.0f);
    ASSERT_TRUE(wisdom <= 1.0f);
    
    std::cout << "  - Wisdom measure: " << wisdom << std::endl;
    
    return true;
}

bool test_4e_meaning_connectivity() {
    std::unordered_map<std::string, float> config;
    Vervaeke4EFramework framework(config);
    
    auto sensory = make_vector({0.4f, 0.5f, 0.6f});
    auto environment = make_vector({0.5f, 0.6f, 0.7f});
    auto goals = make_vector({0.6f, 0.7f, 0.8f});
    
    framework.process_4e_cycle(sensory, environment, goals);
    
    float meaning = framework.assess_meaning_connectivity();
    
    ASSERT_TRUE(meaning >= 0.0f);
    ASSERT_TRUE(meaning <= 1.0f);
    
    std::cout << "  - Meaning connectivity (anti-meaning-crisis): " << meaning << std::endl;
    
    return true;
}

bool test_4e_framework_reset() {
    std::unordered_map<std::string, float> config;
    Vervaeke4EFramework framework(config);
    
    auto sensory = make_vector({0.5f, 0.6f, 0.7f});
    auto environment = make_vector({0.6f, 0.7f, 0.8f});
    auto goals = make_vector({0.7f, 0.8f, 0.9f});
    
    framework.process_4e_cycle(sensory, environment, goals);
    framework.reset();
    
    const auto& state = framework.get_state();
    
    ASSERT_EQ(state.processing_mode, ProcessingMode::BALANCED);
    
    std::cout << "  - Framework reset successfully" << std::endl;
    
    return true;
}

//==============================================================================
// Main Test Runner
//==============================================================================

int main() {
    std::cout << "🚀 Starting Vervaeke 4E Cognition Framework Test Suite" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << std::endl;
    
    // Embodied Cognition Tests
    std::cout << "📦 Embodied Cognition Tests" << std::endl;
    TEST(embodied_cognition_basic) PASS();
    TEST(embodied_body_schema_update) PASS();
    TEST(embodied_spatial_reasoning) PASS();
    std::cout << std::endl;
    
    // Embedded Cognition Tests
    std::cout << "🌍 Embedded Cognition Tests" << std::endl;
    TEST(embedded_context_awareness) PASS();
    TEST(embedded_affordance_detection) PASS();
    TEST(embedded_context_adaptation) PASS();
    std::cout << std::endl;
    
    // Enacted Cognition Tests
    std::cout << "🎭 Enacted Cognition Tests" << std::endl;
    TEST(enacted_active_perception) PASS();
    TEST(enacted_meaning_construction) PASS();
    std::cout << std::endl;
    
    // Extended Cognition Tests
    std::cout << "🔧 Extended Cognition Tests" << std::endl;
    TEST(extended_cognitive_extension) PASS();
    TEST(extended_symbolic_scaffolds) PASS();
    TEST(extended_memory_distribution) PASS();
    std::cout << std::endl;
    
    // Salience Landscape Tests
    std::cout << "🗺️  Salience Landscape Tests" << std::endl;
    TEST(salience_landscape_update) PASS();
    TEST(salience_navigation) PASS();
    std::cout << std::endl;
    
    // Perspectival and Participatory Knowing Tests
    std::cout << "👁️  Perspectival & Participatory Knowing Tests" << std::endl;
    TEST(perspectival_knowing) PASS();
    TEST(participatory_knowing) PASS();
    TEST(perspective_shift) PASS();
    std::cout << std::endl;
    
    // Processing Mode Tests
    std::cout << "⚙️  Processing Mode Tests" << std::endl;
    TEST(processing_mode_determination) PASS();
    TEST(conformative_processing) PASS();
    TEST(transformative_processing) PASS();
    std::cout << std::endl;
    
    // Cognitive-Emotional Integration Tests
    std::cout << "💭❤️  Cognitive-Emotional Integration Tests" << std::endl;
    TEST(cognitive_emotional_integration) PASS();
    TEST(meaning_coherence_computation) PASS();
    TEST(decision_quality_enhancement) PASS();
    std::cout << std::endl;
    
    // Full 4E Framework Tests
    std::cout << "🧠 Full 4E Framework Integration Tests" << std::endl;
    TEST(4e_framework_initialization) PASS();
    TEST(4e_complete_cycle) PASS();
    TEST(4e_wisdom_computation) PASS();
    TEST(4e_meaning_connectivity) PASS();
    TEST(4e_framework_reset) PASS();
    std::cout << std::endl;
    
    // Summary
    std::cout << "=======================================================" << std::endl;
    std::cout << "📊 Test Summary" << std::endl;
    std::cout << "  Tests passed: " << tests_passed << "/" << tests_total << std::endl;
    
    if (tests_passed == tests_total) {
        std::cout << "✅ All tests passed!" << std::endl;
        std::cout << std::endl;
        std::cout << "🎉 Vervaeke 4E Cognition Framework Implementation Complete" << std::endl;
        std::cout << "   - Embodied cognition: Sensorimotor integration ✅" << std::endl;
        std::cout << "   - Embedded cognition: Context awareness ✅" << std::endl;
        std::cout << "   - Enacted cognition: Active perception ✅" << std::endl;
        std::cout << "   - Extended cognition: Tool use & extension ✅" << std::endl;
        std::cout << "   - Salience landscape: Relevance navigation ✅" << std::endl;
        std::cout << "   - Perspectival knowing: Multiple viewpoints ✅" << std::endl;
        std::cout << "   - Participatory knowing: Agent-world coupling ✅" << std::endl;
        std::cout << "   - Processing modes: Transformative/Conformative ✅" << std::endl;
        std::cout << "   - Cognitive-emotional: Integrated decision-making ✅" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed" << std::endl;
        return 1;
    }
}

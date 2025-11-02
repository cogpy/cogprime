#include "../core/meaning_making_system.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace rros::meaning;

// Test utilities
void print_test_header(const std::string& test_name) {
    std::cout << "\n========================================\n";
    std::cout << "Test: " << test_name << "\n";
    std::cout << "========================================\n";
}

void print_test_result(const std::string& test_name, bool passed) {
    std::cout << test_name << ": " << (passed ? "PASSED" : "FAILED") << "\n";
}

// Test 1: Semantic Representation
bool test_semantic_representation() {
    print_test_header("Semantic Representation");
    
    SemanticRepresentation semantic_system;
    
    // Add concepts
    ConceptNode concept1;
    concept1.concept_id = "dog";
    concept1.label = "Dog";
    concept1.embedding = {0.5f, 0.3f, 0.8f, 0.2f};
    concept1.activation = 1.0f;
    
    ConceptNode concept2;
    concept2.concept_id = "cat";
    concept2.label = "Cat";
    concept2.embedding = {0.4f, 0.35f, 0.75f, 0.25f};
    concept2.activation = 0.8f;
    
    semantic_system.add_concept(concept1);
    semantic_system.add_concept(concept2);
    
    // Test similarity
    float similarity = semantic_system.compute_similarity("dog", "cat");
    std::cout << "Dog-Cat similarity: " << similarity << "\n";
    assert(similarity > 0.5f && "Concepts should be similar");
    
    // Test activation spreading
    semantic_system.spread_activation("dog", 1.0f, 2, 0.8f);
    auto activated = semantic_system.get_activated_concepts(0.3f);
    std::cout << "Activated concepts: " << activated.size() << "\n";
    
    // Test statistics
    auto stats = semantic_system.get_statistics();
    std::cout << "Total concepts: " << stats["total_concepts"] << "\n";
    assert(stats["total_concepts"] >= 2.0f);
    
    return true;
}

// Test 2: Contextual Meaning
bool test_contextual_meaning() {
    print_test_header("Contextual Meaning");
    
    auto semantic_system = std::make_shared<SemanticRepresentation>();
    ContextualMeaning context_system(semantic_system);
    
    // Add context frames
    ContextFrame spatial_context;
    spatial_context.frame_id = "park_context";
    spatial_context.type = ContextType::SPATIAL;
    spatial_context.parameters["outdoor"] = 1.0f;
    spatial_context.parameters["daytime"] = 0.8f;
    spatial_context.activation = 1.0f;
    spatial_context.relevance = 0.9f;
    
    ContextFrame social_context;
    social_context.frame_id = "conversation_context";
    social_context.type = ContextType::SOCIAL;
    social_context.parameters["formal"] = 0.3f;
    social_context.parameters["friendly"] = 0.9f;
    social_context.activation = 0.7f;
    
    context_system.add_context_frame(spatial_context);
    context_system.add_context_frame(social_context);
    
    // Test context activation
    context_system.activate_context("park_context", 1.0f);
    auto active_contexts = context_system.get_active_contexts(0.5f);
    std::cout << "Active contexts: " << active_contexts.size() << "\n";
    
    // Test context blending
    auto blended = context_system.blend_contexts(
        {"park_context", "conversation_context"}
    );
    std::cout << "Blended activation: " << blended.activation << "\n";
    assert(blended.activation > 0.0f);
    
    return true;
}

// Test 3: Meaning Integration
bool test_meaning_integration() {
    print_test_header("Meaning Integration");
    
    auto semantic_system = std::make_shared<SemanticRepresentation>();
    auto context_system = std::make_shared<ContextualMeaning>(semantic_system);
    MeaningIntegrator integrator(semantic_system, context_system);
    
    // Build hierarchy
    std::vector<float> input_data = {0.5f, 0.3f, 0.8f, 0.2f, 0.6f};
    auto hierarchy = integrator.build_hierarchy(input_data, "test_context");
    
    std::cout << "Hierarchy ID: " << hierarchy.hierarchy_id << "\n";
    std::cout << "Levels populated: " << hierarchy.level_concepts.size() << "\n";
    
    // Test coherence
    float coherence = integrator.compute_hierarchy_coherence(hierarchy);
    std::cout << "Hierarchy coherence: " << coherence << "\n";
    
    // Enforce coherence
    integrator.enforce_coherence(hierarchy, 0.7f);
    
    // Analyze hierarchy
    auto analysis = integrator.analyze_hierarchy(hierarchy);
    std::cout << "Total concepts in hierarchy: " << analysis["total_concepts"] << "\n";
    
    return true;
}

// Test 4: Symbolic-Subsymbolic Bridge
bool test_symbolic_subsymbolic_bridge() {
    print_test_header("Symbolic-Subsymbolic Bridge");
    
    auto semantic_system = std::make_shared<SemanticRepresentation>();
    SymbolicSubsymbolicBridge bridge(semantic_system);
    
    // Ground symbols
    std::vector<float> dog_grounding = {0.8f, 0.2f, 0.6f, 0.4f};
    std::vector<float> cat_grounding = {0.7f, 0.25f, 0.55f, 0.45f};
    
    bridge.ground_symbol("dog", dog_grounding);
    bridge.ground_symbol("cat", cat_grounding);
    
    // Test vector to symbol
    std::vector<float> test_vector = {0.75f, 0.22f, 0.58f, 0.42f};
    auto symbols = bridge.vector_to_symbols(test_vector, 2, 0.5f);
    
    std::cout << "Symbols activated: " << symbols.size() << "\n";
    for (const auto& symbol : symbols) {
        std::cout << "  - " << symbol << "\n";
    }
    
    // Test symbol composition
    auto composed = bridge.compose_symbols({"dog", "cat"}, "conjunction");
    std::cout << "Composed vector size: " << composed.size() << "\n";
    assert(!composed.empty());
    
    // Test grounding strength
    float strength = bridge.get_grounding_strength("dog");
    std::cout << "Dog grounding strength: " << strength << "\n";
    assert(strength > 0.0f);
    
    return true;
}

// Test 5: Unified Meaning-Making System
bool test_unified_system() {
    print_test_header("Unified Meaning-Making System");
    
    std::unordered_map<std::string, float> config = {
        {"embedding_dim", 64.0f},
        {"min_coherence", 0.6f}
    };
    
    MeaningMakingSystem system(config);
    
    // Ground some concepts
    system.ground_concept("happy", {0.9f, 0.1f, 0.8f, 0.2f});
    system.ground_concept("sad", {0.1f, 0.9f, 0.2f, 0.8f});
    
    // Construct meaning from input
    std::vector<float> input = {0.8f, 0.3f, 0.7f, 0.4f};
    auto meanings = system.construct_meaning(input, "", {{"valence", 0.7f}});
    
    std::cout << "Constructed meanings: " << meanings.size() << "\n";
    for (const auto& meaning : meanings) {
        std::cout << "  - " << meaning << "\n";
    }
    
    // Understand input with hierarchy
    auto hierarchy = system.understand_input(input);
    std::cout << "Understanding hierarchy levels: " << hierarchy.level_concepts.size() << "\n";
    
    // Assess quality
    float quality = system.assess_meaning_quality(hierarchy);
    std::cout << "Meaning quality: " << quality << "\n";
    
    // Validate coherence
    std::vector<std::string> test_concepts = {"happy", "sad"};
    bool coherent = system.validate_meaning_coherence(test_concepts, 0.3f);
    std::cout << "Concepts coherent: " << (coherent ? "yes" : "no") << "\n";
    
    // Get system statistics
    auto stats = system.get_system_statistics();
    std::cout << "\nSystem Statistics:\n";
    for (const auto& [key, value] : stats) {
        std::cout << "  " << key << ": " << value << "\n";
    }
    
    // Analyze meaning state
    auto state = system.analyze_meaning_state();
    std::cout << "\nMeaning State Analysis:\n";
    std::cout << "  Grounding ratio: " << state["grounding_ratio"] << "\n";
    
    return true;
}

// Test 6: Integration and Performance
bool test_integration_and_performance() {
    print_test_header("Integration and Performance");
    
    MeaningMakingSystem system;
    
    // Create context
    ContextFrame context;
    context.frame_id = "test_context";
    context.type = ContextType::TASK;
    context.activation = 1.0f;
    context.relevance = 0.9f;
    system.context_system()->add_context_frame(context);
    
    // Ground multiple concepts
    for (int i = 0; i < 10; ++i) {
        std::string concept = "concept_" + std::to_string(i);
        std::vector<float> grounding(8);
        for (size_t j = 0; j < grounding.size(); ++j) {
            grounding[j] = static_cast<float>(i + j) / 20.0f;
        }
        system.ground_concept(concept, grounding);
    }
    
    // Test performance with multiple inputs
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        std::vector<float> input(8);
        for (size_t j = 0; j < input.size(); ++j) {
            input[j] = static_cast<float>(i % 10 + j) / 20.0f;
        }
        
        auto meanings = system.construct_meaning(input, "test_context");
        auto hierarchy = system.understand_input(input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "100 meaning construction cycles: " << duration.count() << " μs\n";
    std::cout << "Average per cycle: " << duration.count() / 100 << " μs\n";
    
    // Final statistics
    auto final_stats = system.get_system_statistics();
    std::cout << "\nFinal Statistics:\n";
    std::cout << "  Total concepts: " << final_stats["semantic_total_concepts"] << "\n";
    std::cout << "  Total groundings: " << final_stats["bridge_total_groundings"] << "\n";
    std::cout << "  Hierarchies built: " << final_stats["integrator_hierarchies_built"] << "\n";
    
    return true;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║   RROS Meaning-Making System Test Suite               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    
    int passed = 0;
    int total = 0;
    
    // Run all tests
    if (test_semantic_representation()) ++passed; ++total;
    if (test_contextual_meaning()) ++passed; ++total;
    if (test_meaning_integration()) ++passed; ++total;
    if (test_symbolic_subsymbolic_bridge()) ++passed; ++total;
    if (test_unified_system()) ++passed; ++total;
    if (test_integration_and_performance()) ++passed; ++total;
    
    // Summary
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Test Summary                                         ║\n";
    std::cout << "╠════════════════════════════════════════════════════════╣\n";
    std::cout << "║   Passed: " << std::setw(2) << passed << " / " << total << "                                         ║\n";
    std::cout << "║   Success Rate: " << std::fixed << std::setprecision(1) 
              << (100.0f * passed / total) << "%                                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    
    return (passed == total) ? 0 : 1;
}

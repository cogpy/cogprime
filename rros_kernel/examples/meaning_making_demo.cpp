#include "../core/meaning_making_system.hpp"
#include <iostream>
#include <iomanip>

using namespace rros::meaning;

/**
 * @brief Demonstration of the RROS Meaning-Making System
 * 
 * This example shows how to:
 * 1. Create and configure the meaning-making system
 * 2. Ground concepts from experiences
 * 3. Construct contextual meanings
 * 4. Build multi-level understanding hierarchies
 * 5. Validate and refine meanings
 */

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     RROS Meaning-Making System - Demo                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    // ========================================================================
    // 1. System Initialization
    // ========================================================================
    print_section("1. System Initialization");
    
    std::unordered_map<std::string, float> config = {
        {"embedding_dim", 8.0f},
        {"min_coherence", 0.6f},
        {"adaptation_rate", 0.1f}
    };
    
    MeaningMakingSystem system(config);
    std::cout << "✓ Meaning-making system initialized\n";
    std::cout << "  Configuration: " << config.size() << " parameters\n";
    
    // ========================================================================
    // 2. Grounding Concepts from Experience
    // ========================================================================
    print_section("2. Grounding Concepts from Experience");
    
    // Simulate sensory experiences for different concepts
    std::vector<std::pair<std::string, std::vector<float>>> experiences = {
        {"happy", {0.9f, 0.1f, 0.8f, 0.2f, 0.7f, 0.3f, 0.85f, 0.15f}},
        {"sad", {0.1f, 0.9f, 0.2f, 0.8f, 0.3f, 0.7f, 0.15f, 0.85f}},
        {"dog", {0.8f, 0.3f, 0.6f, 0.4f, 0.7f, 0.3f, 0.75f, 0.25f}},
        {"cat", {0.75f, 0.35f, 0.55f, 0.45f, 0.65f, 0.35f, 0.7f, 0.3f}},
        {"park", {0.6f, 0.4f, 0.9f, 0.1f, 0.8f, 0.2f, 0.7f, 0.3f}},
        {"home", {0.5f, 0.5f, 0.4f, 0.6f, 0.3f, 0.7f, 0.45f, 0.55f}}
    };
    
    for (const auto& [concept, experience] : experiences) {
        system.ground_concept(concept, experience);
        std::cout << "✓ Grounded concept: " << concept << "\n";
    }
    
    // ========================================================================
    // 3. Creating Contexts
    // ========================================================================
    print_section("3. Creating Contextual Frames");
    
    // Create park context
    ContextFrame park_context;
    park_context.frame_id = "park_context";
    park_context.type = ContextType::SPATIAL;
    park_context.parameters["outdoor"] = 1.0f;
    park_context.parameters["daytime"] = 0.8f;
    park_context.parameters["social"] = 0.6f;
    park_context.activation = 1.0f;
    park_context.relevance = 0.9f;
    park_context.salient_concepts = {"park", "dog", "happy"};
    
    system.context_system()->add_context_frame(park_context);
    std::cout << "✓ Created 'park' context\n";
    std::cout << "  Type: SPATIAL\n";
    std::cout << "  Salient concepts: " << park_context.salient_concepts.size() << "\n";
    
    // Create home context
    ContextFrame home_context;
    home_context.frame_id = "home_context";
    home_context.type = ContextType::SPATIAL;
    home_context.parameters["indoor"] = 1.0f;
    home_context.parameters["comfortable"] = 0.9f;
    home_context.activation = 0.7f;
    home_context.relevance = 0.8f;
    home_context.salient_concepts = {"home", "cat", "happy"};
    
    system.context_system()->add_context_frame(home_context);
    std::cout << "✓ Created 'home' context\n";
    
    // ========================================================================
    // 4. Constructing Contextual Meanings
    // ========================================================================
    print_section("4. Constructing Contextual Meanings");
    
    // Input representing a situation
    std::vector<float> situation_input = {0.7f, 0.2f, 0.8f, 0.3f, 0.6f, 0.4f, 0.75f, 0.25f};
    
    // Construct meaning in park context
    std::cout << "\nInterpreting situation in PARK context:\n";
    auto park_meanings = system.construct_meaning(
        situation_input, 
        "park_context",
        {{"social_interaction", 0.7f}, {"activity_level", 0.8f}}
    );
    
    std::cout << "  Activated meanings:\n";
    for (const auto& meaning : park_meanings) {
        std::cout << "    - " << meaning << "\n";
    }
    
    // Construct meaning in home context
    std::cout << "\nInterpreting same situation in HOME context:\n";
    auto home_meanings = system.construct_meaning(
        situation_input,
        "home_context",
        {{"comfort_level", 0.9f}, {"activity_level", 0.3f}}
    );
    
    std::cout << "  Activated meanings:\n";
    for (const auto& meaning : home_meanings) {
        std::cout << "    - " << meaning << "\n";
    }
    
    // ========================================================================
    // 5. Multi-Level Understanding
    // ========================================================================
    print_section("5. Building Multi-Level Understanding Hierarchy");
    
    auto hierarchy = system.understand_input(
        situation_input,
        {"park_context"}
    );
    
    std::cout << "✓ Built meaning hierarchy: " << hierarchy.hierarchy_id << "\n";
    std::cout << "  Levels populated: " << hierarchy.level_concepts.size() << "\n";
    
    for (const auto& [level, concepts] : hierarchy.level_concepts) {
        std::string level_name;
        switch(level) {
            case MeaningLevel::SENSORY: level_name = "SENSORY"; break;
            case MeaningLevel::CONCEPTUAL: level_name = "CONCEPTUAL"; break;
            case MeaningLevel::PROPOSITIONAL: level_name = "PROPOSITIONAL"; break;
            case MeaningLevel::NARRATIVE: level_name = "NARRATIVE"; break;
            case MeaningLevel::ABSTRACT: level_name = "ABSTRACT"; break;
            case MeaningLevel::META: level_name = "META"; break;
        }
        std::cout << "  " << level_name << ": " << concepts.size() << " concepts\n";
    }
    
    // ========================================================================
    // 6. Quality Assessment and Validation
    // ========================================================================
    print_section("6. Quality Assessment and Validation");
    
    float quality = system.assess_meaning_quality(hierarchy);
    std::cout << "✓ Meaning quality score: " << std::fixed << std::setprecision(3) 
              << quality << "\n";
    
    bool coherent = system.validate_meaning_coherence(park_meanings, 0.5f);
    std::cout << "✓ Meaning coherence: " << (coherent ? "VALID" : "NEEDS REFINEMENT") << "\n";
    
    // ========================================================================
    // 7. Adaptive Refinement
    // ========================================================================
    print_section("7. Adaptive Refinement from Feedback");
    
    // Simulate positive feedback for "happy" concept
    std::vector<float> positive_feedback = {0.95f, 0.05f, 0.9f, 0.1f, 0.85f, 0.15f, 0.9f, 0.1f};
    
    std::cout << "Refining 'happy' concept with positive feedback...\n";
    system.refine_meaning("happy", positive_feedback);
    std::cout << "✓ Concept refined\n";
    
    // Learn from successful meaning construction
    system.learn_from_feedback(park_meanings, 0.9f);
    std::cout << "✓ Learned associations between concepts\n";
    
    // ========================================================================
    // 8. System Statistics and State
    // ========================================================================
    print_section("8. System Statistics and State");
    
    auto stats = system.get_system_statistics();
    std::cout << "\nSystem Statistics:\n";
    std::cout << "  Semantic concepts: " << stats["semantic_total_concepts"] << "\n";
    std::cout << "  Semantic relations: " << stats["semantic_total_relations"] << "\n";
    std::cout << "  Symbol groundings: " << stats["bridge_total_groundings"] << "\n";
    std::cout << "  Context frames: " << stats["context_total_contexts"] << "\n";
    std::cout << "  Hierarchies built: " << stats["integrator_hierarchies_built"] << "\n";
    
    auto state = system.analyze_meaning_state();
    std::cout << "\nMeaning State Analysis:\n";
    std::cout << "  Grounding ratio: " << std::fixed << std::setprecision(3) 
              << state["grounding_ratio"] << "\n";
    std::cout << "  Context coverage: " << state["context_coverage"] << "\n";
    
    // ========================================================================
    // 9. Translation Between Symbolic and Subsymbolic
    // ========================================================================
    print_section("9. Symbolic-Subsymbolic Translation");
    
    // Translate unknown vector to symbols
    std::vector<float> unknown_state = {0.8f, 0.25f, 0.7f, 0.3f, 0.65f, 0.35f, 0.75f, 0.25f};
    auto symbols = system.translate_to_symbols(unknown_state, 3);
    
    std::cout << "Unknown state translates to symbols:\n";
    for (const auto& symbol : symbols) {
        std::cout << "  - " << symbol << "\n";
    }
    
    // Get distributed representation of a symbol
    auto dog_vector = system.bridge()->symbolize_to_vector("dog");
    std::cout << "\n'dog' symbol as vector: [";
    for (size_t i = 0; i < std::min(size_t(4), dog_vector.size()); ++i) {
        std::cout << std::fixed << std::setprecision(2) << dog_vector[i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << "...]\n";
    
    // ========================================================================
    // 10. Summary
    // ========================================================================
    print_section("10. Summary");
    
    std::cout << "\n✓ Successfully demonstrated:\n";
    std::cout << "  1. System initialization and configuration\n";
    std::cout << "  2. Concept grounding from experience\n";
    std::cout << "  3. Context creation and management\n";
    std::cout << "  4. Context-sensitive meaning construction\n";
    std::cout << "  5. Multi-level understanding hierarchies\n";
    std::cout << "  6. Quality assessment and validation\n";
    std::cout << "  7. Adaptive refinement from feedback\n";
    std::cout << "  8. System statistics and introspection\n";
    std::cout << "  9. Symbolic-subsymbolic translation\n";
    
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Demo Complete - Meaning-Making System Ready         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
    
    return 0;
}

#include "../core/cross_domain_integrator.hpp"
#include "../core/meta_learning_engine.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

using namespace rros;

void test_unified_representation() {
    std::cout << "Test 1: Unified representation creation... ";
    
    UnifiedRepresentation repr;
    
    std::vector<float> visual_data = {0.8f, 0.6f, 0.4f, 0.9f};
    auto visual_concept = repr.create_representation(visual_data, Modality::VISUAL);
    
    assert(!visual_concept.embedding.empty());
    assert(!visual_concept.abstract_features.empty());
    assert(visual_concept.confidence > 0.0f);
    assert(visual_concept.modalities.size() == 1);
    assert(visual_concept.modalities[0] == Modality::VISUAL);
    
    std::cout << "PASS" << std::endl;
}

void test_representation_merging() {
    std::cout << "Test 2: Multi-modal representation merging... ";
    
    UnifiedRepresentation repr;
    
    std::vector<float> visual_data = {0.8f, 0.6f, 0.4f};
    std::vector<float> audio_data = {0.7f, 0.5f, 0.3f};
    
    auto visual_concept = repr.create_representation(visual_data, Modality::VISUAL);
    auto audio_concept = repr.create_representation(audio_data, Modality::AUDITORY);
    
    std::vector<UnifiedConcept> concepts = {visual_concept, audio_concept};
    auto merged = repr.merge_representations(concepts);
    
    assert(!merged.embedding.empty());
    assert(merged.modalities.size() == 2);
    assert(merged.confidence > 0.0f);
    
    std::cout << "PASS (modalities: " << merged.modalities.size() << ")" << std::endl;
}

void test_concept_abstraction() {
    std::cout << "Test 3: Concept abstraction... ";
    
    UnifiedRepresentation repr;
    
    std::vector<float> data = {0.5f, 0.6f, 0.7f, 0.8f};
    auto concept = repr.create_representation(data, Modality::TEXTUAL);
    
    float initial_level = concept.abstraction_level;
    auto abstracted = repr.abstract_concept(concept, 0.9f);
    
    assert(abstracted.abstraction_level > initial_level);
    assert(abstracted.confidence <= concept.confidence);
    
    std::cout << "PASS (level: " << initial_level << " -> " << abstracted.abstraction_level << ")" << std::endl;
}

void test_similarity_computation() {
    std::cout << "Test 4: Concept similarity computation... ";
    
    UnifiedRepresentation repr;
    
    std::vector<float> data1 = {0.8f, 0.6f, 0.4f};
    std::vector<float> data2 = {0.7f, 0.5f, 0.3f};
    std::vector<float> data3 = {0.1f, 0.2f, 0.1f};
    
    auto concept1 = repr.create_representation(data1, Modality::VISUAL);
    auto concept2 = repr.create_representation(data2, Modality::VISUAL);
    auto concept3 = repr.create_representation(data3, Modality::VISUAL);
    
    float sim_similar = repr.compute_similarity(concept1, concept2);
    float sim_different = repr.compute_similarity(concept1, concept3);
    
    assert(sim_similar > sim_different);
    assert(sim_similar >= 0.0f && sim_similar <= 1.0f);
    
    std::cout << "PASS (similar: " << sim_similar << ", different: " << sim_different << ")" << std::endl;
}

void test_cross_modal_attention() {
    std::cout << "Test 5: Cross-modal attention computation... ";
    
    CrossModalAttentionSystem attention_sys;
    UnifiedRepresentation repr;
    
    auto visual = repr.create_representation({0.8f, 0.6f}, Modality::VISUAL);
    auto audio = repr.create_representation({0.7f, 0.5f}, Modality::AUDITORY);
    auto text = repr.create_representation({0.6f, 0.4f}, Modality::TEXTUAL);
    
    std::vector<UnifiedConcept> concepts = {visual, audio, text};
    auto attention = attention_sys.compute_attention(concepts);
    
    assert(attention.modality_weights.size() > 0);
    assert(attention.attention_scores.size() == concepts.size());
    assert(attention.integration_strength >= 0.0f && attention.integration_strength <= 1.0f);
    
    std::cout << "PASS (weights: " << attention.modality_weights.size() << ")" << std::endl;
}

void test_modality_integration() {
    std::cout << "Test 6: Cross-modal integration... ";
    
    CrossModalAttentionSystem attention_sys;
    UnifiedRepresentation repr;
    
    auto visual = repr.create_representation({0.8f, 0.6f, 0.4f}, Modality::VISUAL);
    auto audio = repr.create_representation({0.7f, 0.5f, 0.3f}, Modality::AUDITORY);
    
    std::vector<UnifiedConcept> concepts = {visual, audio};
    auto attention = attention_sys.compute_attention(concepts);
    auto integrated = attention_sys.integrate_modalities(concepts, attention);
    
    assert(!integrated.embedding.empty());
    assert(integrated.modalities.size() >= 2);
    assert(integrated.confidence > 0.0f);
    
    std::cout << "PASS (confidence: " << integrated.confidence << ")" << std::endl;
}

void test_cross_modal_patterns() {
    std::cout << "Test 7: Cross-modal pattern detection... ";
    
    CrossModalAttentionSystem attention_sys;
    UnifiedRepresentation repr;
    
    // Create correlated cross-modal concepts
    auto visual = repr.create_representation({0.8f, 0.6f, 0.4f}, Modality::VISUAL);
    auto audio = repr.create_representation({0.75f, 0.55f, 0.35f}, Modality::AUDITORY);
    
    std::vector<UnifiedConcept> concepts = {visual, audio};
    float pattern_strength = attention_sys.detect_cross_modal_patterns(concepts);
    
    assert(pattern_strength >= 0.0f && pattern_strength <= 1.0f);
    
    std::cout << "PASS (strength: " << pattern_strength << ")" << std::endl;
}

void test_domain_registration() {
    std::cout << "Test 8: Domain registration... ";
    
    CrossDomainIntegrator integrator;
    
    Domain vision("vision", {0.5f, 0.6f, 0.7f});
    Domain language("language", {0.4f, 0.5f, 0.6f});
    
    integrator.register_domain(vision);
    integrator.register_domain(language);
    
    auto metrics = integrator.get_metrics();
    assert(metrics["domains_registered"] == 2.0f);
    
    std::cout << "PASS" << std::endl;
}

void test_abstract_mapping() {
    std::cout << "Test 9: Abstract concept mapping... ";
    
    CrossDomainIntegrator integrator;
    
    Domain vision("vision", {0.5f, 0.6f, 0.7f});
    Domain robotics("robotics", {0.52f, 0.58f, 0.68f});
    
    integrator.register_domain(vision);
    integrator.register_domain(robotics);
    
    auto mapping = integrator.create_abstract_mapping("vision", "robotics");
    
    assert(mapping.source_domain == "vision");
    assert(mapping.target_domain == "robotics");
    assert(mapping.mapping_confidence > 0.0f);
    
    std::cout << "PASS (confidence: " << mapping.mapping_confidence << ")" << std::endl;
}

void test_cross_domain_reasoning() {
    std::cout << "Test 10: Cross-domain reasoning... ";
    
    CrossDomainIntegrator integrator;
    UnifiedRepresentation repr;
    
    auto visual = repr.create_representation({0.8f, 0.6f, 0.4f}, Modality::VISUAL);
    auto textual = repr.create_representation({0.7f, 0.5f, 0.3f}, Modality::TEXTUAL);
    
    std::vector<UnifiedConcept> concepts = {visual, textual};
    std::vector<float> query = {0.75f, 0.55f, 0.35f};
    
    auto result = integrator.cross_domain_reasoning(concepts, query);
    
    assert(!result.embedding.empty());
    assert(result.confidence > 0.0f);
    
    std::cout << "PASS" << std::endl;
}

void test_knowledge_integration() {
    std::cout << "Test 11: Cross-domain knowledge integration... ";
    
    CrossDomainIntegrator integrator;
    UnifiedRepresentation repr;
    
    auto visual1 = repr.create_representation({0.8f, 0.6f}, Modality::VISUAL);
    auto visual2 = repr.create_representation({0.7f, 0.5f}, Modality::VISUAL);
    auto text1 = repr.create_representation({0.6f, 0.4f}, Modality::TEXTUAL);
    
    std::unordered_map<std::string, std::vector<UnifiedConcept>> domain_concepts = {
        {"vision", {visual1, visual2}},
        {"language", {text1}}
    };
    
    auto integrated = integrator.integrate_cross_domain_knowledge(domain_concepts);
    
    assert(!integrated.embedding.empty());
    assert(integrated.abstraction_level > 0.0f);
    
    std::cout << "PASS (abstraction: " << integrated.abstraction_level << ")" << std::endl;
}

void test_domain_translation() {
    std::cout << "Test 12: Domain-specific to general translation... ";
    
    CrossDomainIntegrator integrator;
    
    std::vector<float> visual_knowledge = {0.8f, 0.6f, 0.4f, 0.9f};
    auto general = integrator.translate_to_general(visual_knowledge, "visual_perception");
    
    assert(!general.embedding.empty());
    assert(general.abstraction_level > 0.5f);  // Should be highly abstract
    
    std::cout << "PASS (abstraction: " << general.abstraction_level << ")" << std::endl;
}

void test_general_to_domain_translation() {
    std::cout << "Test 13: General to domain-specific translation... ";
    
    CrossDomainIntegrator integrator;
    
    UnifiedConcept general("general", {0.5f, 0.6f, 0.7f, 0.8f});
    general.abstraction_level = 0.8f;
    
    auto audio_specific = integrator.translate_to_domain(general, "audio_processing");
    
    assert(!audio_specific.empty());
    
    std::cout << "PASS (size: " << audio_specific.size() << ")" << std::endl;
}

void test_knowledge_graph() {
    std::cout << "Test 14: Knowledge graph operations... ";
    
    CrossDomainIntegrator integrator;
    UnifiedRepresentation repr;
    
    // Create knowledge nodes
    KnowledgeNode node1, node2, node3;
    node1.node_id = "concept_1";
    node1.concept = repr.create_representation({0.8f, 0.6f}, Modality::ABSTRACT);
    
    node2.node_id = "concept_2";
    node2.concept = repr.create_representation({0.7f, 0.5f}, Modality::ABSTRACT);
    
    node3.node_id = "concept_3";
    node3.concept = repr.create_representation({0.6f, 0.4f}, Modality::ABSTRACT);
    
    integrator.add_knowledge_node(node1);
    integrator.add_knowledge_node(node2);
    integrator.add_knowledge_node(node3);
    
    integrator.connect_nodes("concept_1", "concept_2", 0.9f);
    integrator.connect_nodes("concept_2", "concept_3", 0.8f);
    
    std::vector<float> query = {0.75f, 0.55f};
    auto results = integrator.query_knowledge_graph(query, 2);
    
    assert(results.size() <= 2);
    
    std::cout << "PASS (results: " << results.size() << ")" << std::endl;
}

void test_consistency_validation() {
    std::cout << "Test 15: Cross-domain consistency validation... ";
    
    CrossDomainIntegrator integrator;
    UnifiedRepresentation repr;
    
    // Create consistent concepts
    auto concept1 = repr.create_representation({0.8f, 0.6f, 0.4f}, Modality::VISUAL);
    auto concept2 = repr.create_representation({0.75f, 0.55f, 0.35f}, Modality::AUDITORY);
    
    std::vector<UnifiedConcept> concepts = {concept1, concept2};
    float consistency = integrator.validate_consistency(concepts);
    
    assert(consistency >= 0.0f && consistency <= 1.0f);
    
    std::cout << "PASS (consistency: " << consistency << ")" << std::endl;
}

void test_domain_adaptation() {
    std::cout << "Test 16: Domain adaptation... ";
    
    CrossDomainIntegrator integrator;
    
    Domain source("source", {0.5f, 0.6f, 0.7f});
    integrator.register_domain(source);
    
    std::vector<float> target_context = {0.55f, 0.65f, 0.75f};
    float adaptation_accuracy = integrator.adapt_domain("source", target_context);
    
    assert(adaptation_accuracy >= 0.0f && adaptation_accuracy <= 1.0f);
    
    std::cout << "PASS (accuracy: " << adaptation_accuracy << ")" << std::endl;
}

void test_alignment() {
    std::cout << "Test 17: Representation alignment... ";
    
    UnifiedRepresentation repr;
    
    auto c1 = repr.create_representation({0.8f, 0.2f, 0.4f}, Modality::VISUAL);
    auto c2 = repr.create_representation({0.2f, 0.8f, 0.4f}, Modality::VISUAL);
    auto c3 = repr.create_representation({0.4f, 0.4f, 0.8f}, Modality::VISUAL);
    
    std::vector<UnifiedConcept> concepts = {c1, c2, c3};
    auto aligned = repr.align_representations(concepts);
    
    assert(aligned.size() == concepts.size());
    
    std::cout << "PASS" << std::endl;
}

void test_metrics_and_statistics() {
    std::cout << "Test 18: Metrics and statistics... ";
    
    CrossDomainIntegrator integrator;
    
    auto metrics = integrator.get_metrics();
    auto stats = integrator.get_statistics();
    
    assert(metrics.size() > 0);
    assert(stats.size() >= metrics.size());
    
    std::cout << "PASS (metrics: " << metrics.size() << ", stats: " << stats.size() << ")" << std::endl;
}

void test_reset_functionality() {
    std::cout << "Test 19: Reset functionality... ";
    
    CrossDomainIntegrator integrator;
    UnifiedRepresentation repr;
    
    Domain d("test", {0.5f});
    integrator.register_domain(d);
    
    auto before = integrator.get_metrics();
    assert(before["domains_registered"] > 0.0f);
    
    integrator.reset();
    repr.reset();
    
    auto after = integrator.get_metrics();
    assert(after["domains_registered"] == 0.0f);
    
    std::cout << "PASS" << std::endl;
}

void performance_benchmark() {
    std::cout << "\n==================================\n";
    std::cout << "Performance Benchmark:\n";
    std::cout << "======================\n";
    
    CrossDomainIntegrator integrator;
    UnifiedRepresentation repr;
    
    // Benchmark unified representation
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {  // Reduced from 100
        std::vector<float> data = {0.5f + i * 0.01f, 0.6f + i * 0.01f};
        repr.create_representation(data, Modality::ABSTRACT);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Unified representation (10 iterations): " << duration.count() << " μs\n";
    std::cout << "Average per representation: " << duration.count() / 10 << " μs\n";
    
    // Benchmark cross-domain reasoning
    auto visual = repr.create_representation({0.8f, 0.6f, 0.4f}, Modality::VISUAL);
    auto audio = repr.create_representation({0.7f, 0.5f, 0.3f}, Modality::AUDITORY);
    std::vector<UnifiedConcept> concepts = {visual, audio};
    std::vector<float> query = {0.75f, 0.55f, 0.35f};
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {  // Reduced from 100
        integrator.cross_domain_reasoning(concepts, query);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Cross-domain reasoning (10 iterations): " << duration.count() << " μs\n";
    std::cout << "Average per reasoning: " << duration.count() / 10 << " μs\n";
}

int main() {
    std::cout << "==================================\n";
    std::cout << "Cross-Domain Integration Framework Test Suite\n";
    std::cout << "==================================\n\n";
    
    try {
        // Unified Representation Tests
        test_unified_representation();
        test_representation_merging();
        test_concept_abstraction();
        test_similarity_computation();
        test_alignment();
        
        // Cross-Modal Attention Tests
        test_cross_modal_attention();
        test_modality_integration();
        test_cross_modal_patterns();
        
        // Cross-Domain Integration Tests
        test_domain_registration();
        test_abstract_mapping();
        test_cross_domain_reasoning();
        test_knowledge_integration();
        test_domain_translation();
        test_general_to_domain_translation();
        
        // Knowledge Graph Tests
        test_knowledge_graph();
        
        // Validation Tests
        test_consistency_validation();
        test_domain_adaptation();
        
        // System Tests
        test_metrics_and_statistics();
        test_reset_functionality();
        
        std::cout << "\n==================================\n";
        std::cout << "All tests PASSED! ✓\n";
        std::cout << "==================================\n";
        
        performance_benchmark();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}

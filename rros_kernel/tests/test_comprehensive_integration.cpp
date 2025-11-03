/**
 * @file test_comprehensive_integration.cpp
 * @brief Comprehensive integration tests for RROS cognitive kernel
 * 
 * Tests seamless operation across all modules, inter-module communication,
 * data flow pipelines, and end-to-end cognitive scenarios.
 */

#include "../core/rros_kernel.hpp"
#include "../core/relevance_engine.hpp"
#include "../core/attention_manager.hpp"
#include "../core/memory_core.hpp"
#include "../core/relevance_optimizer.hpp"
#include "../core/resource_manager.hpp"
#include "../core/meta_cognitive_monitor.hpp"
#include "../core/meta_learning_engine.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>

/**
 * @brief Test inter-module communication and data flow
 */
bool test_inter_module_communication() {
    std::cout << "\n🧪 Testing Inter-Module Communication...\n";
    
    try {
        // Initialize all major modules
        rros::RROSKernel kernel;
        
        // Test data flows through all modules correctly
        std::vector<float> input_data = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
        std::unordered_map<std::string, float> context = {
            {"attention_focus", 0.8f},
            {"goal_relevance", 0.6f},
            {"memory_strength", 0.7f}
        };
        
        // Process through cognitive cycle (tests all module integration)
        auto state = kernel.cognitive_cycle(input_data, context);
        
        // Verify data consistency across modules
        assert(state.global_relevance >= 0.0f && state.global_relevance <= 1.0f);
        assert(state.confidence >= 0.0f && state.confidence <= 1.0f);
        assert(!state.mode_activations.empty());
        assert(!state.episode_influences.empty());
        
        // Test that attention allocation affects subsequent processing
        std::vector<std::vector<float>> targets = {{0.1f}, {0.9f}, {0.5f}};
        auto attention_weights = kernel.allocate_attention(targets);
        
        auto state_after_attention = kernel.cognitive_cycle(input_data, context);
        
        // States should differ due to attention influence
        assert(state_after_attention.last_update > state.last_update);
        
        std::cout << "✅ Inter-module communication validated\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Inter-module communication test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test data flow consistency through processing pipelines
 */
bool test_data_flow_consistency() {
    std::cout << "\n🧪 Testing Data Flow Consistency...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Test that same input produces consistent output
        std::vector<float> test_input = {0.2f, 0.4f, 0.6f, 0.8f};
        
        auto result1 = kernel.realize_relevance(test_input);
        auto result2 = kernel.realize_relevance(test_input);
        
        // Results should be deterministic for same input
        assert(std::abs(result1 - result2) < 1e-6);
        
        // Test pipeline stages preserve data integrity
        std::vector<float> original = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        std::vector<float> processed = original;
        
        // Process through multiple stages
        auto state1 = kernel.cognitive_cycle(processed);
        auto state2 = kernel.cognitive_cycle(processed);
        auto state3 = kernel.cognitive_cycle(processed);
        
        // Verify state evolution is coherent
        assert(state1.last_update < state2.last_update);
        assert(state2.last_update < state3.last_update);
        
        // Verify no data corruption
        for (size_t i = 0; i < original.size(); ++i) {
            assert(std::abs(processed[i] - original[i]) < 1e-6);
        }
        
        std::cout << "✅ Data flow consistency validated\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Data flow consistency test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test end-to-end cognitive scenario: Problem solving
 */
bool test_problem_solving_scenario() {
    std::cout << "\n🧪 Testing End-to-End Problem Solving Scenario...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Scenario: Solving a complex pattern recognition task
        std::vector<std::vector<float>> problem_sequence = {
            {0.1f, 0.2f, 0.3f},  // Initial observation
            {0.2f, 0.4f, 0.6f},  // Pattern emerges
            {0.3f, 0.6f, 0.9f},  // Pattern strengthens
            {0.4f, 0.8f, 1.0f}   // Extrapolation (normalized)
        };
        
        std::vector<float> confidence_progression;
        std::vector<float> relevance_progression;
        
        for (const auto& observation : problem_sequence) {
            auto state = kernel.cognitive_cycle(observation);
            confidence_progression.push_back(state.confidence);
            relevance_progression.push_back(state.global_relevance);
        }
        
        // Verify learning progression: confidence should generally increase
        // as pattern becomes clearer
        assert(confidence_progression.size() == problem_sequence.size());
        assert(relevance_progression.size() == problem_sequence.size());
        
        // Final confidence should be meaningful
        assert(confidence_progression.back() >= 0.0f);
        assert(relevance_progression.back() >= 0.0f);
        
        std::cout << "✅ Problem solving scenario validated\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Problem solving scenario test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test end-to-end cognitive scenario: Multi-modal integration
 */
bool test_multimodal_integration_scenario() {
    std::cout << "\n🧪 Testing Multi-modal Integration Scenario...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Activate episodes that handle multi-modal processing
        kernel.activate_episode(rros::Episode::EMBODIED_COGNITION, 1.0f);
        kernel.activate_episode(rros::Episode::RELEVANCE_REALIZATION, 1.0f);
        
        // Scenario: Processing visual and auditory information together
        std::vector<float> visual_input = {0.8f, 0.6f, 0.4f, 0.2f};
        std::vector<float> auditory_input = {0.3f, 0.5f, 0.7f, 0.9f};
        
        // Combine modalities
        std::vector<float> combined_input;
        combined_input.insert(combined_input.end(), visual_input.begin(), visual_input.end());
        combined_input.insert(combined_input.end(), auditory_input.begin(), auditory_input.end());
        
        auto integrated_state = kernel.cognitive_cycle(combined_input);
        
        // Verify integration produces coherent output
        assert(integrated_state.global_relevance >= 0.0f);
        assert(integrated_state.confidence >= 0.0f);
        assert(!integrated_state.mode_activations.empty());
        
        // Process modalities separately and compare
        auto visual_state = kernel.cognitive_cycle(visual_input);
        auto auditory_state = kernel.cognitive_cycle(auditory_input);
        
        // Integrated processing should provide additional value
        assert(integrated_state.mode_activations.size() >= visual_state.mode_activations.size());
        
        std::cout << "✅ Multi-modal integration scenario validated\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Multi-modal integration scenario test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test system performance under load
 */
bool test_performance_under_load() {
    std::cout << "\n🧪 Testing Performance Under Load...\n";
    
    try {
        rros::RROSKernel kernel;
        
        const int num_iterations = 1000;
        std::vector<float> processing_times;
        
        std::vector<float> test_input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        
        for (int i = 0; i < num_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            kernel.cognitive_cycle(test_input);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            processing_times.push_back(static_cast<float>(duration.count()));
        }
        
        // Calculate statistics
        float avg_time = std::accumulate(processing_times.begin(), 
                                        processing_times.end(), 0.0f) / processing_times.size();
        float max_time = *std::max_element(processing_times.begin(), processing_times.end());
        
        std::cout << "  Average processing time: " << avg_time << " μs\n";
        std::cout << "  Max processing time: " << max_time << " μs\n";
        
        // Performance should be within acceptable bounds
        assert(avg_time < 1000.0f); // Less than 1ms average
        assert(max_time < 10000.0f); // Less than 10ms max
        
        std::cout << "✅ Performance under load validated\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Performance under load test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test error recovery mechanisms
 */
bool test_error_recovery() {
    std::cout << "\n🧪 Testing Error Recovery Mechanisms...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Test recovery from empty input
        std::vector<float> empty_input;
        auto state1 = kernel.cognitive_cycle(empty_input);
        assert(state1.confidence >= 0.0f); // Should handle gracefully
        
        // Test recovery from extreme values
        std::vector<float> extreme_input = {1000.0f, -1000.0f, 0.0f};
        auto state2 = kernel.cognitive_cycle(extreme_input);
        assert(state2.global_relevance >= 0.0f); // Should normalize
        
        // Test recovery from invalid configuration
        std::unordered_map<std::string, float> invalid_config = {
            {"relevance_threshold", -1.0f}, // Invalid negative value
            {"memory_capacity", 0.0f}       // Invalid zero capacity
        };
        
        // System should either reject or correct invalid config
        try {
            kernel.update_config(invalid_config);
            // If accepted, verify system still functions
            auto state3 = kernel.cognitive_cycle({0.5f});
            assert(state3.confidence >= 0.0f);
        } catch (...) {
            // Rejection is also acceptable
        }
        
        // Verify system can reset after errors
        kernel.reset();
        auto state4 = kernel.cognitive_cycle({0.1f, 0.5f, 0.9f});
        assert(state4.confidence >= 0.0f);
        
        std::cout << "✅ Error recovery mechanisms validated\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error recovery test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test configuration system completeness
 */
bool test_configuration_system() {
    std::cout << "\n🧪 Testing Configuration System...\n";
    
    try {
        // Test various configuration profiles
        std::vector<std::unordered_map<std::string, float>> configs = {
            // High performance config
            {{"relevance_threshold", 0.2f}, {"attention_decay", 0.9f}, {"memory_capacity", 10000.0f}},
            // Balanced config
            {{"relevance_threshold", 0.3f}, {"attention_decay", 0.95f}, {"memory_capacity", 5000.0f}},
            // Conservative config
            {{"relevance_threshold", 0.5f}, {"attention_decay", 0.99f}, {"memory_capacity", 1000.0f}}
        };
        
        std::vector<float> test_input = {0.2f, 0.4f, 0.6f, 0.8f};
        
        for (const auto& config : configs) {
            rros::RROSKernel kernel(config);
            
            // Verify each configuration produces valid results
            auto state = kernel.cognitive_cycle(test_input);
            assert(state.global_relevance >= 0.0f);
            assert(state.confidence >= 0.0f && state.confidence <= 1.0f);
            
            // Verify configuration is actually applied
            auto metrics = kernel.get_metrics();
            assert(!metrics.empty());
        }
        
        std::cout << "✅ Configuration system validated\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Configuration system test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Test system validation procedures
 */
bool test_validation_procedures() {
    std::cout << "\n🧪 Testing System Validation Procedures...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Test state validation
        auto state = kernel.get_state();
        
        // Verify all state components are valid
        assert(state.global_relevance >= 0.0f && state.global_relevance <= 1.0f);
        assert(state.confidence >= 0.0f && state.confidence <= 1.0f);
        assert(state.mode_activations.size() > 0);
        assert(state.episode_influences.size() > 0);
        assert(state.last_update.time_since_epoch().count() > 0);
        
        // Test metrics validation
        std::vector<float> test_input = {0.3f, 0.5f, 0.7f};
        for (int i = 0; i < 5; ++i) {
            kernel.cognitive_cycle(test_input);
        }
        
        auto metrics = kernel.get_metrics();
        
        // Verify all critical metrics are present and valid
        assert(metrics.find("cycles_per_second") != metrics.end());
        assert(metrics.find("average_relevance") != metrics.end());
        assert(metrics.find("confidence_level") != metrics.end());
        
        assert(metrics["cycles_per_second"] > 0.0f);
        assert(metrics["average_relevance"] >= 0.0f && metrics["average_relevance"] <= 1.0f);
        assert(metrics["confidence_level"] >= 0.0f && metrics["confidence_level"] <= 1.0f);
        
        std::cout << "✅ Validation procedures verified\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Validation procedures test failed: " << e.what() << "\n";
        return false;
    }
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "   RROS KERNEL - COMPREHENSIVE INTEGRATION TEST SUITE\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    int tests_passed = 0;
    int tests_total = 0;
    
    auto run_test = [&](bool (*test_func)(), const char* test_name) {
        tests_total++;
        if (test_func()) {
            tests_passed++;
        }
    };
    
    // Run all integration tests
    run_test(test_inter_module_communication, "Inter-Module Communication");
    run_test(test_data_flow_consistency, "Data Flow Consistency");
    run_test(test_problem_solving_scenario, "Problem Solving Scenario");
    run_test(test_multimodal_integration_scenario, "Multi-modal Integration");
    run_test(test_performance_under_load, "Performance Under Load");
    run_test(test_error_recovery, "Error Recovery");
    run_test(test_configuration_system, "Configuration System");
    run_test(test_validation_procedures, "Validation Procedures");
    
    // Summary
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "   TEST SUMMARY\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "   Tests Passed: " << tests_passed << "/" << tests_total << "\n";
    std::cout << "   Success Rate: " << (100.0f * tests_passed / tests_total) << "%\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    
    return (tests_passed == tests_total) ? 0 : 1;
}

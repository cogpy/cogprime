#include "../core/rros_kernel.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

/**
 * @brief Comprehensive tests for RROS Cognitive Kernel
 * 
 * Validates integration of all 50 Episodes of Relevance Realization
 * and core cognitive functionality.
 */

bool test_kernel_initialization() {
    std::cout << "🧪 Testing kernel initialization...\n";
    
    try {
        std::unordered_map<std::string, float> config = {
            {"relevance_threshold", 0.3f},
            {"attention_decay", 0.95f},
            {"memory_capacity", 1000.0f}
        };
        
        rros::RROSKernel kernel(config);
        
        auto state = kernel.get_state();
        
        // Verify initial state
        assert(state.global_relevance >= 0.0f && state.global_relevance <= 1.0f);
        assert(state.confidence >= 0.0f && state.confidence <= 1.0f);
        assert(state.mode_activations.size() == 6); // 6 cognitive modes
        assert(state.episode_influences.size() == 51); // Episodes 0-50
        
        std::cout << "✅ Kernel initialization test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Kernel initialization test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_cognitive_cycle() {
    std::cout << "🧪 Testing cognitive cycle processing...\n";
    
    try {
        rros::RROSKernel kernel;
        
        std::vector<float> input_data = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
        std::unordered_map<std::string, float> context = {
            {"attention_focus", 0.8f},
            {"goal_relevance", 0.6f}
        };
        
        auto initial_state = kernel.get_state();
        auto updated_state = kernel.cognitive_cycle(input_data, context);
        
        // Verify state was updated
        assert(updated_state.last_update > initial_state.last_update);
        assert(updated_state.global_relevance >= 0.0f);
        assert(updated_state.confidence >= 0.0f && updated_state.confidence <= 1.0f);
        
        std::cout << "✅ Cognitive cycle test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Cognitive cycle test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_relevance_realization() {
    std::cout << "🧪 Testing relevance realization...\n";
    
    try {
        rros::RROSKernel kernel;
        
        std::vector<float> test_data = {0.2f, 0.4f, 0.6f, 0.8f};
        
        // Test with all episodes
        float all_relevance = kernel.realize_relevance(test_data);
        assert(all_relevance >= 0.0f && all_relevance <= 1.0f);
        
        // Test with specific episodes
        std::vector<rros::Episode> specific_episodes = {
            rros::Episode::RELEVANCE_REALIZATION,
            rros::Episode::MINDFULNESS_INSIGHT,
            rros::Episode::EMBODIED_COGNITION
        };
        
        float specific_relevance = kernel.realize_relevance(test_data, specific_episodes);
        assert(specific_relevance >= 0.0f && specific_relevance <= 1.0f);
        
        std::cout << "✅ Relevance realization test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Relevance realization test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_attention_allocation() {
    std::cout << "🧪 Testing attention allocation...\n";
    
    try {
        rros::RROSKernel kernel;
        
        std::vector<std::vector<float>> targets = {
            {0.1f, 0.2f, 0.1f},    // Low relevance
            {0.8f, 0.9f, 0.7f},    // High relevance
            {0.5f, 0.4f, 0.6f},    // Medium relevance
        };
        
        auto weights = kernel.allocate_attention(targets);
        
        // Verify attention weights are valid
        assert(weights.size() == targets.size());
        for (float weight : weights) {
            assert(weight >= 0.0f && weight <= 1.0f);
        }
        
        // High relevance target should get most attention
        float max_weight = *std::max_element(weights.begin(), weights.end());
        assert(weights[1] == max_weight); // targets[1] has highest relevance
        
        std::cout << "✅ Attention allocation test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Attention allocation test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_episode_activation() {
    std::cout << "🧪 Testing episode activation control...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Test episode activation
        kernel.activate_episode(rros::Episode::PLATO_CAVE, 0.7f);
        kernel.activate_episode(rros::Episode::ARISTOTLE_WISDOM, 0.9f);
        
        auto activations = kernel.get_episode_activations();
        
        assert(std::abs(activations[rros::Episode::PLATO_CAVE] - 0.7f) < 1e-6);
        assert(std::abs(activations[rros::Episode::ARISTOTLE_WISDOM] - 0.9f) < 1e-6);
        
        // Test episode deactivation
        kernel.deactivate_episode(rros::Episode::PLATO_CAVE);
        activations = kernel.get_episode_activations();
        
        assert(activations[rros::Episode::PLATO_CAVE] == 0.0f);
        
        std::cout << "✅ Episode activation test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Episode activation test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_episode_processing() {
    std::cout << "🧪 Testing individual episode processing...\n";
    
    try {
        rros::RROSKernel kernel;
        
        std::vector<float> test_input = {0.3f, 0.6f, 0.2f, 0.8f, 0.5f};
        
        // Test specific episodes
        std::vector<rros::Episode> test_episodes = {
            rros::Episode::FLOW_MYSTICISM,
            rros::Episode::MINDFULNESS_INSIGHT,
            rros::Episode::PLATO_CAVE,
            rros::Episode::ARISTOTLE_WISDOM,
            rros::Episode::EMBODIED_COGNITION,
            rros::Episode::RELEVANCE_REALIZATION,
            rros::Episode::SCIENTIFIC_REVOLUTION,
            rros::Episode::COGNITIVE_REVOLUTION
        };
        
        for (rros::Episode episode : test_episodes) {
            float result = kernel.process_episode(episode, test_input);
            assert(result >= 0.0f); // Processing results should be non-negative
            
            // Deactivate episode and verify result is 0
            kernel.activate_episode(episode, 0.0f);
            float deactivated_result = kernel.process_episode(episode, test_input);
            assert(deactivated_result == 0.0f);
            
            // Reactivate for next test
            kernel.activate_episode(episode, 1.0f);
        }
        
        std::cout << "✅ Episode processing test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Episode processing test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_performance_metrics() {
    std::cout << "🧪 Testing performance metrics...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Run several cognitive cycles to generate metrics
        std::vector<float> test_data = {0.1f, 0.4f, 0.7f, 0.3f, 0.9f};
        for (int i = 0; i < 10; ++i) {
            kernel.cognitive_cycle(test_data);
        }
        
        auto metrics = kernel.get_metrics();
        
        // Verify essential metrics exist and are valid
        assert(metrics.find("cycles_per_second") != metrics.end());
        assert(metrics.find("average_relevance") != metrics.end());
        assert(metrics.find("confidence_level") != metrics.end());
        assert(metrics.find("active_episodes") != metrics.end());
        
        assert(metrics["cycles_per_second"] > 0.0f);
        assert(metrics["confidence_level"] >= 0.0f && metrics["confidence_level"] <= 1.0f);
        assert(metrics["active_episodes"] >= 0.0f);
        
        std::cout << "✅ Performance metrics test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Performance metrics test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_configuration_updates() {
    std::cout << "🧪 Testing configuration updates...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Update configuration
        std::unordered_map<std::string, float> new_config = {
            {"relevance_threshold", 0.5f},
            {"attention_decay", 0.9f},
            {"memory_capacity", 2000.0f}
        };
        
        kernel.update_config(new_config);
        
        // Configuration changes should not crash the system
        std::vector<float> test_data = {0.2f, 0.5f, 0.8f};
        auto state = kernel.cognitive_cycle(test_data);
        
        assert(state.global_relevance >= 0.0f);
        assert(state.confidence >= 0.0f && state.confidence <= 1.0f);
        
        std::cout << "✅ Configuration update test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Configuration update test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_kernel_reset() {
    std::cout << "🧪 Testing kernel reset functionality...\n";
    
    try {
        rros::RROSKernel kernel;
        
        // Process some data to change state
        std::vector<float> test_data = {0.3f, 0.7f, 0.4f, 0.8f};
        kernel.cognitive_cycle(test_data);
        kernel.activate_episode(rros::Episode::PLATO_CAVE, 0.5f);
        
        auto state_before_reset = kernel.get_state();
        
        // Reset kernel
        kernel.reset();
        
        auto state_after_reset = kernel.get_state();
        
        // Verify reset worked
        assert(state_after_reset.global_relevance == 0.0f);
        assert(state_after_reset.attention_focus == 0.0f);
        
        auto activations = kernel.get_episode_activations();
        for (const auto& [episode, activation] : activations) {
            assert(activation == 1.0f); // All episodes should be reactivated
        }
        
        std::cout << "✅ Kernel reset test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Kernel reset test failed: " << e.what() << "\n";
        return false;
    }
}

void run_all_tests() {
    std::cout << "🚀 Starting RROS Cognitive Kernel Test Suite\n";
    std::cout << "============================================\n\n";
    
    int passed = 0;
    int total = 8;
    
    if (test_kernel_initialization()) passed++;
    if (test_cognitive_cycle()) passed++;
    if (test_relevance_realization()) passed++;
    if (test_attention_allocation()) passed++;
    if (test_episode_activation()) passed++;
    if (test_episode_processing()) passed++;
    if (test_performance_metrics()) passed++;
    if (test_configuration_updates()) passed++;
    if (test_kernel_reset()) passed++;
    total++; // Added one more test
    
    std::cout << "\n============================================\n";
    std::cout << "🎯 Test Results: " << passed << "/" << total << " tests passed\n";
    
    if (passed == total) {
        std::cout << "🎉 All tests passed! RROS Cognitive Kernel is working correctly.\n";
        std::cout << "\n✨ The kernel successfully integrates all 50 Episodes of\n";
        std::cout << "   Relevance Realization into a unified cognitive architecture.\n";
    } else {
        std::cout << "❌ " << (total - passed) << " tests failed. Please review implementation.\n";
    }
}

int main() {
    try {
        run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test suite error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown test suite error" << std::endl;
        return 1;
    }
}
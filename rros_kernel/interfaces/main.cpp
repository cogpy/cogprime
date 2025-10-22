#include "../core/rros_kernel.hpp"
#include <iostream>
#include <vector>
#include <chrono>

/**
 * @brief Main demonstration of RROS Cognitive Kernel
 * 
 * Demonstrates integration of 50 Episodes of Relevance Realization
 * into a unified C++ cognitive architecture.
 */

void print_banner() {
    std::cout << "================================================================\n";
    std::cout << "     RROS (Relevance Realization Operating System) v1.0\n";
    std::cout << "          Pure Cognitive Kernel - C++ Implementation\n";
    std::cout << "================================================================\n";
    std::cout << "Integrating 50 Episodes of Relevance Realization by John Vervaeke\n";
    std::cout << "into a unified cognitive architecture for AGI development.\n\n";
}

void demonstrate_episode_processing(rros::RROSKernel& kernel) {
    std::cout << "🧠 Episode Processing Demonstration\n";
    std::cout << "-----------------------------------\n";
    
    // Test different types of input patterns
    std::vector<std::vector<float>> test_patterns = {
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},           // Linear progression
        {1.0f, 0.0f, 1.0f, 0.0f, 1.0f},           // Oscillation
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},           // Constant
        {0.1f, 0.3f, 0.9f, 0.2f, 0.7f},           // Random variation
        {0.0f, 0.1f, 0.4f, 0.9f, 1.0f}            // Exponential growth
    };
    
    std::vector<std::string> pattern_names = {
        "Linear Progression", "Oscillation", "Constant", "Random Variation", "Exponential Growth"
    };
    
    for (size_t i = 0; i < test_patterns.size(); ++i) {
        std::cout << "\n📊 Testing Pattern: " << pattern_names[i] << "\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        auto state = kernel.cognitive_cycle(test_patterns[i]);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "   Processing time: " << duration.count() << " μs\n";
        std::cout << "   Global relevance: " << state.global_relevance << "\n";
        std::cout << "   Attention focus: " << state.attention_focus << "\n";
        std::cout << "   Confidence: " << state.confidence << "\n";
        
        // Show top 3 active episodes
        std::vector<std::pair<rros::Episode, float>> episode_activations;
        for (const auto& [episode, activation] : state.episode_influences) {
            episode_activations.emplace_back(episode, activation);
        }
        
        std::sort(episode_activations.begin(), episode_activations.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "   Top active episodes:\n";
        for (int j = 0; j < 3 && j < static_cast<int>(episode_activations.size()); ++j) {
            std::cout << "     Episode " << static_cast<int>(episode_activations[j].first) 
                     << ": " << episode_activations[j].second << "\n";
        }
    }
}

void demonstrate_relevance_realization(rros::RROSKernel& kernel) {
    std::cout << "\n🎯 Relevance Realization Demonstration\n";
    std::cout << "-------------------------------------\n";
    
    // Test relevance realization with different episode combinations
    std::vector<float> query_data = {0.3f, 0.7f, 0.2f, 0.8f, 0.5f};
    
    std::cout << "Query pattern: [";
    for (size_t i = 0; i < query_data.size(); ++i) {
        std::cout << query_data[i];
        if (i < query_data.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";
    
    // Test with all episodes
    float all_episodes_relevance = kernel.realize_relevance(query_data);
    std::cout << "All episodes relevance: " << all_episodes_relevance << "\n";
    
    // Test with specific episode groups
    std::vector<std::vector<rros::Episode>> episode_groups = {
        {rros::Episode::FLOW_MYSTICISM, rros::Episode::MINDFULNESS_INSIGHT},
        {rros::Episode::PLATO_CAVE, rros::Episode::ARISTOTLE_WISDOM},
        {rros::Episode::RELEVANCE_REALIZATION, rros::Episode::EMBODIED_COGNITION},
        {rros::Episode::COGNITIVE_REVOLUTION, rros::Episode::SCIENTIFIC_REVOLUTION}
    };
    
    std::vector<std::string> group_names = {
        "Mystical/Mindfulness", "Classical Philosophy", "Core RR/4E Cognition", "Modern Revolutions"
    };
    
    for (size_t i = 0; i < episode_groups.size(); ++i) {
        float group_relevance = kernel.realize_relevance(query_data, episode_groups[i]);
        std::cout << group_names[i] << " relevance: " << group_relevance << "\n";
    }
}

void demonstrate_attention_allocation(rros::RROSKernel& kernel) {
    std::cout << "\n👁️  Attention Allocation Demonstration\n";
    std::cout << "-------------------------------------\n";
    
    std::vector<std::vector<float>> attention_targets = {
        {0.2f, 0.3f, 0.1f},    // Low salience
        {0.8f, 0.9f, 0.7f},    // High salience
        {0.5f, 0.5f, 0.5f},    // Medium salience
        {0.1f, 0.9f, 0.2f},    // Mixed salience
        {0.0f, 0.0f, 0.0f}     // No salience
    };
    
    std::vector<std::string> target_names = {
        "Low Salience", "High Salience", "Medium Salience", "Mixed Salience", "No Salience"
    };
    
    auto attention_weights = kernel.allocate_attention(attention_targets);
    
    std::cout << "Attention allocation results:\n";
    for (size_t i = 0; i < attention_weights.size() && i < target_names.size(); ++i) {
        std::cout << "  " << target_names[i] << ": " << attention_weights[i] << "\n";
    }
}

void show_system_metrics(rros::RROSKernel& kernel) {
    std::cout << "\n📈 System Performance Metrics\n";
    std::cout << "----------------------------\n";
    
    auto metrics = kernel.get_metrics();
    
    for (const auto& [name, value] : metrics) {
        std::cout << "  " << name << ": " << value << "\n";
    }
}

int main() {
    try {
        print_banner();
        
        // Initialize RROS kernel with default configuration
        std::unordered_map<std::string, float> config = {
            {"relevance_threshold", 0.3f},
            {"attention_decay", 0.95f},
            {"memory_capacity", 5000.0f},
            {"episode_integration_rate", 0.15f},
            {"attention_temperature", 2.0f}
        };
        
        std::cout << "🚀 Initializing RROS Cognitive Kernel...\n";
        rros::RROSKernel kernel(config);
        
        std::cout << "✅ Kernel initialized successfully!\n";
        std::cout << "   Episodes integrated: 51 (Episode 0-50)\n";
        std::cout << "   Cognitive modes: 6\n";
        std::cout << "   Architecture: Pure C++ implementation\n\n";
        
        // Run demonstrations
        demonstrate_episode_processing(kernel);
        demonstrate_relevance_realization(kernel);
        demonstrate_attention_allocation(kernel);
        show_system_metrics(kernel);
        
        std::cout << "\n🎉 RROS Cognitive Kernel demonstration completed!\n";
        std::cout << "The kernel successfully integrates all 50 Episodes of Relevance\n";
        std::cout << "Realization into a unified cognitive architecture.\n\n";
        
        std::cout << "Next steps for development:\n";
        std::cout << "1. Integrate with existing Python CogPrime system\n";
        std::cout << "2. Add neural network backends for episode processing\n";
        std::cout << "3. Implement distributed processing capabilities\n";
        std::cout << "4. Add real-time learning and adaptation\n";
        std::cout << "5. Create domain-specific episode specializations\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown error occurred" << std::endl;
        return 1;
    }
}
#include "../core/meta_cognitive_synthesis.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

/**
 * @brief Comprehensive tests for Meta-Cognitive Synthesis Framework
 * 
 * Validates all 8 required capabilities of the framework
 */

namespace {
    const float EPSILON = 0.001f;
    
    bool float_equals(float a, float b) {
        return std::abs(a - b) < EPSILON;
    }
}

bool test_initialization() {
    std::cout << "🧪 Testing meta-cognitive synthesis initialization...\n";
    
    try {
        std::unordered_map<std::string, float> config = {
            {"learning_rate", 0.01f},
            {"max_history_size", 1000.0f}
        };
        
        rros::MetaCognitiveSynthesis synthesis(config);
        
        auto state = synthesis.get_state();
        assert(state.self_awareness >= 0.0f && state.self_awareness <= 1.0f);
        assert(state.process_understanding >= 0.0f && state.process_understanding <= 1.0f);
        assert(state.strategic_alignment >= 0.0f && state.strategic_alignment <= 1.0f);
        
        auto stats = synthesis.get_statistics();
        assert(stats["total_syntheses"] == 0.0f);
        
        std::cout << "✅ Initialization test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Initialization test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_higher_order_thinking() {
    std::cout << "🧪 Testing higher-order thinking capabilities...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        rros::CognitiveProcess process("test_process");
        process.state = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
        process.efficiency = 0.6f;
        process.relevance = 0.7f;
        process.type = "reasoning";
        
        // Test thinking about thinking at different depths
        auto insights_depth1 = synthesis.think_about_thinking(process, 1);
        assert(insights_depth1["recursive_depth"] == 1.0f);
        assert(insights_depth1.find("coherence") != insights_depth1.end());
        
        auto insights_depth2 = synthesis.think_about_thinking(process, 2);
        assert(insights_depth2["recursive_depth"] == 2.0f);
        
        auto insights_depth3 = synthesis.think_about_thinking(process, 3);
        assert(insights_depth3["recursive_depth"] == 3.0f);
        
        std::cout << "  Depth 1 coherence: " << insights_depth1["coherence"] << "\n";
        std::cout << "  Depth 2 coherence: " << insights_depth2["coherence"] << "\n";
        std::cout << "  Depth 3 coherence: " << insights_depth3["coherence"] << "\n";
        
        std::cout << "✅ Higher-order thinking test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Higher-order thinking test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_self_awareness() {
    std::cout << "🧪 Testing self-awareness and introspection...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        // Assess self-awareness
        auto state = synthesis.assess_self_awareness();
        assert(state.self_awareness >= 0.0f && state.self_awareness <= 1.0f);
        assert(state.process_understanding >= 0.0f && state.process_understanding <= 1.0f);
        assert(state.strategic_alignment >= 0.0f && state.strategic_alignment <= 1.0f);
        
        std::cout << "  Self-awareness: " << state.self_awareness << "\n";
        std::cout << "  Process understanding: " << state.process_understanding << "\n";
        std::cout << "  Strategic alignment: " << state.strategic_alignment << "\n";
        
        // Test introspection
        auto introspection = synthesis.introspect(100);
        assert(introspection.find("self_awareness") != introspection.end());
        assert(introspection.find("knowledge_coherence") != introspection.end());
        
        std::cout << "  Knowledge coherence: " << introspection["knowledge_coherence"] << "\n";
        
        std::cout << "✅ Self-awareness test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Self-awareness test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_process_reasoning() {
    std::cout << "🧪 Testing cognitive process reasoning and analysis...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        rros::CognitiveProcess process("reasoning_process");
        process.state = {0.2f, 0.4f, 0.6f, 0.8f};
        process.efficiency = 0.5f;
        process.relevance = 0.7f;
        process.properties["complexity"] = 0.6f;
        process.properties["accuracy"] = 0.8f;
        
        // Analyze process
        auto analysis = synthesis.analyze_process(process);
        assert(analysis.find("efficiency") != analysis.end());
        assert(analysis.find("relevance") != analysis.end());
        assert(analysis["efficiency"] == process.efficiency);
        assert(analysis["relevance"] == process.relevance);
        
        // Check for optimization recommendations
        assert(analysis.find("optimization_needed") != analysis.end());
        
        std::cout << "  Efficiency: " << analysis["efficiency"] << "\n";
        std::cout << "  Optimization needed: " << analysis["optimization_needed"] << "\n";
        
        std::cout << "✅ Process reasoning test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Process reasoning test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_strategy_selection() {
    std::cout << "🧪 Testing meta-cognitive strategy selection...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        std::vector<rros::MetaCognitiveStrategy> strategies;
        
        rros::MetaCognitiveStrategy strategy1("reflective");
        strategy1.effectiveness = 0.7f;
        strategy1.applicability = 0.8f;
        strategies.push_back(strategy1);
        
        rros::MetaCognitiveStrategy strategy2("analytical");
        strategy2.effectiveness = 0.6f;
        strategy2.applicability = 0.7f;
        strategies.push_back(strategy2);
        
        rros::MetaCognitiveStrategy strategy3("integrative");
        strategy3.effectiveness = 0.8f;
        strategy3.applicability = 0.6f;
        strategies.push_back(strategy3);
        
        std::unordered_map<std::string, float> context = {
            {"task_complexity", 0.7f},
            {"time_pressure", 0.4f}
        };
        
        auto [selected_strategy, confidence] = synthesis.select_strategy(context, strategies);
        assert(!selected_strategy.name.empty());
        assert(confidence >= 0.0f && confidence <= 1.0f);
        
        std::cout << "  Selected strategy: " << selected_strategy.name << "\n";
        std::cout << "  Selection confidence: " << confidence << "\n";
        
        std::cout << "✅ Strategy selection test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Strategy selection test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_strategy_optimization() {
    std::cout << "🧪 Testing meta-cognitive strategy optimization...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        rros::MetaCognitiveStrategy strategy("adaptive");
        strategy.effectiveness = 0.5f;
        strategy.parameters["learning_rate"] = 0.1f;
        
        std::unordered_map<std::string, float> performance = {
            {"performance_0", 0.6f},
            {"performance_1", 0.7f},
            {"performance_2", 0.75f}
        };
        
        auto optimized = synthesis.optimize_strategy(strategy, performance);
        assert(optimized.name == strategy.name);
        
        std::cout << "  Original effectiveness: " << strategy.effectiveness << "\n";
        std::cout << "  Optimized effectiveness: " << optimized.effectiveness << "\n";
        
        std::cout << "✅ Strategy optimization test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Strategy optimization test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_recursive_processing() {
    std::cout << "🧪 Testing recursive meta-cognitive processing...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        std::vector<float> initial_state = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        
        // Test at different recursion depths
        auto result1 = synthesis.recursive_process(initial_state, 1);
        assert(result1["max_depth_reached"] == 1.0f);
        
        auto result2 = synthesis.recursive_process(initial_state, 2);
        assert(result2["max_depth_reached"] == 2.0f);
        
        auto result3 = synthesis.recursive_process(initial_state, 3);
        assert(result3["max_depth_reached"] == 3.0f);
        
        assert(result3.find("average_confidence") != result3.end());
        assert(result3.find("average_novelty") != result3.end());
        
        std::cout << "  Max depth: " << result3["max_depth_reached"] << "\n";
        std::cout << "  Total insights: " << result3["total_insights"] << "\n";
        std::cout << "  Average confidence: " << result3["average_confidence"] << "\n";
        
        std::cout << "✅ Recursive processing test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Recursive processing test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_knowledge_representation() {
    std::cout << "🧪 Testing meta-cognitive knowledge representation...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        // Store meta-knowledge
        std::unordered_map<std::string, float> knowledge1 = {
            {"insight_quality", 0.8f},
            {"novelty", 0.6f},
            {"applicability", 0.7f}
        };
        
        synthesis.store_meta_knowledge(knowledge1, "reasoning");
        
        std::unordered_map<std::string, float> knowledge2 = {
            {"insight_quality", 0.7f},
            {"novelty", 0.9f},
            {"applicability", 0.5f}
        };
        
        synthesis.store_meta_knowledge(knowledge2, "perception");
        
        // Retrieve meta-knowledge
        std::unordered_map<std::string, float> query = {
            {"insight_quality", 0.75f}
        };
        
        auto retrieved = synthesis.retrieve_meta_knowledge(query);
        assert(!retrieved.empty());
        
        std::cout << "  Retrieved entries: " << retrieved.size() << "\n";
        
        std::cout << "✅ Knowledge representation test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Knowledge representation test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_learning_and_adaptation() {
    std::cout << "🧪 Testing meta-cognitive learning and adaptation...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        // Learn from experiences
        rros::CognitiveProcess exp1("experience_1");
        exp1.state = {0.3f, 0.5f, 0.7f};
        exp1.efficiency = 0.6f;
        exp1.type = "learning";
        
        synthesis.learn_from_experience(exp1, 0.8f);  // Good outcome
        
        rros::CognitiveProcess exp2("experience_2");
        exp2.state = {0.2f, 0.4f, 0.6f};
        exp2.efficiency = 0.5f;
        exp2.type = "learning";
        
        synthesis.learn_from_experience(exp2, 0.4f);  // Poor outcome
        
        auto stats_before = synthesis.get_statistics();
        float learning_rate_before = stats_before["learning_rate"];
        
        // Adapt capabilities based on performance
        std::unordered_map<std::string, float> performance = {
            {"performance_avg", 0.75f},
            {"performance_variance", 0.1f}
        };
        
        synthesis.adapt_capabilities(performance);
        
        auto stats_after = synthesis.get_statistics();
        float learning_rate_after = stats_after["learning_rate"];
        
        assert(stats_after["total_experiences"] == 2.0f);
        
        std::cout << "  Total experiences: " << stats_after["total_experiences"] << "\n";
        std::cout << "  Learning rate before: " << learning_rate_before << "\n";
        std::cout << "  Learning rate after: " << learning_rate_after << "\n";
        std::cout << "  Successful adaptations: " << stats_after["successful_adaptations"] << "\n";
        
        std::cout << "✅ Learning and adaptation test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Learning and adaptation test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_process_registration() {
    std::cout << "🧪 Testing cognitive process registration and tracking...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        rros::CognitiveProcess process1("memory_retrieval");
        process1.state = {0.5f, 0.6f, 0.7f};
        process1.type = "memory";
        
        rros::CognitiveProcess process2("attention_allocation");
        process2.state = {0.4f, 0.5f, 0.6f};
        process2.type = "attention";
        
        synthesis.register_process(process1);
        synthesis.register_process(process2);
        
        auto stats = synthesis.get_statistics();
        assert(stats["registered_processes"] == 2.0f);
        
        // Update process state
        std::vector<float> new_state = {0.6f, 0.7f, 0.8f};
        synthesis.update_process_state("memory_retrieval", new_state);
        
        std::cout << "  Registered processes: " << stats["registered_processes"] << "\n";
        
        std::cout << "✅ Process registration test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Process registration test failed: " << e.what() << "\n";
        return false;
    }
}

bool test_integration() {
    std::cout << "🧪 Testing integrated meta-cognitive synthesis workflow...\n";
    
    try {
        rros::MetaCognitiveSynthesis synthesis;
        
        // 1. Register a cognitive process
        rros::CognitiveProcess process("integrated_reasoning");
        process.state = {0.3f, 0.5f, 0.7f, 0.9f};
        process.efficiency = 0.65f;
        process.relevance = 0.75f;
        process.type = "reasoning";
        process.properties["complexity"] = 0.7f;
        
        synthesis.register_process(process);
        
        // 2. Think about the process
        auto insights = synthesis.think_about_thinking(process, 2);
        assert(!insights.empty());
        
        // 3. Analyze the process
        auto analysis = synthesis.analyze_process(process);
        assert(!analysis.empty());
        
        // 4. Store meta-knowledge
        synthesis.store_meta_knowledge(insights, "reasoning");
        
        // 5. Learn from experience
        synthesis.learn_from_experience(process, 0.8f);
        
        // 6. Assess self-awareness
        auto state = synthesis.assess_self_awareness();
        
        // 7. Adapt based on performance
        std::unordered_map<std::string, float> performance = {
            {"performance_score", 0.8f}
        };
        synthesis.adapt_capabilities(performance);
        
        // 8. Introspect on the entire workflow
        auto introspection = synthesis.introspect(100);
        
        auto final_stats = synthesis.get_statistics();
        
        std::cout << "  Total syntheses: " << final_stats["total_syntheses"] << "\n";
        std::cout << "  Self-awareness: " << final_stats["self_awareness"] << "\n";
        std::cout << "  Process understanding: " << final_stats["process_understanding"] << "\n";
        std::cout << "  Strategic alignment: " << final_stats["strategic_alignment"] << "\n";
        
        std::cout << "✅ Integration test passed\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Integration test failed: " << e.what() << "\n";
        return false;
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Meta-Cognitive Synthesis Framework - Comprehensive Tests    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    int passed = 0;
    int total = 11;
    
    if (test_initialization()) passed++;
    if (test_higher_order_thinking()) passed++;
    if (test_self_awareness()) passed++;
    if (test_process_reasoning()) passed++;
    if (test_strategy_selection()) passed++;
    if (test_strategy_optimization()) passed++;
    if (test_recursive_processing()) passed++;
    if (test_knowledge_representation()) passed++;
    if (test_learning_and_adaptation()) passed++;
    if (test_process_registration()) passed++;
    if (test_integration()) passed++;
    
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "Test Results: " << passed << "/" << total << " tests passed\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    return (passed == total) ? 0 : 1;
}

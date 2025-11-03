#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <chrono>
#include "../core/meta_strategic_reasoner.hpp"
#include "../core/strategy_effectiveness_evaluator.hpp"

using namespace rros;

// Test helper function
void assert_near(float a, float b, float epsilon = 0.15f, const std::string& msg = "") {
    if (std::abs(a - b) > epsilon) {
        std::cerr << "FAIL: " << msg << " - Expected: " << b << ", Got: " << a << std::endl;
        assert(false);
    }
}

// Test 1: MetaStrategicReasoner initialization
void test_initialization() {
    std::cout << "Test 1: MetaStrategicReasoner initialization... ";
    
    std::unordered_map<std::string, float> config = {
        {"exploration_rate", 0.3f},
        {"learning_rate", 0.1f},
        {"planning_horizon", 8.0f}
    };
    
    MetaStrategicReasoner reasoner(config);
    
    auto stats = reasoner.get_statistics();
    assert(stats["total_strategic_decisions"] == 0.0f);
    assert(stats["successful_adaptations"] == 0.0f);
    
    std::cout << "PASS" << std::endl;
}

// Test 2: Strategic plan generation
void test_plan_generation() {
    std::cout << "Test 2: Strategic plan generation... ";
    
    MetaStrategicReasoner reasoner;
    
    StrategicContext context;
    context.goals["accuracy"] = 0.9f;
    context.goals["speed"] = 0.7f;
    context.current_state["complexity"] = 0.5f;
    
    StrategicPlan plan = reasoner.generate_strategic_plan(context, 10.0f);
    
    assert(!plan.plan_id.empty());
    assert(!plan.strategy_sequence.empty());
    assert(plan.confidence >= 0.0f && plan.confidence <= 1.0f);
    assert(plan.priority >= 0.0f && plan.priority <= 1.0f);
    
    std::cout << "PASS (strategies: " << plan.strategy_sequence.size() << ")" << std::endl;
}

// Test 3: Plan effectiveness evaluation
void test_plan_evaluation() {
    std::cout << "Test 3: Plan effectiveness evaluation... ";
    
    MetaStrategicReasoner reasoner;
    
    StrategicContext context;
    context.goals["performance"] = 0.8f;
    
    StrategicPlan plan = reasoner.generate_strategic_plan(context, 10.0f);
    
    auto metrics = reasoner.evaluate_plan_effectiveness(plan, context);
    
    assert(metrics.effectiveness_score >= 0.0f && metrics.effectiveness_score <= 1.0f);
    assert(metrics.adaptation_rate >= 0.0f);
    assert(metrics.generalization_ability >= 0.0f);
    
    std::cout << "PASS (effectiveness: " << metrics.effectiveness_score << ")" << std::endl;
}

// Test 4: Strategic adaptation
void test_strategic_adaptation() {
    std::cout << "Test 4: Strategic adaptation... ";
    
    MetaStrategicReasoner reasoner;
    
    StrategicContext context;
    context.goals["accuracy"] = 0.9f;
    
    StrategicPlan initial_plan = reasoner.generate_strategic_plan(context, 10.0f);
    
    // Simulate poor performance
    std::unordered_map<std::string, float> poor_feedback = {
        {"accuracy", 0.4f},
        {"success_rate", 0.3f}
    };
    
    StrategicPlan adapted_plan = reasoner.adapt_strategic_approach(
        initial_plan, poor_feedback, context);
    
    assert(adapted_plan.plan_id != initial_plan.plan_id);
    assert(!adapted_plan.strategy_sequence.empty());
    
    std::cout << "PASS" << std::endl;
}

// Test 5: Strategy optimization
void test_strategy_optimization() {
    std::cout << "Test 5: Strategy optimization... ";
    
    MetaStrategicReasoner reasoner;
    
    std::vector<std::string> available_strategies = {
        "analytical", "creative", "systematic", "adaptive"
    };
    
    StrategicContext context;
    context.goals["quality"] = 0.85f;
    context.current_state["resources"] = 0.6f;
    
    auto optimized = reasoner.optimize_strategy_selection(available_strategies, context);
    
    assert(!optimized.empty());
    assert(optimized.size() <= available_strategies.size());
    
    std::cout << "PASS (selected: " << optimized.size() << ")" << std::endl;
}

// Test 6: Strategic knowledge transfer
void test_knowledge_transfer() {
    std::cout << "Test 6: Strategic knowledge transfer... ";
    
    MetaStrategicReasoner reasoner;
    
    std::vector<std::string> strategies = {"analytical", "systematic"};
    
    float effectiveness = reasoner.transfer_strategic_knowledge(
        "robotics", "vision", strategies);
    
    assert(effectiveness >= 0.0f && effectiveness <= 1.0f);
    
    std::cout << "PASS (effectiveness: " << effectiveness << ")" << std::endl;
}

// Test 7: Learning from experience
void test_experiential_learning() {
    std::cout << "Test 7: Learning from experience... ";
    
    MetaStrategicReasoner reasoner;
    
    StrategicContext context;
    context.goals["accuracy"] = 0.8f;
    
    StrategicPlan plan = reasoner.generate_strategic_plan(context, 10.0f);
    
    // Simulate successful execution
    std::unordered_map<std::string, float> good_outcomes = {
        {"accuracy", 0.85f},
        {"speed", 0.75f}
    };
    
    reasoner.learn_from_experience(plan, good_outcomes, context);
    
    // Check that strategy metrics improved
    auto stats = reasoner.get_statistics();
    assert(stats["experience_count"] > 0.0f);
    
    std::cout << "PASS" << std::endl;
}

// Test 8: Performance prediction
void test_performance_prediction() {
    std::cout << "Test 8: Performance prediction... ";
    
    MetaStrategicReasoner reasoner;
    
    // Train with some experiences
    for (int i = 0; i < 5; ++i) {
        StrategicContext context;
        context.goals["quality"] = 0.7f + i * 0.05f;
        
        StrategicPlan plan = reasoner.generate_strategic_plan(context, 10.0f);
        
        std::unordered_map<std::string, float> outcomes = {
            {"quality", 0.75f + i * 0.03f}
        };
        
        reasoner.learn_from_experience(plan, outcomes, context);
    }
    
    // Predict performance
    StrategicContext new_context;
    new_context.goals["quality"] = 0.8f;
    
    float prediction = reasoner.predict_strategy_performance("analytical", new_context);
    
    assert(prediction >= 0.0f && prediction <= 1.0f);
    
    std::cout << "PASS (prediction: " << prediction << ")" << std::endl;
}

// Test 9: Strategic insights
void test_strategic_insights() {
    std::cout << "Test 9: Strategic insights... ";
    
    MetaStrategicReasoner reasoner;
    
    // Generate some experiences
    for (int i = 0; i < 10; ++i) {
        StrategicContext context;
        context.goals["performance"] = 0.5f + i * 0.05f;
        
        StrategicPlan plan = reasoner.generate_strategic_plan(context, 10.0f);
        
        std::unordered_map<std::string, float> outcomes = {
            {"performance", 0.6f + i * 0.04f}
        };
        
        reasoner.learn_from_experience(plan, outcomes, context);
    }
    
    auto insights = reasoner.get_strategic_insights();
    
    assert(insights.find("total_experiences") != insights.end());
    assert(insights["total_experiences"] > 0.0f);
    
    std::cout << "PASS (experiences: " << insights["total_experiences"] << ")" << std::endl;
}

// Test 10: StrategyEffectivenessEvaluator initialization
void test_evaluator_init() {
    std::cout << "Test 10: StrategyEffectivenessEvaluator initialization... ";
    
    std::unordered_map<std::string, float> config = {
        {"prediction_threshold", 0.7f}
    };
    
    StrategyEffectivenessEvaluator evaluator(config);
    
    std::cout << "PASS" << std::endl;
}

// Test 11: Effectiveness evaluation
void test_effectiveness_evaluation() {
    std::cout << "Test 11: Effectiveness evaluation... ";
    
    StrategyEffectivenessEvaluator evaluator;
    
    std::vector<float> context_data = {0.5f, 0.6f, 0.7f};
    std::vector<float> performance_history = {0.8f, 0.75f, 0.82f, 0.78f, 0.81f};
    
    EvaluationCriteria criteria;
    criteria.accuracy_weight = 0.4f;
    criteria.robustness_weight = 0.3f;
    
    auto results = evaluator.evaluate_effectiveness(
        "test_strategy", context_data, performance_history, criteria);
    
    assert(results.overall_effectiveness >= 0.0f && results.overall_effectiveness <= 1.0f);
    assert(results.accuracy_score >= 0.0f && results.accuracy_score <= 1.0f);
    assert(results.confidence >= 0.0f && results.confidence <= 1.0f);
    
    std::cout << "PASS (effectiveness: " << results.overall_effectiveness << ")" << std::endl;
}

// Test 12: Strategy comparison
void test_strategy_comparison() {
    std::cout << "Test 12: Strategy comparison... ";
    
    StrategyEffectivenessEvaluator evaluator;
    
    // Train with some data
    std::vector<float> context1 = {0.5f, 0.6f};
    evaluator.update_prediction_model("strategy_a", context1, 0.8f);
    evaluator.update_prediction_model("strategy_b", context1, 0.6f);
    
    std::vector<std::string> strategies = {"strategy_a", "strategy_b"};
    std::vector<float> test_context = {0.55f, 0.65f};
    
    auto ranked = evaluator.compare_strategies(strategies, test_context);
    
    assert(ranked.size() == 2);
    assert(ranked[0].second >= ranked[1].second);  // Properly ranked
    
    std::cout << "PASS (top: " << ranked[0].first << ")" << std::endl;
}

// Test 13: Performance degradation detection
void test_degradation_detection() {
    std::cout << "Test 13: Performance degradation detection... ";
    
    StrategyEffectivenessEvaluator evaluator;
    
    // Build history
    std::vector<float> context = {0.5f};
    for (int i = 0; i < 10; ++i) {
        evaluator.update_prediction_model("test_strategy", context, 0.8f);
    }
    
    // Check for degradation with recent poor performance
    std::vector<float> recent_poor = {0.4f, 0.45f, 0.42f};
    float degradation = evaluator.detect_degradation("test_strategy", recent_poor);
    
    assert(degradation >= 0.0f && degradation <= 1.0f);
    assert(degradation > 0.2f);  // Should detect degradation
    
    std::cout << "PASS (degradation: " << degradation << ")" << std::endl;
}

// Test 14: Prediction model updates
void test_prediction_updates() {
    std::cout << "Test 14: Prediction model updates... ";
    
    StrategyEffectivenessEvaluator evaluator;
    
    std::string strategy = "learning_strategy";
    
    // Provide training data
    for (int i = 0; i < 10; ++i) {
        std::vector<float> context = {0.5f + i * 0.05f};
        float performance = 0.6f + i * 0.03f;
        evaluator.update_prediction_model(strategy, context, performance);
    }
    
    // Get metrics
    auto metrics = evaluator.get_detailed_metrics(strategy);
    
    assert(metrics.find("sample_count") != metrics.end());
    assert(metrics["sample_count"] >= 10.0f);
    
    std::cout << "PASS (samples: " << metrics["sample_count"] << ")" << std::endl;
}

// Test 15: Integration test
void test_integration() {
    std::cout << "Test 15: Integration test... ";
    
    MetaStrategicReasoner reasoner;
    StrategyEffectivenessEvaluator evaluator;
    
    // Create context
    StrategicContext context;
    context.goals["accuracy"] = 0.85f;
    context.goals["efficiency"] = 0.75f;
    context.current_state["complexity"] = 0.6f;
    
    // Generate plan
    StrategicPlan plan = reasoner.generate_strategic_plan(context, 10.0f);
    
    // Evaluate plan effectiveness
    auto plan_metrics = reasoner.evaluate_plan_effectiveness(plan, context);
    
    // Simulate execution and evaluate strategies
    for (const auto& strategy : plan.strategy_sequence) {
        std::vector<float> ctx_vec = {0.6f, 0.85f, 0.75f};
        std::vector<float> perf_hist = {0.75f, 0.78f, 0.80f, 0.82f};
        
        auto eval_results = evaluator.evaluate_effectiveness(
            strategy, ctx_vec, perf_hist);
        
        // Update reasoner with results
        std::unordered_map<std::string, float> outcomes;
        outcomes["effectiveness"] = eval_results.overall_effectiveness;
        
        reasoner.learn_from_experience(plan, outcomes, context);
    }
    
    // Verify learning occurred
    auto insights = reasoner.get_strategic_insights();
    assert(insights["total_experiences"] > 0.0f);
    
    std::cout << "PASS" << std::endl;
}

// Test 16: Performance benchmark
void test_performance_benchmark() {
    std::cout << "Test 16: Performance benchmark... ";
    
    MetaStrategicReasoner reasoner;
    
    const int iterations = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        StrategicContext context;
        context.goals["metric"] = 0.5f + (i % 10) * 0.05f;
        
        StrategicPlan plan = reasoner.generate_strategic_plan(context, 10.0f);
        
        std::unordered_map<std::string, float> outcomes = {
            {"metric", 0.6f + (i % 10) * 0.04f}
        };
        
        reasoner.learn_from_experience(plan, outcomes, context);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_time = static_cast<float>(duration.count()) / iterations;
    
    std::cout << "PASS (avg: " << avg_time << " μs)" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "Meta-Strategic Enhancement System Tests" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << std::endl;
    
    try {
        test_initialization();
        test_plan_generation();
        test_plan_evaluation();
        test_strategic_adaptation();
        test_strategy_optimization();
        test_knowledge_transfer();
        test_experiential_learning();
        test_performance_prediction();
        test_strategic_insights();
        test_evaluator_init();
        test_effectiveness_evaluation();
        test_strategy_comparison();
        test_degradation_detection();
        test_prediction_updates();
        test_integration();
        test_performance_benchmark();
        
        std::cout << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "All tests PASSED! ✓" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

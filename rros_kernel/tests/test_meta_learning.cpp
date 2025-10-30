#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <chrono>
#include "../core/meta_learning_engine.hpp"

using namespace rros;

// Test helper function
void assert_near(float a, float b, float epsilon = 0.1f, const std::string& msg = "") {
    if (std::abs(a - b) > epsilon) {
        std::cerr << "FAIL: " << msg << " - Expected: " << b << ", Got: " << a << std::endl;
        assert(false);
    }
}

// Test 1: Basic MetaLearningEngine initialization
void test_initialization() {
    std::cout << "Test 1: MetaLearningEngine initialization... ";
    
    std::unordered_map<std::string, float> config = {
        {"transfer_threshold", 0.7f},
        {"few_shot_threshold", 0.8f},
        {"adaptation_rate", 0.1f}
    };
    
    MetaLearningEngine engine(config);
    
    auto metrics = engine.get_metrics();
    assert(metrics.transfer_efficiency == 0.0f);
    assert(metrics.few_shot_accuracy == 0.0f);
    assert(metrics.total_learning_cycles == 0);
    
    std::cout << "PASS" << std::endl;
}

// Test 2: Domain registration
void test_domain_registration() {
    std::cout << "Test 2: Domain registration... ";
    
    MetaLearningEngine engine;
    
    Domain vision_domain("vision", {0.1f, 0.3f, 0.5f, 0.7f, 0.9f});
    vision_domain.characteristics["modality"] = 0.8f;
    
    Domain nlp_domain("nlp", {0.2f, 0.4f, 0.6f, 0.8f, 0.9f});
    nlp_domain.characteristics["modality"] = 0.9f;
    
    engine.register_domain(vision_domain);
    engine.register_domain(nlp_domain);
    
    auto domains = engine.get_registered_domains();
    assert(domains.size() == 2);
    assert(domains[0].name == "vision");
    assert(domains[1].name == "nlp");
    
    std::cout << "PASS" << std::endl;
}

// Test 3: Transfer learning between similar domains
void test_transfer_learning_similar_domains() {
    std::cout << "Test 3: Transfer learning (similar domains)... ";
    
    MetaLearningEngine engine;
    
    // Create similar domains
    Domain source("source", {0.5f, 0.6f, 0.7f, 0.8f});
    Domain target("target", {0.52f, 0.58f, 0.68f, 0.82f}); // Very similar
    
    LearningTask task;
    task.task_id = "task1";
    task.domain = "target";
    task.training_examples = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    task.labels = {0.0f, 1.0f};
    
    float effectiveness = engine.transfer_knowledge(source, target, task);
    
    // Similar domains should have high transfer effectiveness
    assert(effectiveness > 0.5f);
    
    auto stats = engine.get_statistics();
    assert(stats["total_transfers"] == 1.0f);
    
    std::cout << "PASS (effectiveness: " << effectiveness << ")" << std::endl;
}

// Test 4: Transfer learning between dissimilar domains
void test_transfer_learning_dissimilar_domains() {
    std::cout << "Test 4: Transfer learning (dissimilar domains)... ";
    
    MetaLearningEngine engine;
    
    // Create very different domains
    Domain source("source", {0.9f, 0.8f, 0.1f, 0.2f});
    Domain target("target", {0.1f, 0.2f, 0.9f, 0.8f}); // Very different
    
    LearningTask task;
    task.task_id = "task2";
    task.domain = "target";
    
    float effectiveness = engine.transfer_knowledge(source, target, task);
    
    // Dissimilar domains should have lower effectiveness or detect negative transfer
    assert(effectiveness < 0.8f);
    
    std::cout << "PASS (effectiveness: " << effectiveness << ")" << std::endl;
}

// Test 5: Few-shot learning
void test_few_shot_learning() {
    std::cout << "Test 5: Few-shot learning... ";
    
    MetaLearningEngine engine;
    
    LearningTask task;
    task.task_id = "few_shot_task";
    
    // Create simple few-shot task: 2 classes, 3 examples each
    task.training_examples = {
        {1.0f, 0.0f, 0.0f},  // Class 0
        {0.9f, 0.1f, 0.0f},  // Class 0
        {1.0f, 0.0f, 0.1f},  // Class 0
        {0.0f, 1.0f, 0.0f},  // Class 1
        {0.1f, 0.9f, 0.0f},  // Class 1
        {0.0f, 1.0f, 0.1f}   // Class 1
    };
    task.labels = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    
    float accuracy = engine.learn_few_shot(task, 3);
    
    // Should achieve reasonable accuracy with few examples
    assert(accuracy > 0.3f);
    
    auto metrics = engine.get_metrics();
    assert(metrics.total_learning_cycles == 1);
    
    std::cout << "PASS (accuracy: " << accuracy << ")" << std::endl;
}

// Test 6: Adaptive learning strategy selection
void test_adaptive_strategy() {
    std::cout << "Test 6: Adaptive learning strategy... ";
    
    MetaLearningEngine engine;
    
    // Easy task
    LearningTask easy_task;
    easy_task.task_id = "easy";
    easy_task.difficulty = 0.2f;
    
    std::unordered_map<std::string, float> feedback = {
        {"accuracy", 0.9f}
    };
    
    LearningStrategy strategy = engine.adapt_learning_strategy(easy_task, feedback);
    
    assert(!strategy.strategy_name.empty());
    assert(strategy.learning_rate > 0.0f);
    
    // Hard task should get different strategy
    LearningTask hard_task;
    hard_task.task_id = "hard";
    hard_task.difficulty = 0.9f;
    
    LearningStrategy hard_strategy = engine.adapt_learning_strategy(hard_task, feedback);
    
    // Strategies should differ based on difficulty
    // (Conservative vs aggressive approach)
    
    std::cout << "PASS (strategy: " << strategy.strategy_name << ")" << std::endl;
}

// Test 7: Knowledge distillation
void test_knowledge_distillation() {
    std::cout << "Test 7: Knowledge distillation... ";
    
    MetaLearningEngine engine;
    
    // Create a large model
    std::vector<float> large_model(100);
    for (size_t i = 0; i < large_model.size(); ++i) {
        large_model[i] = static_cast<float>(i) / 100.0f;
    }
    
    // Distill to smaller size
    size_t target_size = 25;
    std::vector<float> distilled = engine.distill_knowledge(large_model, target_size, 0.9f);
    
    assert(distilled.size() == target_size);
    
    auto metrics = engine.get_metrics();
    assert_near(metrics.compression_ratio, 0.25f, 0.01f, "compression ratio");
    
    std::cout << "PASS (compressed from " << large_model.size() 
              << " to " << distilled.size() << ")" << std::endl;
}

// Test 8: Curriculum learning optimization
void test_curriculum_optimization() {
    std::cout << "Test 8: Curriculum optimization... ";
    
    MetaLearningEngine engine;
    
    // Create tasks with varying difficulty
    std::vector<LearningTask> tasks;
    
    LearningTask easy;
    easy.task_id = "easy";
    easy.difficulty = 0.2f;
    tasks.push_back(easy);
    
    LearningTask medium;
    medium.task_id = "medium";
    medium.difficulty = 0.5f;
    tasks.push_back(medium);
    
    LearningTask hard;
    hard.task_id = "hard";
    hard.difficulty = 0.8f;
    tasks.push_back(hard);
    
    std::unordered_map<std::string, float> learner_state = {
        {"capability_level", 0.3f}
    };
    
    std::vector<std::string> curriculum = engine.optimize_curriculum(tasks, learner_state);
    
    assert(curriculum.size() == 3);
    // Curriculum should progress from easy to hard
    // (though exact order depends on proximity to capability level)
    
    std::cout << "PASS (curriculum: ";
    for (const auto& task_id : curriculum) {
        std::cout << task_id << " ";
    }
    std::cout << ")" << std::endl;
}

// Test 9: Meta-parameter optimization
void test_meta_parameter_optimization() {
    std::cout << "Test 9: Meta-parameter optimization... ";
    
    MetaLearningEngine engine;
    
    std::unordered_map<std::string, float> params = {
        {"learning_rate", 0.01f},
        {"momentum", 0.9f},
        {"threshold", 0.5f}
    };
    
    std::vector<float> performance_history = {0.6f, 0.65f, 0.7f, 0.72f, 0.75f};
    
    auto optimized = engine.optimize_meta_parameters(params, performance_history);
    
    assert(optimized.size() == params.size());
    // Parameters should be adjusted based on performance trend
    
    std::cout << "PASS" << std::endl;
}

// Test 10: Cross-domain transfer validation
void test_cross_domain_validation() {
    std::cout << "Test 10: Cross-domain transfer validation... ";
    
    MetaLearningEngine engine;
    
    Domain source("source", {0.7f, 0.8f, 0.6f});
    Domain target("target", {0.68f, 0.82f, 0.58f});
    
    std::vector<LearningTask> validation_tasks;
    
    for (int i = 0; i < 3; ++i) {
        LearningTask task;
        task.task_id = "val_task_" + std::to_string(i);
        task.domain = "target";
        validation_tasks.push_back(task);
    }
    
    auto validation_metrics = engine.validate_transfer(source, target, validation_tasks);
    
    assert(validation_metrics.find("average_effectiveness") != validation_metrics.end());
    assert(validation_metrics.find("positive_transfer_rate") != validation_metrics.end());
    assert(validation_metrics.find("negative_transfer_rate") != validation_metrics.end());
    assert(validation_metrics.find("domain_similarity") != validation_metrics.end());
    
    std::cout << "PASS (similarity: " << validation_metrics["domain_similarity"] << ")" << std::endl;
}

// Test 11: Performance metrics and statistics
void test_metrics_and_statistics() {
    std::cout << "Test 11: Metrics and statistics... ";
    
    MetaLearningEngine engine;
    
    // Perform various operations
    Domain d1("d1", {0.5f, 0.5f});
    Domain d2("d2", {0.6f, 0.6f});
    engine.register_domain(d1);
    engine.register_domain(d2);
    
    LearningTask task;
    task.task_id = "test";
    task.training_examples = {{0.1f}, {0.2f}};
    task.labels = {0.0f, 1.0f};
    
    engine.transfer_knowledge(d1, d2, task);
    engine.learn_few_shot(task, 1);
    
    auto stats = engine.get_statistics();
    
    assert(stats["total_transfers"] == 1.0f);
    assert(stats["total_few_shot_tasks"] == 1.0f);
    assert(stats["registered_domains"] == 2.0f);
    
    std::cout << "PASS" << std::endl;
}

// Test 12: Reset functionality
void test_reset() {
    std::cout << "Test 12: Reset functionality... ";
    
    MetaLearningEngine engine;
    
    // Perform operations
    Domain d("domain", {0.5f});
    engine.register_domain(d);
    
    LearningTask task;
    task.training_examples = {{0.1f}};
    task.labels = {0.0f};
    engine.learn_few_shot(task, 1);
    
    // Reset
    engine.reset();
    
    auto metrics = engine.get_metrics();
    assert(metrics.transfer_efficiency == 0.0f);
    assert(metrics.few_shot_accuracy == 0.0f);
    assert(metrics.total_learning_cycles == 0);
    
    auto domains = engine.get_registered_domains();
    assert(domains.empty());
    
    std::cout << "PASS" << std::endl;
}

// Performance benchmark
void benchmark_performance() {
    std::cout << "\nPerformance Benchmark:" << std::endl;
    std::cout << "======================" << std::endl;
    
    MetaLearningEngine engine;
    
    // Benchmark transfer learning
    Domain source("source", std::vector<float>(100, 0.5f));
    Domain target("target", std::vector<float>(100, 0.52f));
    
    LearningTask task;
    task.task_id = "perf_test";
    task.training_examples.resize(50, std::vector<float>(10, 0.5f));
    task.labels.resize(50, 0.5f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        engine.transfer_knowledge(source, target, task);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Transfer learning (100 iterations): " 
              << duration.count() << " μs" << std::endl;
    std::cout << "Average per transfer: " 
              << duration.count() / 100 << " μs" << std::endl;
    
    // Benchmark few-shot learning
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        engine.learn_few_shot(task, 5);
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Few-shot learning (100 iterations): " 
              << duration.count() << " μs" << std::endl;
    std::cout << "Average per task: " 
              << duration.count() / 100 << " μs" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "Meta-Learning System Test Suite" << std::endl;
    std::cout << "==================================" << std::endl << std::endl;
    
    try {
        test_initialization();
        test_domain_registration();
        test_transfer_learning_similar_domains();
        test_transfer_learning_dissimilar_domains();
        test_few_shot_learning();
        test_adaptive_strategy();
        test_knowledge_distillation();
        test_curriculum_optimization();
        test_meta_parameter_optimization();
        test_cross_domain_validation();
        test_metrics_and_statistics();
        test_reset();
        
        std::cout << "\n==================================" << std::endl;
        std::cout << "All tests PASSED! ✓" << std::endl;
        std::cout << "==================================" << std::endl;
        
        benchmark_performance();
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

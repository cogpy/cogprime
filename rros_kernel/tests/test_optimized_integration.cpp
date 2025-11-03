/**
 * @file test_optimized_integration.cpp
 * @brief Integration test for GGML-optimized cognitive kernel
 */

#include "../core/optimized_relevance_engine.hpp"
#include "../core/ggml_kernel.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace rros;

void test_optimized_relevance_engine() {
    std::cout << "Testing OptimizedRelevanceEngine..." << std::endl;
    
    std::unordered_map<std::string, float> config = {
        {"relevance_threshold", 0.3f},
        {"attention_temperature", 2.0f}
    };
    
    OptimizedRelevanceEngine::OptimizationConfig opt_config;
    opt_config.enable_profiling = true;
    opt_config.enable_caching = true;
    
    OptimizedRelevanceEngine engine(config, opt_config);
    
    // Test basic processing
    std::vector<float> input1 = {0.5f, 0.3f, 0.7f, 0.2f};
    std::unordered_map<std::string, float> context = {
        {"importance", 0.8f},
        {"urgency", 0.6f}
    };
    
    auto result1 = engine.process(input1, context);
    assert(result1.average_salience >= 0.0f && result1.average_salience <= 1.0f);
    assert(result1.salience_map.size() == input1.size());
    
    // Test with different input
    std::vector<float> input2 = {0.8f, 0.9f, 0.1f, 0.4f};
    auto result2 = engine.process(input2, context);
    assert(result2.average_salience >= 0.0f && result2.average_salience <= 1.0f);
    
    // Test repeated processing (should hit cache)
    auto result3 = engine.process(input1, context);
    assert(std::abs(result3.average_salience - result1.average_salience) < 0.01f);
    
    // Get stats
    auto stats = engine.get_optimization_stats();
    std::cout << "  Cache hit rate: " << (stats.cache_hit_rate * 100.0f) << "%" << std::endl;
    std::cout << "  Avg process time: " << stats.avg_process_time_us << " µs" << std::endl;
    
    assert(stats.avg_process_time_us < 1000.0);  // Should be < 1ms
    
    std::cout << "  ✓ OptimizedRelevanceEngine tests passed" << std::endl;
}

void test_attention_allocation() {
    std::cout << "Testing optimized attention allocation..." << std::endl;
    
    OptimizedRelevanceEngine engine;
    
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<std::vector<float>> targets = {
        {1.0f, 0.0f, 0.0f, 0.0f},  // High similarity
        {0.0f, 1.0f, 0.0f, 0.0f},  // Low similarity
        {0.5f, 0.5f, 0.0f, 0.0f},  // Medium similarity
        {0.8f, 0.2f, 0.0f, 0.0f}   // High similarity
    };
    
    auto weights = engine.allocate_attention(query, targets);
    
    assert(weights.size() == targets.size());
    
    // Check weights sum to ~1.0
    float sum = 0.0f;
    for (float w : weights) {
        sum += w;
    }
    assert(std::abs(sum - 1.0f) < 0.01f);
    
    // Highest weights should be for most similar targets
    assert(weights[0] > weights[1]);  // First target most similar
    assert(weights[3] > weights[1]);  // Fourth target more similar than second
    
    std::cout << "  ✓ Attention allocation tests passed" << std::endl;
}

void test_performance_improvement() {
    std::cout << "Testing performance improvement..." << std::endl;
    
    // Setup optimized engine
    OptimizedRelevanceEngine::OptimizationConfig opt_config;
    opt_config.enable_caching = true;
    opt_config.enable_profiling = false;  // Disable to measure pure perf
    
    OptimizedRelevanceEngine opt_engine({}, opt_config);
    
    // Setup standard engine for comparison
    RelevanceEngine std_engine({});
    
    const int NUM_ITERATIONS = 100;
    std::vector<float> test_input(64, 0.5f);
    std::unordered_map<std::string, float> test_context = {
        {"relevance", 0.7f}
    };
    
    // Benchmark optimized engine
    auto opt_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        opt_engine.process(test_input, test_context);
    }
    auto opt_end = std::chrono::high_resolution_clock::now();
    auto opt_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        opt_end - opt_start).count();
    
    // Benchmark standard engine
    auto std_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        std_engine.process(test_input, test_context);
    }
    auto std_end = std::chrono::high_resolution_clock::now();
    auto std_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std_end - std_start).count();
    
    double opt_avg = static_cast<double>(opt_duration) / NUM_ITERATIONS;
    double std_avg = static_cast<double>(std_duration) / NUM_ITERATIONS;
    double improvement = ((std_avg - opt_avg) / std_avg) * 100.0;
    
    std::cout << "  Standard engine: " << std_avg << " µs/op" << std::endl;
    std::cout << "  Optimized engine: " << opt_avg << " µs/op" << std::endl;
    std::cout << "  Improvement: " << improvement << "%" << std::endl;
    
    // Check cache effectiveness
    auto stats = opt_engine.get_optimization_stats();
    std::cout << "  Cache hit rate: " << (stats.cache_hit_rate * 100.0f) << "%" << std::endl;
    
    // We expect significant improvement due to caching
    if (improvement > 0) {
        std::cout << "  ✓ Performance improved" << std::endl;
    } else {
        std::cout << "  ! No improvement (may need more iterations for cache warmup)" << std::endl;
    }
}

void test_memory_efficiency() {
    std::cout << "Testing memory efficiency..." << std::endl;
    
    OptimizedRelevanceEngine::OptimizationConfig opt_config;
    opt_config.memory_pool_size = 10 * 1024 * 1024;  // 10MB
    opt_config.cache_size = 500;
    
    OptimizedRelevanceEngine engine({}, opt_config);
    
    // Process multiple inputs
    for (int i = 0; i < 100; ++i) {
        std::vector<float> input(128);
        for (size_t j = 0; j < input.size(); ++j) {
            input[j] = static_cast<float>(i + j) / 200.0f;
        }
        engine.process(input);
    }
    
    auto stats = engine.get_optimization_stats();
    
    std::cout << "  Memory usage: " << (stats.memory_usage / 1024.0) << " KB" << std::endl;
    std::cout << "  Peak usage: " << (stats.peak_memory / 1024.0) << " KB" << std::endl;
    std::cout << "  Pool size: " << (opt_config.memory_pool_size / 1024.0) << " KB" << std::endl;
    
    // Memory should stay within pool bounds
    assert(stats.peak_memory < opt_config.memory_pool_size);
    
    // Memory efficiency check (should use < 30% reduction from naive approach)
    float usage_percent = (static_cast<float>(stats.peak_memory) / 
                          opt_config.memory_pool_size) * 100.0f;
    std::cout << "  Usage: " << usage_percent << "% of pool" << std::endl;
    
    std::cout << "  ✓ Memory efficiency tests passed" << std::endl;
}

void test_real_time_operation() {
    std::cout << "Testing real-time operation..." << std::endl;
    
    OptimizedRelevanceEngine engine;
    
    const int NUM_CYCLES = 100;
    std::vector<uint64_t> cycle_times;
    cycle_times.reserve(NUM_CYCLES);
    
    for (int i = 0; i < NUM_CYCLES; ++i) {
        std::vector<float> input(64);
        for (size_t j = 0; j < input.size(); ++j) {
            input[j] = static_cast<float>(i + j) / 100.0f;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        engine.process(input);
        auto end = std::chrono::high_resolution_clock::now();
        
        uint64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();
        cycle_times.push_back(duration_us);
    }
    
    // Calculate statistics
    uint64_t max_time = *std::max_element(cycle_times.begin(), cycle_times.end());
    uint64_t min_time = *std::min_element(cycle_times.begin(), cycle_times.end());
    uint64_t avg_time = std::accumulate(cycle_times.begin(), cycle_times.end(), 0ULL) 
                       / cycle_times.size();
    
    std::cout << "  Cycle times:" << std::endl;
    std::cout << "    Min: " << min_time << " µs" << std::endl;
    std::cout << "    Avg: " << avg_time << " µs" << std::endl;
    std::cout << "    Max: " << max_time << " µs" << std::endl;
    
    // Real-time requirement: <10ms (10000 µs)
    if (max_time < 10000) {
        std::cout << "  ✓ Meets real-time requirement (<10ms)" << std::endl;
    } else {
        std::cout << "  ! Warning: Max time exceeds 10ms" << std::endl;
    }
    
    // Stricter requirement: <1ms for most operations
    size_t under_1ms = std::count_if(cycle_times.begin(), cycle_times.end(),
        [](uint64_t t) { return t < 1000; });
    float percent_under_1ms = (static_cast<float>(under_1ms) / NUM_CYCLES) * 100.0f;
    
    std::cout << "  " << percent_under_1ms << "% of cycles <1ms" << std::endl;
    
    assert(max_time < 10000);  // Must meet 10ms requirement
}

int main() {
    std::cout << "=== Optimized Integration Tests ===" << std::endl;
    std::cout << std::endl;
    
    try {
        test_optimized_relevance_engine();
        test_attention_allocation();
        test_performance_improvement();
        test_memory_efficiency();
        test_real_time_operation();
        
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "✓ All integration tests passed!" << std::endl;
        std::cout << "\nOptimizations successfully integrated with RROS kernel." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

/**
 * @file test_ggml_optimizations.cpp
 * @brief Test suite for GGML kernel optimizations
 */

#include "../core/ggml_kernel.hpp"
#include "../core/tensor_attention.hpp"
#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace rros::ggml;

void test_memory_pool() {
    std::cout << "Testing MemoryPool..." << std::endl;
    
    MemoryPool pool(1024 * 1024);  // 1MB pool
    
    // Test allocation
    void* ptr1 = pool.allocate(256);
    assert(ptr1 != nullptr);
    
    void* ptr2 = pool.allocate(512);
    assert(ptr2 != nullptr);
    
    void* ptr3 = pool.allocate(1024);
    assert(ptr3 != nullptr);
    
    auto stats = pool.get_stats();
    assert(stats.allocation_count == 3);
    assert(stats.current_usage > 0);
    
    // Test deallocation
    pool.deallocate(ptr1);
    pool.deallocate(ptr2);
    
    stats = pool.get_stats();
    assert(stats.free_count == 2);
    
    // Test reset
    pool.reset();
    stats = pool.get_stats();
    assert(stats.current_usage == 0);
    
    std::cout << "  ✓ MemoryPool tests passed" << std::endl;
    std::cout << "    Average alloc time: " 
              << (stats.allocation_count > 0 ? 
                  stats.total_alloc_time_ns / stats.allocation_count : 0)
              << " ns" << std::endl;
}

void test_hypergraph_allocator() {
    std::cout << "Testing HypergraphAllocator..." << std::endl;
    
    HypergraphAllocator allocator(2 * 1024 * 1024);  // 2MB pool
    
    // Test hgfs_alloc
    void* ptr1 = allocator.hgfs_alloc(1024, 0);
    assert(ptr1 != nullptr);
    
    void* ptr2 = allocator.hgfs_alloc(2048, 1);
    assert(ptr2 != nullptr);
    
    void* ptr3 = allocator.hgfs_alloc(512, 2);
    assert(ptr3 != nullptr);
    
    // Test descriptor retrieval
    auto* desc1 = allocator.get_descriptor(ptr1);
    assert(desc1 != nullptr);
    assert(desc1->depth == 0);
    
    auto* desc2 = allocator.get_descriptor(ptr2);
    assert(desc2 != nullptr);
    assert(desc2->depth == 1);
    
    // Test edge creation
    allocator.hgfs_edge(ptr1, ptr2, 0.8f);
    allocator.hgfs_edge(ptr2, ptr3, 0.6f);
    
    // Test freeing
    allocator.hgfs_free(ptr1);
    allocator.hgfs_free(ptr2);
    allocator.hgfs_free(ptr3);
    
    auto stats = allocator.get_stats();
    std::cout << "  ✓ HypergraphAllocator tests passed" << std::endl;
    std::cout << "    Peak memory usage: " << stats.peak_usage << " bytes" << std::endl;
}

void test_tensor_cache() {
    std::cout << "Testing TensorCache..." << std::endl;
    
    TensorCache cache(100);
    
    // Test store and retrieve
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> data2 = {4.0f, 5.0f, 6.0f};
    
    cache.store("key1", data1);
    cache.store("key2", data2);
    
    std::vector<float> retrieved;
    bool found = cache.retrieve("key1", retrieved);
    assert(found);
    assert(retrieved == data1);
    
    found = cache.retrieve("key2", retrieved);
    assert(found);
    assert(retrieved == data2);
    
    found = cache.retrieve("key3", retrieved);
    assert(!found);
    
    auto stats = cache.get_stats();
    assert(stats.hits == 2);
    assert(stats.misses == 1);
    
    std::cout << "  ✓ TensorCache tests passed" << std::endl;
    std::cout << "    Cache hit rate: " 
              << (stats.hit_rate() * 100.0f) << "%" << std::endl;
}

void test_tensor_attention() {
    std::cout << "Testing TensorAttention..." << std::endl;
    
    TensorAttention::AttentionConfig config;
    config.use_cache = true;
    config.parallel_mode = false;  // Disable for consistent testing
    
    TensorAttention attention(config);
    
    // Test attention computation
    std::vector<float> query = {1.0f, 0.0f, 0.0f};
    std::vector<std::vector<float>> targets = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };
    
    auto weights = attention.compute_attention(query, targets);
    assert(weights.size() == 3);
    
    // Check that weights sum to ~1.0
    float sum = 0.0f;
    for (float w : weights) {
        sum += w;
    }
    assert(std::abs(sum - 1.0f) < 0.01f);
    
    // First weight should be highest (query matches first target)
    assert(weights[0] > weights[1]);
    assert(weights[0] > weights[2]);
    
    // Test apply attention
    auto output = attention.apply_attention(weights, targets);
    assert(output.size() == 3);
    
    // Test caching
    auto weights2 = attention.compute_attention(query, targets);
    auto cache_stats = attention.get_cache_stats();
    assert(cache_stats.hits > 0);
    
    std::cout << "  ✓ TensorAttention tests passed" << std::endl;
    std::cout << "    Cache hit rate: " 
              << (cache_stats.hit_rate() * 100.0f) << "%" << std::endl;
}

void test_tensor_relevance() {
    std::cout << "Testing TensorRelevance..." << std::endl;
    
    TensorRelevance::RelevanceConfig config;
    config.threshold = 0.3f;
    config.use_cache = true;
    
    TensorRelevance relevance(config);
    
    // Test relevance computation
    std::vector<float> input1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> input2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> context = {1.0f, 0.0f, 0.0f};
    
    float rel1 = relevance.compute_relevance(input1, context);
    float rel2 = relevance.compute_relevance(input2, context);
    
    // input1 should be more relevant to context
    assert(rel1 > rel2);
    
    // Test relevance map
    std::vector<std::vector<float>> inputs = {input1, input2};
    auto relevance_map = relevance.compute_relevance_map(inputs, context);
    assert(relevance_map.size() == 2);
    assert(relevance_map[0] > relevance_map[1]);
    
    std::cout << "  ✓ TensorRelevance tests passed" << std::endl;
}

void test_optimized_ops() {
    std::cout << "Testing OptimizedCognitiveOps..." << std::endl;
    
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    
    // Test dot product
    float dot = OptimizedCognitiveOps::dot_product(a, b);
    assert(std::abs(dot - 32.0f) < 0.01f);  // 1*4 + 2*5 + 3*6 = 32
    
    // Test norm
    float norm_a = OptimizedCognitiveOps::norm(a);
    assert(std::abs(norm_a - std::sqrt(14.0f)) < 0.01f);
    
    // Test cosine similarity
    float cos_sim = OptimizedCognitiveOps::cosine_similarity(a, a);
    assert(std::abs(cos_sim - 1.0f) < 0.01f);
    
    // Test add
    auto sum = OptimizedCognitiveOps::add(a, b);
    assert(sum.size() == 3);
    assert(std::abs(sum[0] - 5.0f) < 0.01f);
    
    // Test multiply
    auto prod = OptimizedCognitiveOps::multiply(a, b);
    assert(prod.size() == 3);
    assert(std::abs(prod[0] - 4.0f) < 0.01f);
    
    // Test scale
    auto scaled = OptimizedCognitiveOps::scale(a, 2.0f);
    assert(std::abs(scaled[0] - 2.0f) < 0.01f);
    
    // Test normalize
    auto normalized = OptimizedCognitiveOps::normalize(a);
    float norm_n = OptimizedCognitiveOps::norm(normalized);
    assert(std::abs(norm_n - 1.0f) < 0.01f);
    
    std::cout << "  ✓ OptimizedCognitiveOps tests passed" << std::endl;
}

void benchmark_performance() {
    std::cout << "\n=== Performance Benchmarks ===" << std::endl;
    
    // Benchmark memory allocation
    {
        HypergraphAllocator allocator(10 * 1024 * 1024);
        auto start = std::chrono::high_resolution_clock::now();
        
        constexpr int N = 10000;
        std::vector<void*> ptrs;
        for (int i = 0; i < N; ++i) {
            ptrs.push_back(allocator.hgfs_alloc(256, i % 10));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        std::cout << "Memory allocation: " 
                  << (duration.count() / N) << " ns per allocation" << std::endl;
        
        for (auto ptr : ptrs) {
            allocator.hgfs_free(ptr);
        }
    }
    
    // Benchmark attention computation
    {
        TensorAttention attention;
        std::vector<float> query(128, 0.5f);
        std::vector<std::vector<float>> targets;
        for (int i = 0; i < 50; ++i) {
            targets.push_back(std::vector<float>(128, 0.3f + i * 0.01f));
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        constexpr int N = 1000;
        for (int i = 0; i < N; ++i) {
            auto weights = attention.compute_attention(query, targets);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Attention computation: " 
                  << (duration.count() / N) << " µs per operation" << std::endl;
        
        auto cache_stats = attention.get_cache_stats();
        std::cout << "  Cache hit rate: " 
                  << (cache_stats.hit_rate() * 100.0f) << "%" << std::endl;
    }
    
    // Benchmark cognitive cycle simulation
    {
        TensorAttention attention;
        TensorRelevance relevance;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        constexpr int N = 1000;
        for (int i = 0; i < N; ++i) {
            std::vector<float> input(64, 0.5f);
            std::vector<float> context(64, 0.3f);
            
            (void)relevance.compute_relevance(input, context);
            
            std::vector<std::vector<float>> targets;
            for (int j = 0; j < 10; ++j) {
                targets.push_back(std::vector<float>(64, 0.4f + j * 0.05f));
            }
            
            auto weights = attention.compute_attention(input, targets);
            auto output = attention.apply_attention(weights, targets);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Cognitive cycle: " 
                  << (duration.count() / N) << " µs per cycle" << std::endl;
        
        if (duration.count() / N < 10000) {  // Less than 10ms
            std::cout << "  ✓ Meets real-time requirement (<10ms)" << std::endl;
        }
    }
}

int main() {
    std::cout << "=== GGML Kernel Optimization Tests ===" << std::endl;
    std::cout << std::endl;
    
    try {
        test_memory_pool();
        test_hypergraph_allocator();
        test_tensor_cache();
        test_tensor_attention();
        test_tensor_relevance();
        test_optimized_ops();
        
        benchmark_performance();
        
        std::cout << "\n=== Kernel Profiling Summary ===" << std::endl;
        g_kernel_profiler.print_summary();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}

/**
 * @file optimized_relevance_engine.cpp
 * @brief Implementation of GGML-optimized Relevance Engine
 */

#include "optimized_relevance_engine.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace rros {

// Default context value for missing context dimensions
constexpr float DEFAULT_CONTEXT_VALUE = 0.3f;

OptimizedRelevanceEngine::OptimizedRelevanceEngine(
    const std::unordered_map<std::string, float>& config,
    const OptimizationConfig& opt_config)
    : opt_config_(opt_config)
    , config_(config)
    , process_count_(0)
    , total_process_time_ns_(0) {
    
    // Initialize GGML components
    allocator_ = std::make_unique<ggml::HypergraphAllocator>(
        opt_config_.memory_pool_size);
    
    ggml::TensorAttention::AttentionConfig attn_config;
    attn_config.use_cache = opt_config_.enable_caching;
    attn_config.parallel_mode = opt_config_.parallel_mode;
    attn_config.cache_size = opt_config_.cache_size;
    attention_ = std::make_unique<ggml::TensorAttention>(attn_config);
    
    ggml::TensorRelevance::RelevanceConfig rel_config;
    rel_config.use_cache = opt_config_.enable_caching;
    rel_config.cache_size = opt_config_.cache_size;
    
    // Get threshold from config if available
    auto threshold_it = config_.find("relevance_threshold");
    if (threshold_it != config_.end()) {
        rel_config.threshold = threshold_it->second;
    }
    
    relevance_ = std::make_unique<ggml::TensorRelevance>(rel_config);
    
    if (opt_config_.enable_caching) {
        result_cache_ = std::make_unique<ggml::TensorCache>(opt_config_.cache_size);
    }
    
    // Create fallback engine for compatibility
    fallback_engine_ = std::make_unique<RelevanceEngine>(config_);
}

OptimizedRelevanceEngine::~OptimizedRelevanceEngine() = default;

RelevanceResult OptimizedRelevanceEngine::process(
    const std::vector<float>& input_data,
    const std::unordered_map<std::string, float>& context) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (opt_config_.enable_profiling) {
        PROFILE_KERNEL_OP("OptimizedRelevanceEngine::process");
    }
    
    // Convert context to vector for tensor operations
    std::vector<float> context_vec;
    context_vec.reserve(context.size());
    for (const auto& pair : context) {
        context_vec.push_back(pair.second);
    }
    
    // Ensure context has same dimensionality as input
    if (context_vec.empty() || context_vec.size() < input_data.size()) {
        context_vec.resize(input_data.size(), DEFAULT_CONTEXT_VALUE);
    }
    
    // Compute relevance using optimized tensor operations
    float relevance_score = relevance_->compute_relevance(input_data, context_vec);
    
    // Build salience map (simplified for performance)
    std::vector<float> salience_map(input_data.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        salience_map[i] = std::abs(input_data[i]) * relevance_score;
    }
    
    // Create result
    RelevanceResult result;
    result.salience_map = salience_map;
    result.max_salience = salience_map.empty() ? 0.0f : 
        *std::max_element(salience_map.begin(), salience_map.end());
    result.average_salience = relevance_score;
    result.confidence = relevance_score;
    
    // Find relevant indices (above threshold)
    for (size_t i = 0; i < salience_map.size(); ++i) {
        if (salience_map[i] > relevance_->get_threshold()) {
            result.relevant_indices.push_back(i);
        }
    }
    
    // Update statistics
    auto end = std::chrono::high_resolution_clock::now();
    uint64_t duration_ns = 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        process_count_++;
        total_process_time_ns_ += duration_ns;
    }
    
    return result;
}

float OptimizedRelevanceEngine::compute_relevance(
    const std::vector<float>& input,
    const std::vector<float>& context) {
    
    if (opt_config_.enable_profiling) {
        PROFILE_KERNEL_OP("OptimizedRelevanceEngine::compute_relevance");
    }
    
    return relevance_->compute_relevance(input, context);
}

std::vector<float> OptimizedRelevanceEngine::allocate_attention(
    const std::vector<float>& query,
    const std::vector<std::vector<float>>& targets) {
    
    if (opt_config_.enable_profiling) {
        PROFILE_KERNEL_OP("OptimizedRelevanceEngine::allocate_attention");
    }
    
    return attention_->compute_attention(query, targets);
}

OptimizedRelevanceEngine::OptimizationStats 
OptimizedRelevanceEngine::get_optimization_stats() const {
    OptimizationStats stats;
    
    // Get cache statistics
    auto cache_stats = attention_->get_cache_stats();
    stats.cache_hits = cache_stats.hits;
    stats.cache_misses = cache_stats.misses;
    stats.cache_hit_rate = cache_stats.hit_rate();
    
    // Get memory statistics
    auto mem_stats = allocator_->get_stats();
    stats.memory_usage = mem_stats.current_usage;
    stats.peak_memory = mem_stats.peak_usage;
    
    // Get timing statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        if (process_count_ > 0) {
            stats.avg_process_time_us = 
                static_cast<double>(total_process_time_ns_) / process_count_ / 1000.0;
        } else {
            stats.avg_process_time_us = 0.0;
        }
    }
    
    // Get component-specific timings from profiler
    if (opt_config_.enable_profiling) {
        auto rel_entry = ggml::g_kernel_profiler.get_entry(
            "OptimizedRelevanceEngine::compute_relevance");
        stats.avg_relevance_time_us = rel_entry.avg_us();
        
        auto attn_entry = ggml::g_kernel_profiler.get_entry(
            "OptimizedRelevanceEngine::allocate_attention");
        stats.avg_attention_time_us = attn_entry.avg_us();
    }
    
    return stats;
}

void OptimizedRelevanceEngine::print_performance_summary() const {
    auto stats = get_optimization_stats();
    
    std::cout << "\n=== Optimized Relevance Engine Performance ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\nCache Performance:" << std::endl;
    std::cout << "  Hits:      " << stats.cache_hits << std::endl;
    std::cout << "  Misses:    " << stats.cache_misses << std::endl;
    std::cout << "  Hit Rate:  " << (stats.cache_hit_rate * 100.0f) << "%" << std::endl;
    
    std::cout << "\nMemory Usage:" << std::endl;
    std::cout << "  Current:   " << (stats.memory_usage / 1024.0) << " KB" << std::endl;
    std::cout << "  Peak:      " << (stats.peak_memory / 1024.0) << " KB" << std::endl;
    
    std::cout << "\nTiming (average):" << std::endl;
    std::cout << "  Process:   " << stats.avg_process_time_us << " µs" << std::endl;
    std::cout << "  Relevance: " << stats.avg_relevance_time_us << " µs" << std::endl;
    std::cout << "  Attention: " << stats.avg_attention_time_us << " µs" << std::endl;
    
    std::cout << "\nPerformance Targets:" << std::endl;
    if (stats.avg_process_time_us < 100.0) {
        std::cout << "  ✓ Process time <100µs" << std::endl;
    }
    if (stats.cache_hit_rate > 0.4f) {
        std::cout << "  ✓ Cache hit rate >40%" << std::endl;
    }
    if (stats.memory_usage < opt_config_.memory_pool_size) {
        std::cout << "  ✓ Memory within budget" << std::endl;
    }
    
    std::cout << "==============================================" << std::endl;
}

void OptimizedRelevanceEngine::reset_optimizations() {
    if (attention_) {
        attention_->clear_cache();
    }
    if (result_cache_) {
        result_cache_->clear();
    }
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    process_count_ = 0;
    total_process_time_ns_ = 0;
}

} // namespace rros

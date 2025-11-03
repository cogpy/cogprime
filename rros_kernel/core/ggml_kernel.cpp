/**
 * @file ggml_kernel.cpp
 * @brief GGML Tensor Backend Implementation
 * 
 * High-performance implementation of cognitive kernel primitives
 * using optimized memory pooling, caching, and parallel execution.
 */

#include "ggml_kernel.hpp"
#include <algorithm>
#include <cstring>
#include <thread>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>

namespace rros {
namespace ggml {

// Global profiler instance
PerformanceProfiler g_kernel_profiler;

//=============================================================================
// MemoryPool Implementation
//=============================================================================

MemoryPool::MemoryPool(size_t pool_size)
    : pool_size_(pool_size), current_offset_(0) {
    pool_memory_.resize(pool_size);
}

MemoryPool::~MemoryPool() {
    reset();
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    PROFILE_KERNEL_OP("MemoryPool::allocate");
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Align size to requested alignment
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    
    // Align current offset
    size_t aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);
    
    // Check if we have enough space
    if (aligned_offset + aligned_size > pool_size_) {
        return nullptr;  // Pool exhausted
    }
    
    void* ptr = pool_memory_.data() + aligned_offset;
    current_offset_ = aligned_offset + aligned_size;
    
    // Track allocation
    allocations_[ptr] = aligned_size;
    
    // Update stats
    stats_.total_allocated += aligned_size;
    stats_.current_usage += aligned_size;
    stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
    stats_.allocation_count++;
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.total_alloc_time_ns += 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    return ptr;
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    PROFILE_KERNEL_OP("MemoryPool::deallocate");
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        size_t size = it->second;
        stats_.total_freed += size;
        stats_.current_usage -= size;
        stats_.free_count++;
        allocations_.erase(it);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.total_free_time_ns += 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_offset_ = 0;
    allocations_.clear();
    stats_.current_usage = 0;
}

//=============================================================================
// HypergraphAllocator Implementation
//=============================================================================

HypergraphAllocator::HypergraphAllocator(size_t pool_size) {
    pool_ = std::make_unique<MemoryPool>(pool_size);
}

HypergraphAllocator::~HypergraphAllocator() {
    // Cleanup all descriptors
    std::lock_guard<std::mutex> lock(mutex_);
    descriptors_.clear();
    edges_.clear();
}

void* HypergraphAllocator::hgfs_alloc(size_t size, uint32_t depth) {
    PROFILE_KERNEL_OP("hgfs_alloc");
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Ensure minimum allocation size
    size_t alloc_size = std::max(size, HGFS_MIN_ALLOC);
    
    // Allocate from pool
    void* ptr = pool_->allocate(alloc_size, HGFS_ALIGNMENT);
    
    if (ptr) {
        // Create tensor descriptor
        TensorDescriptor desc;
        desc.shape = {alloc_size / sizeof(float)};  // Default to 1D float tensor
        desc.total_elements = alloc_size / sizeof(float);
        desc.element_size = sizeof(float);
        desc.depth = depth;
        desc.data = ptr;
        
        descriptors_[ptr] = desc;
        
        // Initialize edges container
        edges_[ptr] = {};
    }
    
    return ptr;
}

void HypergraphAllocator::hgfs_free(void* ptr) {
    if (!ptr) return;
    
    PROFILE_KERNEL_OP("hgfs_free");
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Remove descriptor and edges
    descriptors_.erase(ptr);
    edges_.erase(ptr);
    
    // Free from pool
    pool_->deallocate(ptr);
}

void HypergraphAllocator::hgfs_edge(void* source, void* target, float weight) {
    PROFILE_KERNEL_OP("hgfs_edge");
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (edges_.find(source) != edges_.end()) {
        edges_[source].push_back({target, weight});
    }
}

TensorDescriptor* HypergraphAllocator::get_descriptor(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = descriptors_.find(ptr);
    return it != descriptors_.end() ? &it->second : nullptr;
}

MemoryStats HypergraphAllocator::get_stats() const {
    return pool_->get_stats();
}

//=============================================================================
// TensorCache Implementation
//=============================================================================

TensorCache::TensorCache(size_t max_entries)
    : max_entries_(max_entries) {
}

bool TensorCache::store(const std::string& key, const std::vector<float>& data) {
    PROFILE_KERNEL_OP("TensorCache::store");
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if we need to evict
    if (cache_.size() >= max_entries_ && cache_.find(key) == cache_.end()) {
        evict_lru();
    }
    
    // Store in cache
    cache_[key] = data;
    
    // Update access order
    auto it = std::find(access_order_.begin(), access_order_.end(), key);
    if (it != access_order_.end()) {
        access_order_.erase(it);
    }
    access_order_.push_back(key);
    
    stats_.entries = cache_.size();
    
    return true;
}

bool TensorCache::retrieve(const std::string& key, std::vector<float>& data) {
    PROFILE_KERNEL_OP("TensorCache::retrieve");
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        data = it->second;
        
        // Update access order (move to end = most recent)
        auto order_it = std::find(access_order_.begin(), access_order_.end(), key);
        if (order_it != access_order_.end()) {
            access_order_.erase(order_it);
        }
        access_order_.push_back(key);
        
        stats_.hits++;
        return true;
    }
    
    stats_.misses++;
    return false;
}

void TensorCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    access_order_.clear();
    stats_.entries = 0;
}

void TensorCache::evict_lru() {
    // Evict least recently used entry
    if (!access_order_.empty()) {
        std::string lru_key = access_order_.front();
        access_order_.erase(access_order_.begin());
        cache_.erase(lru_key);
        stats_.entries = cache_.size();
    }
}

//=============================================================================
// ParallelExecutor Implementation
//=============================================================================

ParallelExecutor::ParallelExecutor(size_t num_threads)
    : num_threads_(num_threads) {
    if (num_threads_ == 0) {
        num_threads_ = std::thread::hardware_concurrency();
        if (num_threads_ == 0) {
            num_threads_ = 4;  // Fallback
        }
    }
}

ParallelExecutor::~ParallelExecutor() = default;

// Template instantiations in header due to template nature

//=============================================================================
// PerformanceProfiler Implementation
//=============================================================================

void PerformanceProfiler::record(const std::string& name, uint64_t duration_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ProfileEntry& entry = entries_[name];
    entry.name = name;
    entry.count++;
    entry.total_ns += duration_ns;
    entry.min_ns = std::min(entry.min_ns, duration_ns);
    entry.max_ns = std::max(entry.max_ns, duration_ns);
}

PerformanceProfiler::ProfileEntry 
PerformanceProfiler::get_entry(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = entries_.find(name);
    return it != entries_.end() ? it->second : ProfileEntry{name};
}

std::vector<PerformanceProfiler::ProfileEntry> 
PerformanceProfiler::get_all_entries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<ProfileEntry> result;
    result.reserve(entries_.size());
    
    for (const auto& pair : entries_) {
        result.push_back(pair.second);
    }
    
    // Sort by total time descending
    std::sort(result.begin(), result.end(), 
        [](const ProfileEntry& a, const ProfileEntry& b) {
            return a.total_ns > b.total_ns;
        });
    
    return result;
}

void PerformanceProfiler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
}

void PerformanceProfiler::print_summary() const {
    auto entries = get_all_entries();
    
    std::cout << "\n=== Kernel Performance Profile ===" << std::endl;
    std::cout << std::setw(40) << std::left << "Operation"
              << std::setw(12) << std::right << "Count"
              << std::setw(15) << "Avg (µs)"
              << std::setw(15) << "Min (µs)"
              << std::setw(15) << "Max (µs)"
              << std::setw(15) << "Total (ms)" << std::endl;
    std::cout << std::string(112, '-') << std::endl;
    
    for (const auto& entry : entries) {
        std::cout << std::setw(40) << std::left << entry.name
                  << std::setw(12) << std::right << entry.count
                  << std::setw(15) << std::fixed << std::setprecision(3) << entry.avg_us()
                  << std::setw(15) << std::fixed << std::setprecision(3) 
                      << entry.min_ns / 1000.0
                  << std::setw(15) << std::fixed << std::setprecision(3) 
                      << entry.max_ns / 1000.0
                  << std::setw(15) << std::fixed << std::setprecision(3) 
                      << entry.total_ns / 1000000.0
                  << std::endl;
    }
    std::cout << "===================================" << std::endl;
}

//=============================================================================
// ScopedTimer Implementation
//=============================================================================

ScopedTimer::ScopedTimer(PerformanceProfiler& profiler, const std::string& name)
    : profiler_(profiler), name_(name) {
    start_ = std::chrono::high_resolution_clock::now();
}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    uint64_t duration_ns = 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
    profiler_.record(name_, duration_ns);
}

} // namespace ggml
} // namespace rros

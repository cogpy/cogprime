/**
 * @file ggml_kernel.hpp
 * @brief GGML Tensor Backend for OpenCog Cognitive Kernel
 * 
 * Provides high-performance tensor operations for cognitive processing
 * using GGML (Georgi Gerganov Machine Learning) as the computational substrate.
 * 
 * This implements the Echo.Kern primitives for AGI-OS foundation:
 * - Hypergraph Filesystem Allocator (hgfs_alloc)
 * - Tensor-based Attention Mechanisms
 * - Memory Pooling and Efficient Allocation
 * - Parallel Processing Support
 */

#ifndef RROS_GGML_KERNEL_HPP
#define RROS_GGML_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <mutex>

namespace rros {
namespace ggml {

// Forward declarations
struct TensorContext;
struct TensorNode;
struct MemoryPool;

/**
 * Memory allocation granularity for hypergraph filesystem
 */
constexpr size_t HGFS_ALIGNMENT = 64;  // Cache line alignment
constexpr size_t HGFS_MIN_ALLOC = 256;  // Minimum allocation size
constexpr size_t HGFS_POOL_SIZE = 64 * 1024 * 1024;  // 64MB default pool

/**
 * Performance targets from kernel specifications
 */
constexpr uint64_t SCHED_TICK_TARGET_NS = 5000;      // ≤5µs scheduler tick
constexpr uint64_t MEM_OP_TARGET_NS = 100;           // ≤100ns memory ops
constexpr uint64_t COGNITIVE_CYCLE_TARGET_NS = 10000000;  // ≤10ms critical cycles

/**
 * @struct MemoryStats
 * @brief Memory usage statistics for profiling
 */
struct MemoryStats {
    size_t total_allocated = 0;
    size_t total_freed = 0;
    size_t current_usage = 0;
    size_t peak_usage = 0;
    size_t allocation_count = 0;
    size_t free_count = 0;
    uint64_t total_alloc_time_ns = 0;
    uint64_t total_free_time_ns = 0;
};

/**
 * @struct TensorDescriptor
 * @brief Lightweight tensor metadata
 */
struct TensorDescriptor {
    std::vector<size_t> shape;
    size_t total_elements = 0;
    size_t element_size = sizeof(float);
    uint32_t depth = 0;  // Membrane depth (OEIS A000081)
    void* data = nullptr;
    
    size_t total_bytes() const {
        return total_elements * element_size;
    }
};

/**
 * @class MemoryPool
 * @brief High-performance memory pool with O(1) allocation
 */
class MemoryPool {
public:
    explicit MemoryPool(size_t pool_size = HGFS_POOL_SIZE);
    ~MemoryPool();
    
    /**
     * @brief Allocate aligned memory from pool
     * @param size Size in bytes
     * @param alignment Alignment requirement (default: cache line)
     * @return Pointer to allocated memory or nullptr
     */
    void* allocate(size_t size, size_t alignment = HGFS_ALIGNMENT);
    
    /**
     * @brief Deallocate memory back to pool
     * @param ptr Pointer to deallocate
     */
    void deallocate(void* ptr);
    
    /**
     * @brief Reset pool to initial state
     */
    void reset();
    
    /**
     * @brief Get memory statistics
     */
    MemoryStats get_stats() const { return stats_; }
    
private:
    std::vector<uint8_t> pool_memory_;
    size_t pool_size_;
    size_t current_offset_;
    std::unordered_map<void*, size_t> allocations_;
    MemoryStats stats_;
    mutable std::mutex mutex_;
};

/**
 * @class HypergraphAllocator
 * @brief Hypergraph Filesystem Allocator (hgfs_alloc)
 * 
 * Implements memory allocation as hypergraph tensor nodes with
 * membrane depth tracking for cognitive hierarchy.
 */
class HypergraphAllocator {
public:
    explicit HypergraphAllocator(size_t pool_size = HGFS_POOL_SIZE);
    ~HypergraphAllocator();
    
    /**
     * @brief hgfs_alloc - Allocate memory as a tensor node
     * @param size Size in bytes
     * @param depth Membrane depth (OEIS A000081)
     * @return Pointer to allocated tensor data
     */
    void* hgfs_alloc(size_t size, uint32_t depth = 0);
    
    /**
     * @brief hgfs_free - Free hypergraph node
     * @param ptr Pointer to free
     */
    void hgfs_free(void* ptr);
    
    /**
     * @brief Register hypergraph edge between nodes
     * @param source Source node pointer
     * @param target Target node pointer
     * @param weight Edge weight
     */
    void hgfs_edge(void* source, void* target, float weight = 1.0f);
    
    /**
     * @brief Get tensor descriptor for allocated memory
     */
    TensorDescriptor* get_descriptor(void* ptr);
    
    /**
     * @brief Get memory statistics
     */
    MemoryStats get_stats() const;
    
private:
    std::unique_ptr<MemoryPool> pool_;
    std::unordered_map<void*, TensorDescriptor> descriptors_;
    std::unordered_map<void*, std::vector<std::pair<void*, float>>> edges_;
    mutable std::mutex mutex_;
};

/**
 * @class TensorCache
 * @brief LRU cache for frequently accessed tensor patterns
 */
class TensorCache {
public:
    explicit TensorCache(size_t max_entries = 1000);
    
    /**
     * @brief Store tensor in cache
     * @param key Cache key
     * @param data Tensor data
     * @return true if stored successfully
     */
    bool store(const std::string& key, const std::vector<float>& data);
    
    /**
     * @brief Retrieve tensor from cache
     * @param key Cache key
     * @param data Output tensor data
     * @return true if found in cache
     */
    bool retrieve(const std::string& key, std::vector<float>& data);
    
    /**
     * @brief Clear cache
     */
    void clear();
    
    /**
     * @brief Get cache statistics
     */
    struct CacheStats {
        size_t hits = 0;
        size_t misses = 0;
        size_t entries = 0;
        float hit_rate() const {
            size_t total = hits + misses;
            return total > 0 ? static_cast<float>(hits) / total : 0.0f;
        }
    };
    
    CacheStats get_stats() const { return stats_; }
    
private:
    size_t max_entries_;
    std::unordered_map<std::string, std::vector<float>> cache_;
    std::vector<std::string> access_order_;
    CacheStats stats_;
    mutable std::mutex mutex_;
    
    void evict_lru();
};

/**
 * @class ParallelExecutor
 * @brief Parallel processing coordinator for multi-core execution
 */
class ParallelExecutor {
public:
    explicit ParallelExecutor(size_t num_threads = 0);  // 0 = auto-detect
    ~ParallelExecutor();
    
    /**
     * @brief Execute function in parallel across data
     * @param data Input data vector
     * @param func Function to execute per element
     * @return Results vector
     */
    template<typename T, typename Func>
    std::vector<T> parallel_map(const std::vector<T>& data, Func func);
    
    /**
     * @brief Parallel reduce operation
     * @param data Input data
     * @param func Reduction function
     * @param init Initial value
     * @return Reduced result
     */
    template<typename T, typename Func>
    T parallel_reduce(const std::vector<T>& data, Func func, T init);
    
    /**
     * @brief Get number of available threads
     */
    size_t num_threads() const { return num_threads_; }
    
private:
    size_t num_threads_;
};

/**
 * @class PerformanceProfiler
 * @brief Lightweight performance profiling for cognitive operations
 */
class PerformanceProfiler {
public:
    struct ProfileEntry {
        std::string name;
        uint64_t count = 0;
        uint64_t total_ns = 0;
        uint64_t min_ns = UINT64_MAX;
        uint64_t max_ns = 0;
        
        double avg_ns() const {
            return count > 0 ? static_cast<double>(total_ns) / count : 0.0;
        }
        
        double avg_us() const { return avg_ns() / 1000.0; }
        double avg_ms() const { return avg_ns() / 1000000.0; }
    };
    
    /**
     * @brief Record timing for named operation
     */
    void record(const std::string& name, uint64_t duration_ns);
    
    /**
     * @brief Get profile entry
     */
    ProfileEntry get_entry(const std::string& name) const;
    
    /**
     * @brief Get all profile entries
     */
    std::vector<ProfileEntry> get_all_entries() const;
    
    /**
     * @brief Clear all profile data
     */
    void clear();
    
    /**
     * @brief Print profile summary
     */
    void print_summary() const;
    
private:
    std::unordered_map<std::string, ProfileEntry> entries_;
    mutable std::mutex mutex_;
};

/**
 * @class ScopedTimer
 * @brief RAII timer for automatic profiling
 */
class ScopedTimer {
public:
    ScopedTimer(PerformanceProfiler& profiler, const std::string& name);
    ~ScopedTimer();
    
private:
    PerformanceProfiler& profiler_;
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

/**
 * Global profiler instance for kernel operations
 */
extern PerformanceProfiler g_kernel_profiler;

/**
 * Profiling macro for automatic timing
 */
#define PROFILE_KERNEL_OP(name) \
    rros::ggml::ScopedTimer _timer_##__LINE__(rros::ggml::g_kernel_profiler, name)

} // namespace ggml
} // namespace rros

#endif // RROS_GGML_KERNEL_HPP

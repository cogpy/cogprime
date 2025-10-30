#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <chrono>
#include <functional>

namespace rros {

// Forward declarations
class TransferLearningManager;
class FewShotLearner;
class AdaptiveLearningStrategy;
class KnowledgeDistiller;
class CurriculumOptimizer;
class MetaParameterOptimizer;

/**
 * @brief Domain representation for transfer learning
 */
struct Domain {
    std::string name;
    std::vector<float> feature_space;
    std::unordered_map<std::string, float> characteristics;
    float similarity_threshold;
    
    Domain();
    Domain(const std::string& n, const std::vector<float>& features);
};

/**
 * @brief Learning task representation
 */
struct LearningTask {
    std::string task_id;
    std::string domain;
    std::vector<std::vector<float>> training_examples;
    std::vector<float> labels;
    std::unordered_map<std::string, float> task_characteristics;
    float difficulty;
    
    LearningTask();
};

/**
 * @brief Meta-learning performance metrics
 */
struct MetaLearningMetrics {
    float transfer_efficiency;         // How well knowledge transfers
    float few_shot_accuracy;           // Accuracy with minimal examples
    float adaptation_speed;            // How fast system adapts
    float compression_ratio;           // Knowledge distillation ratio
    float curriculum_optimality;       // How optimal the learning sequence is
    uint64_t total_learning_cycles;
    std::chrono::high_resolution_clock::time_point last_update;
    
    MetaLearningMetrics();
};

/**
 * @brief Learning strategy representation
 */
struct LearningStrategy {
    std::string strategy_name;
    std::string strategy_type;
    float learning_rate;
    float momentum;
    std::unordered_map<std::string, float> hyperparameters;
    float performance_score;
    
    LearningStrategy();
};

/**
 * @brief Meta-Learning Engine for sophisticated learning capabilities
 * 
 * Implements meta-learning that enables the system to learn how to learn
 * more effectively. Includes transfer learning, few-shot learning, and
 * adaptive learning strategies that improve with experience across domains.
 * 
 * Integrates with GGML tensor operations for high-performance learning.
 */
class MetaLearningEngine {
public:
    /**
     * @brief Constructor initializes meta-learning subsystems
     * @param config Configuration parameters
     */
    explicit MetaLearningEngine(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~MetaLearningEngine();
    
    // Core meta-learning operations
    
    /**
     * @brief Transfer knowledge from source domain to target domain
     * @param source_domain Source domain with existing knowledge
     * @param target_domain Target domain for knowledge transfer
     * @param task Target task in new domain
     * @return Transfer effectiveness score [0.0, 1.0]
     */
    float transfer_knowledge(
        const Domain& source_domain,
        const Domain& target_domain,
        const LearningTask& task
    );
    
    /**
     * @brief Learn from few examples (few-shot learning)
     * @param task Task with minimal training examples
     * @param num_shots Number of examples per class
     * @return Learning success rate [0.0, 1.0]
     */
    float learn_few_shot(
        const LearningTask& task,
        uint32_t num_shots
    );
    
    /**
     * @brief Adapt learning strategy based on task characteristics
     * @param task Current learning task
     * @param performance_feedback Recent performance metrics
     * @return Adapted learning strategy
     */
    LearningStrategy adapt_learning_strategy(
        const LearningTask& task,
        const std::unordered_map<std::string, float>& performance_feedback
    );
    
    /**
     * @brief Distill knowledge from complex model to simpler one
     * @param source_model Source model parameters
     * @param target_size Target model size
     * @param preserve_performance Minimum performance to preserve [0.0, 1.0]
     * @return Distilled model parameters
     */
    std::vector<float> distill_knowledge(
        const std::vector<float>& source_model,
        size_t target_size,
        float preserve_performance = 0.9f
    );
    
    /**
     * @brief Optimize curriculum learning sequence
     * @param tasks Available learning tasks
     * @param learner_state Current learner capabilities
     * @return Optimized task sequence
     */
    std::vector<std::string> optimize_curriculum(
        const std::vector<LearningTask>& tasks,
        const std::unordered_map<std::string, float>& learner_state
    );
    
    /**
     * @brief Optimize meta-parameters for learning
     * @param current_params Current meta-parameters
     * @param performance_history Historical performance data
     * @return Optimized meta-parameters
     */
    std::unordered_map<std::string, float> optimize_meta_parameters(
        const std::unordered_map<std::string, float>& current_params,
        const std::vector<float>& performance_history
    );
    
    /**
     * @brief Validate cross-domain knowledge transfer
     * @param source_domain Source domain
     * @param target_domain Target domain
     * @param validation_tasks Tasks for validation
     * @return Validation metrics
     */
    std::unordered_map<std::string, float> validate_transfer(
        const Domain& source_domain,
        const Domain& target_domain,
        const std::vector<LearningTask>& validation_tasks
    );
    
    // State management
    
    /**
     * @brief Get current meta-learning metrics
     * @return Current performance metrics
     */
    const MetaLearningMetrics& get_metrics() const { return metrics_; }
    
    /**
     * @brief Update configuration
     * @param config New configuration parameters
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset meta-learning state
     */
    void reset();
    
    /**
     * @brief Get system statistics
     * @return Detailed statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Register a new domain for learning
     * @param domain Domain to register
     */
    void register_domain(const Domain& domain);
    
    /**
     * @brief Get registered domains
     * @return List of registered domains
     */
    std::vector<Domain> get_registered_domains() const;

private:
    // Subsystems
    std::unique_ptr<TransferLearningManager> transfer_manager_;
    std::unique_ptr<FewShotLearner> few_shot_learner_;
    std::unique_ptr<AdaptiveLearningStrategy> adaptive_strategy_;
    std::unique_ptr<KnowledgeDistiller> knowledge_distiller_;
    std::unique_ptr<CurriculumOptimizer> curriculum_optimizer_;
    std::unique_ptr<MetaParameterOptimizer> meta_param_optimizer_;
    
    // State
    MetaLearningMetrics metrics_;
    std::vector<Domain> registered_domains_;
    std::vector<LearningStrategy> strategy_history_;
    
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Thread safety
    mutable std::mutex engine_mutex_;
    
    // Statistics
    uint64_t total_transfers_;
    uint64_t successful_transfers_;
    uint64_t total_few_shot_tasks_;
    uint64_t successful_few_shot_tasks_;
    
    // Internal methods
    void initialize_subsystems();
    float compute_domain_similarity(const Domain& d1, const Domain& d2) const;
    float compute_transfer_efficiency(
        float source_performance,
        float target_performance,
        float baseline_performance
    ) const;
    bool detect_negative_transfer(
        const Domain& source,
        const Domain& target,
        float performance_delta
    ) const;
    void update_metrics();
    float estimate_task_difficulty(const LearningTask& task) const;
};

} // namespace rros

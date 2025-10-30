#include "meta_learning_engine.hpp"
#include "transfer_learning_manager.hpp"
#include "few_shot_learner.hpp"
#include "adaptive_learning_strategy.hpp"
#include "knowledge_distiller.hpp"
#include "curriculum_optimizer.hpp"
#include "meta_parameter_optimizer.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace rros {

// Domain implementation
Domain::Domain() : similarity_threshold(0.5f) {}

Domain::Domain(const std::string& n, const std::vector<float>& features)
    : name(n), feature_space(features), similarity_threshold(0.5f) {}

// LearningTask implementation
LearningTask::LearningTask() : difficulty(0.5f) {}

// MetaLearningMetrics implementation
MetaLearningMetrics::MetaLearningMetrics()
    : transfer_efficiency(0.0f)
    , few_shot_accuracy(0.0f)
    , adaptation_speed(0.0f)
    , compression_ratio(0.0f)
    , curriculum_optimality(0.0f)
    , total_learning_cycles(0)
    , last_update(std::chrono::high_resolution_clock::now()) {}

// LearningStrategy implementation
LearningStrategy::LearningStrategy()
    : learning_rate(0.01f)
    , momentum(0.9f)
    , performance_score(0.0f) {}

// MetaLearningEngine implementation
MetaLearningEngine::MetaLearningEngine(const std::unordered_map<std::string, float>& config)
    : config_(config)
    , total_transfers_(0)
    , successful_transfers_(0)
    , total_few_shot_tasks_(0)
    , successful_few_shot_tasks_(0) {
    
    // Set default configuration values
    if (config_.find("transfer_threshold") == config_.end()) {
        config_["transfer_threshold"] = 0.7f;
    }
    if (config_.find("few_shot_threshold") == config_.find("few_shot_threshold")) {
        config_["few_shot_threshold"] = 0.8f;
    }
    if (config_.find("min_compression_ratio") == config_.end()) {
        config_["min_compression_ratio"] = 0.5f;
    }
    if (config_.find("adaptation_rate") == config_.end()) {
        config_["adaptation_rate"] = 0.1f;
    }
    
    initialize_subsystems();
}

MetaLearningEngine::~MetaLearningEngine() = default;

void MetaLearningEngine::initialize_subsystems() {
    transfer_manager_ = std::make_unique<TransferLearningManager>(config_);
    few_shot_learner_ = std::make_unique<FewShotLearner>(config_);
    adaptive_strategy_ = std::make_unique<AdaptiveLearningStrategy>(config_);
    knowledge_distiller_ = std::make_unique<KnowledgeDistiller>(config_);
    curriculum_optimizer_ = std::make_unique<CurriculumOptimizer>(config_);
    meta_param_optimizer_ = std::make_unique<MetaParameterOptimizer>(config_);
}

float MetaLearningEngine::transfer_knowledge(
    const Domain& source_domain,
    const Domain& target_domain,
    const LearningTask& task) {
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    total_transfers_++;
    
    // Compute domain similarity
    float similarity = compute_domain_similarity(source_domain, target_domain);
    
    // Check if transfer is viable
    if (similarity < config_["transfer_threshold"]) {
        // Domains too dissimilar, transfer might have negative effects
        return 0.0f;
    }
    
    // Perform knowledge transfer using transfer manager
    float transfer_effectiveness = transfer_manager_->transfer(
        source_domain,
        target_domain,
        task
    );
    
    // Detect negative transfer
    float baseline_performance = 0.5f; // Default baseline
    if (detect_negative_transfer(source_domain, target_domain, transfer_effectiveness - baseline_performance)) {
        return 0.0f;
    }
    
    if (transfer_effectiveness >= config_["transfer_threshold"]) {
        successful_transfers_++;
    }
    
    // Update metrics
    metrics_.transfer_efficiency = static_cast<float>(successful_transfers_) / 
                                   static_cast<float>(total_transfers_);
    metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    return transfer_effectiveness;
}

float MetaLearningEngine::learn_few_shot(
    const LearningTask& task,
    uint32_t num_shots) {
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    total_few_shot_tasks_++;
    
    // Validate task has enough examples
    if (task.training_examples.size() < num_shots) {
        return 0.0f;
    }
    
    // Use few-shot learner to learn from minimal examples
    float accuracy = few_shot_learner_->learn(task, num_shots);
    
    if (accuracy >= config_["few_shot_threshold"]) {
        successful_few_shot_tasks_++;
    }
    
    // Update metrics
    metrics_.few_shot_accuracy = static_cast<float>(successful_few_shot_tasks_) /
                                 static_cast<float>(total_few_shot_tasks_);
    metrics_.total_learning_cycles++;
    metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    return accuracy;
}

LearningStrategy MetaLearningEngine::adapt_learning_strategy(
    const LearningTask& task,
    const std::unordered_map<std::string, float>& performance_feedback) {
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    // Estimate task difficulty
    float difficulty = estimate_task_difficulty(task);
    
    // Use adaptive strategy selector to choose optimal strategy
    LearningStrategy strategy = adaptive_strategy_->select_strategy(
        task,
        performance_feedback,
        difficulty
    );
    
    // Record strategy in history
    strategy_history_.push_back(strategy);
    if (strategy_history_.size() > 1000) {
        strategy_history_.erase(strategy_history_.begin());
    }
    
    // Update adaptation speed metric
    if (strategy_history_.size() >= 2) {
        float strategy_change = std::abs(
            strategy.performance_score - 
            strategy_history_[strategy_history_.size() - 2].performance_score
        );
        metrics_.adaptation_speed = config_["adaptation_rate"] * strategy_change +
                                    (1.0f - config_["adaptation_rate"]) * metrics_.adaptation_speed;
    }
    
    metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    return strategy;
}

std::vector<float> MetaLearningEngine::distill_knowledge(
    const std::vector<float>& source_model,
    size_t target_size,
    float preserve_performance) {
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    // Validate inputs
    if (source_model.empty() || target_size == 0 || target_size >= source_model.size()) {
        return source_model;
    }
    
    // Use knowledge distiller to compress model
    std::vector<float> distilled_model = knowledge_distiller_->distill(
        source_model,
        target_size,
        preserve_performance
    );
    
    // Update compression ratio metric
    metrics_.compression_ratio = static_cast<float>(target_size) / 
                                 static_cast<float>(source_model.size());
    metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    return distilled_model;
}

std::vector<std::string> MetaLearningEngine::optimize_curriculum(
    const std::vector<LearningTask>& tasks,
    const std::unordered_map<std::string, float>& learner_state) {
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    // Use curriculum optimizer to find optimal learning sequence
    std::vector<std::string> task_sequence = curriculum_optimizer_->optimize(
        tasks,
        learner_state
    );
    
    // Evaluate curriculum quality
    float curriculum_quality = curriculum_optimizer_->evaluate_curriculum(task_sequence, tasks);
    
    // Update metrics
    metrics_.curriculum_optimality = curriculum_quality;
    metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    return task_sequence;
}

std::unordered_map<std::string, float> MetaLearningEngine::optimize_meta_parameters(
    const std::unordered_map<std::string, float>& current_params,
    const std::vector<float>& performance_history) {
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    // Use meta-parameter optimizer to tune parameters
    auto optimized_params = meta_param_optimizer_->optimize(
        current_params,
        performance_history
    );
    
    // Update configuration with optimized parameters
    for (const auto& [key, value] : optimized_params) {
        config_[key] = value;
    }
    
    metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    return optimized_params;
}

std::unordered_map<std::string, float> MetaLearningEngine::validate_transfer(
    const Domain& source_domain,
    const Domain& target_domain,
    const std::vector<LearningTask>& validation_tasks) {
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    std::unordered_map<std::string, float> validation_metrics;
    
    float total_effectiveness = 0.0f;
    float positive_transfers = 0.0f;
    float negative_transfers = 0.0f;
    
    for (const auto& task : validation_tasks) {
        // Perform transfer for validation
        float effectiveness = transfer_manager_->transfer(source_domain, target_domain, task);
        total_effectiveness += effectiveness;
        
        float baseline = 0.5f;
        if (effectiveness > baseline) {
            positive_transfers++;
        } else if (effectiveness < baseline * 0.9f) {
            negative_transfers++;
        }
    }
    
    validation_metrics["average_effectiveness"] = validation_tasks.empty() ? 
        0.0f : total_effectiveness / static_cast<float>(validation_tasks.size());
    validation_metrics["positive_transfer_rate"] = validation_tasks.empty() ?
        0.0f : positive_transfers / static_cast<float>(validation_tasks.size());
    validation_metrics["negative_transfer_rate"] = validation_tasks.empty() ?
        0.0f : negative_transfers / static_cast<float>(validation_tasks.size());
    validation_metrics["domain_similarity"] = compute_domain_similarity(source_domain, target_domain);
    
    return validation_metrics;
}

void MetaLearningEngine::update_config(const std::unordered_map<std::string, float>& config) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    for (const auto& [key, value] : config) {
        config_[key] = value;
    }
    
    // Propagate config updates to subsystems
    transfer_manager_->update_config(config_);
    few_shot_learner_->update_config(config_);
    adaptive_strategy_->update_config(config_);
    knowledge_distiller_->update_config(config_);
    curriculum_optimizer_->update_config(config_);
    meta_param_optimizer_->update_config(config_);
}

void MetaLearningEngine::reset() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    metrics_ = MetaLearningMetrics();
    registered_domains_.clear();
    strategy_history_.clear();
    total_transfers_ = 0;
    successful_transfers_ = 0;
    total_few_shot_tasks_ = 0;
    successful_few_shot_tasks_ = 0;
    
    // Reset all subsystems
    transfer_manager_->reset();
    few_shot_learner_->reset();
    adaptive_strategy_->reset();
    knowledge_distiller_->reset();
    curriculum_optimizer_->reset();
    meta_param_optimizer_->reset();
}

std::unordered_map<std::string, float> MetaLearningEngine::get_statistics() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    std::unordered_map<std::string, float> stats;
    
    stats["total_transfers"] = static_cast<float>(total_transfers_);
    stats["successful_transfers"] = static_cast<float>(successful_transfers_);
    stats["transfer_success_rate"] = total_transfers_ > 0 ?
        static_cast<float>(successful_transfers_) / static_cast<float>(total_transfers_) : 0.0f;
    
    stats["total_few_shot_tasks"] = static_cast<float>(total_few_shot_tasks_);
    stats["successful_few_shot_tasks"] = static_cast<float>(successful_few_shot_tasks_);
    stats["few_shot_success_rate"] = total_few_shot_tasks_ > 0 ?
        static_cast<float>(successful_few_shot_tasks_) / static_cast<float>(total_few_shot_tasks_) : 0.0f;
    
    stats["transfer_efficiency"] = metrics_.transfer_efficiency;
    stats["few_shot_accuracy"] = metrics_.few_shot_accuracy;
    stats["adaptation_speed"] = metrics_.adaptation_speed;
    stats["compression_ratio"] = metrics_.compression_ratio;
    stats["curriculum_optimality"] = metrics_.curriculum_optimality;
    stats["total_learning_cycles"] = static_cast<float>(metrics_.total_learning_cycles);
    stats["registered_domains"] = static_cast<float>(registered_domains_.size());
    stats["strategy_history_size"] = static_cast<float>(strategy_history_.size());
    
    return stats;
}

void MetaLearningEngine::register_domain(const Domain& domain) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    // Check if domain already registered
    for (const auto& d : registered_domains_) {
        if (d.name == domain.name) {
            return; // Already registered
        }
    }
    
    registered_domains_.push_back(domain);
}

std::vector<Domain> MetaLearningEngine::get_registered_domains() const {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return registered_domains_;
}

// Private methods

float MetaLearningEngine::compute_domain_similarity(const Domain& d1, const Domain& d2) const {
    // Compute cosine similarity between domain feature spaces
    if (d1.feature_space.empty() || d2.feature_space.empty()) {
        return 0.0f;
    }
    
    size_t min_size = std::min(d1.feature_space.size(), d2.feature_space.size());
    
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += d1.feature_space[i] * d2.feature_space[i];
        norm1 += d1.feature_space[i] * d1.feature_space[i];
        norm2 += d2.feature_space[i] * d2.feature_space[i];
    }
    
    float denominator = std::sqrt(norm1) * std::sqrt(norm2);
    if (denominator < 1e-6f) {
        return 0.0f;
    }
    
    return dot_product / denominator;
}

float MetaLearningEngine::compute_transfer_efficiency(
    float source_performance,
    float target_performance,
    float baseline_performance) const {
    
    // Transfer efficiency = (target_perf - baseline) / (source_perf - baseline)
    float numerator = target_performance - baseline_performance;
    float denominator = source_performance - baseline_performance;
    
    if (denominator < 1e-6f) {
        return 0.0f;
    }
    
    return std::max(0.0f, std::min(1.0f, numerator / denominator));
}

bool MetaLearningEngine::detect_negative_transfer(
    const Domain& source,
    const Domain& target,
    float performance_delta) const {
    
    // Negative transfer occurs when performance decreases significantly
    if (performance_delta < -0.1f) {
        return true;
    }
    
    // Also check domain characteristics for incompatibility
    float similarity = compute_domain_similarity(source, target);
    if (similarity < 0.3f && performance_delta < 0.0f) {
        return true; // Low similarity with negative performance = negative transfer
    }
    
    return false;
}

void MetaLearningEngine::update_metrics() {
    metrics_.transfer_efficiency = total_transfers_ > 0 ?
        static_cast<float>(successful_transfers_) / static_cast<float>(total_transfers_) : 0.0f;
    metrics_.few_shot_accuracy = total_few_shot_tasks_ > 0 ?
        static_cast<float>(successful_few_shot_tasks_) / static_cast<float>(total_few_shot_tasks_) : 0.0f;
    metrics_.last_update = std::chrono::high_resolution_clock::now();
}

float MetaLearningEngine::estimate_task_difficulty(const LearningTask& task) const {
    // Estimate difficulty based on task characteristics
    float difficulty = 0.5f; // Default medium difficulty
    
    // Factor in number of training examples (fewer = harder)
    if (!task.training_examples.empty()) {
        difficulty += 0.2f * (1.0f - std::min(1.0f, 
            static_cast<float>(task.training_examples.size()) / 100.0f));
    }
    
    // Factor in explicit difficulty if provided
    if (task.difficulty > 0.0f) {
        difficulty = 0.7f * difficulty + 0.3f * task.difficulty;
    }
    
    // Factor in task characteristics
    auto complexity_it = task.task_characteristics.find("complexity");
    if (complexity_it != task.task_characteristics.end()) {
        difficulty = 0.5f * difficulty + 0.5f * complexity_it->second;
    }
    
    return std::max(0.0f, std::min(1.0f, difficulty));
}

} // namespace rros

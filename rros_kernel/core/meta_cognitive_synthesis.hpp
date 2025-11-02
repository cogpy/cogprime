#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include "meta_cognitive_monitor.hpp"

namespace rros {

// Forward declarations
class HigherOrderThinking;
class CognitiveStrategySelector;
class MetaKnowledgeRepresentation;

/**
 * @brief Meta-cognitive state representation
 * 
 * Captures the complete meta-cognitive state including awareness,
 * understanding, and strategic orientation
 */
struct MetaCognitiveState {
    float self_awareness;           // Degree of self-awareness [0.0, 1.0]
    float process_understanding;    // Understanding of cognitive processes
    float strategic_alignment;      // Alignment with meta-cognitive goals
    float recursive_depth;          // Current depth of recursive processing
    std::vector<float> meta_knowledge_vector;  // Encoded meta-knowledge
    std::unordered_map<std::string, float> insights;  // Current insights
    std::chrono::high_resolution_clock::time_point timestamp;
    
    MetaCognitiveState();
};

/**
 * @brief Cognitive process descriptor
 * 
 * Describes a cognitive process for meta-level reasoning
 */
struct CognitiveProcess {
    std::string name;
    std::string type;               // e.g., "reasoning", "perception", "memory"
    std::vector<float> state;       // Current process state
    float efficiency;               // Measured efficiency [0.0, 1.0]
    float relevance;                // Relevance to current goals
    std::unordered_map<std::string, float> properties;
    
    CognitiveProcess();
    explicit CognitiveProcess(const std::string& proc_name);
};

/**
 * @brief Meta-cognitive strategy
 * 
 * Represents a strategy for meta-cognitive processing
 */
struct MetaCognitiveStrategy {
    std::string name;
    std::string description;
    float effectiveness;            // Historical effectiveness
    float applicability;            // Applicability to current context
    std::vector<std::string> prerequisites;
    std::unordered_map<std::string, float> parameters;
    
    MetaCognitiveStrategy();
    explicit MetaCognitiveStrategy(const std::string& strategy_name);
};

/**
 * @brief Meta-Cognitive Synthesis Framework
 * 
 * Integrates insights from cognitive science, philosophy of mind, and AI
 * to create a unified approach to meta-cognition within CogPrime architecture.
 * 
 * This framework enables:
 * - Higher-order thinking about thinking
 * - Self-awareness and introspection
 * - Reasoning about cognitive processes
 * - Strategy selection and optimization
 * - Recursive meta-cognitive processing
 * - Meta-cognitive knowledge representation
 * - Learning and adaptation of meta-cognitive capabilities
 * 
 * Based on:
 * - Episode 10: Higher-Order Thought
 * - Episode 41: Wisdom and Contemplation  
 * - Episode 50: Integrated cognitive synthesis
 */
class MetaCognitiveSynthesis {
public:
    /**
     * @brief Constructor initializes synthesis framework
     * @param config Configuration parameters
     */
    explicit MetaCognitiveSynthesis(
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~MetaCognitiveSynthesis();
    
    /**
     * @brief Perform higher-order thinking about a cognitive process
     * @param process Process to reason about
     * @param depth Recursive depth for meta-level analysis
     * @return Meta-level insights and understanding
     */
    std::unordered_map<std::string, float> think_about_thinking(
        const CognitiveProcess& process,
        uint32_t depth = 1
    );
    
    /**
     * @brief Generate self-awareness assessment
     * @return Self-awareness metrics and insights
     */
    MetaCognitiveState assess_self_awareness();
    
    /**
     * @brief Introspect on cognitive state
     * @param lookback_window Time window to analyze (in cycles)
     * @return Introspective insights
     */
    std::unordered_map<std::string, float> introspect(uint32_t lookback_window = 100);
    
    /**
     * @brief Analyze cognitive process for optimization
     * @param process Process to analyze
     * @return Analysis results with recommendations
     */
    std::unordered_map<std::string, float> analyze_process(
        const CognitiveProcess& process
    );
    
    /**
     * @brief Select optimal meta-cognitive strategy
     * @param context Current cognitive context
     * @param available_strategies Available strategies
     * @return Selected strategy with confidence
     */
    std::pair<MetaCognitiveStrategy, float> select_strategy(
        const std::unordered_map<std::string, float>& context,
        const std::vector<MetaCognitiveStrategy>& available_strategies
    );
    
    /**
     * @brief Optimize meta-cognitive strategy parameters
     * @param strategy Strategy to optimize
     * @param performance_feedback Recent performance metrics
     * @return Optimized strategy
     */
    MetaCognitiveStrategy optimize_strategy(
        const MetaCognitiveStrategy& strategy,
        const std::unordered_map<std::string, float>& performance_feedback
    );
    
    /**
     * @brief Perform recursive meta-cognitive processing
     * @param initial_state Initial cognitive state
     * @param max_depth Maximum recursion depth
     * @return Final meta-cognitive insights
     */
    std::unordered_map<std::string, float> recursive_process(
        const std::vector<float>& initial_state,
        uint32_t max_depth = 3
    );
    
    /**
     * @brief Store meta-cognitive knowledge
     * @param knowledge Knowledge to store
     * @param context Context for knowledge
     */
    void store_meta_knowledge(
        const std::unordered_map<std::string, float>& knowledge,
        const std::string& context
    );
    
    /**
     * @brief Retrieve relevant meta-cognitive knowledge
     * @param query Query context
     * @return Retrieved knowledge
     */
    std::unordered_map<std::string, float> retrieve_meta_knowledge(
        const std::unordered_map<std::string, float>& query
    );
    
    /**
     * @brief Learn from meta-cognitive experience
     * @param experience Experience data
     * @param outcome Outcome quality
     */
    void learn_from_experience(
        const CognitiveProcess& experience,
        float outcome
    );
    
    /**
     * @brief Adapt meta-cognitive capabilities based on performance
     * @param performance_metrics Recent performance data
     */
    void adapt_capabilities(
        const std::unordered_map<std::string, float>& performance_metrics
    );
    
    /**
     * @brief Get current meta-cognitive state
     * @return Current state
     */
    MetaCognitiveState get_state() const;
    
    /**
     * @brief Reset synthesis framework
     */
    void reset();
    
    /**
     * @brief Get synthesis statistics
     * @return Current statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Register cognitive process for monitoring
     * @param process Process to monitor
     */
    void register_process(const CognitiveProcess& process);
    
    /**
     * @brief Update process state
     * @param process_name Process identifier
     * @param new_state New state vector
     */
    void update_process_state(
        const std::string& process_name,
        const std::vector<float>& new_state
    );

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Sub-components
    std::unique_ptr<MetaCognitiveMonitor> monitor_;
    std::unique_ptr<HigherOrderThinking> higher_order_thinking_;
    std::unique_ptr<CognitiveStrategySelector> strategy_selector_;
    std::unique_ptr<MetaKnowledgeRepresentation> knowledge_rep_;
    
    // Current state
    MetaCognitiveState current_state_;
    
    // Registered processes
    std::unordered_map<std::string, CognitiveProcess> registered_processes_;
    
    // Strategy repository
    std::vector<MetaCognitiveStrategy> strategy_repository_;
    
    // Learning state
    struct LearningState {
        std::unordered_map<std::string, float> experience_weights;
        float learning_rate;
        uint64_t total_experiences;
    } learning_state_;
    
    // Statistics
    uint64_t total_syntheses_;
    uint64_t successful_adaptations_;
    
    // Thread safety
    mutable std::mutex synthesis_mutex_;
    
    // Internal methods
    void initialize_components();
    void update_self_awareness();
    float compute_process_understanding();
    float compute_strategic_alignment();
    void evolve_meta_knowledge(
        const std::unordered_map<std::string, float>& new_insights
    );
    std::vector<float> synthesize_insights(
        const std::vector<std::unordered_map<std::string, float>>& insight_sets
    );
};

} // namespace rros

#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <deque>

namespace rros {

/**
 * @brief Thought abstraction level
 */
enum class ThoughtLevel {
    FIRST_ORDER,        // Direct perception/cognition
    SECOND_ORDER,       // Thinking about thinking
    THIRD_ORDER,        // Thinking about thinking about thinking
    META_LEVEL          // Arbitrary meta-level
};

/**
 * @brief Higher-order thought representation
 */
struct HigherOrderThought {
    ThoughtLevel level;
    std::string content;            // Symbolic representation
    std::vector<float> state;       // Vector representation
    float coherence;                // Internal coherence measure
    float insight_value;            // Insight quality metric
    std::vector<std::string> references;  // References to lower-level thoughts
    
    HigherOrderThought();
    explicit HigherOrderThought(ThoughtLevel lvl);
};

/**
 * @brief Insight generation result
 */
struct Insight {
    std::string type;               // Type of insight
    std::string description;        // Human-readable description
    float confidence;               // Confidence in insight
    float novelty;                  // Novelty score
    std::vector<float> features;    // Feature representation
    std::unordered_map<std::string, float> attributes;
    
    Insight();
};

/**
 * @brief Higher-Order Thinking Engine
 * 
 * Implements the capacity for thinking about thinking - meta-cognition
 * at multiple levels of abstraction. Enables:
 * - Recursive reasoning about cognitive processes
 * - Insight generation through abstraction
 * - Understanding of understanding
 * - Self-reflective analysis
 * 
 * Based on Episode 10: Higher-Order Thought and recursive meta-cognitive
 * capabilities from wisdom literature (Episode 41).
 */
class HigherOrderThinking {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    explicit HigherOrderThinking(
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Destructor
     */
    ~HigherOrderThinking();
    
    /**
     * @brief Generate higher-order thought about a cognitive state
     * @param first_order_state First-order cognitive state
     * @param target_level Target abstraction level
     * @return Higher-order thought representation
     */
    HigherOrderThought generate_higher_order_thought(
        const std::vector<float>& first_order_state,
        ThoughtLevel target_level = ThoughtLevel::SECOND_ORDER
    );
    
    /**
     * @brief Recursively analyze a thought
     * @param thought Thought to analyze
     * @param depth Maximum recursion depth
     * @return Stack of thoughts at increasing abstraction levels
     */
    std::vector<HigherOrderThought> recursive_analysis(
        const HigherOrderThought& thought,
        uint32_t depth = 3
    );
    
    /**
     * @brief Generate insights from higher-order analysis
     * @param thoughts Collection of higher-order thoughts
     * @return Generated insights
     */
    std::vector<Insight> generate_insights(
        const std::vector<HigherOrderThought>& thoughts
    );
    
    /**
     * @brief Evaluate coherence of a thought
     * @param thought Thought to evaluate
     * @return Coherence score [0.0, 1.0]
     */
    float evaluate_coherence(const HigherOrderThought& thought);
    
    /**
     * @brief Synthesize multiple thoughts into unified understanding
     * @param thoughts Thoughts to synthesize
     * @return Synthesized thought
     */
    HigherOrderThought synthesize_thoughts(
        const std::vector<HigherOrderThought>& thoughts
    );
    
    /**
     * @brief Detect patterns in thought sequences
     * @param thought_sequence Sequence of thoughts
     * @return Detected patterns
     */
    std::unordered_map<std::string, float> detect_thought_patterns(
        const std::deque<HigherOrderThought>& thought_sequence
    );
    
    /**
     * @brief Assess understanding depth
     * @param subject Subject matter representation
     * @param thought_history History of thoughts about subject
     * @return Understanding depth score [0.0, 1.0]
     */
    float assess_understanding_depth(
        const std::vector<float>& subject,
        const std::vector<HigherOrderThought>& thought_history
    );
    
    /**
     * @brief Generate meta-cognitive reflection
     * @param cognitive_process Process to reflect on
     * @return Reflection insights
     */
    std::unordered_map<std::string, float> reflect_on_process(
        const std::unordered_map<std::string, float>& cognitive_process
    );
    
    /**
     * @brief Get thinking statistics
     * @return Current statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Reset thinking engine
     */
    void reset();

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Thought history
    std::deque<HigherOrderThought> thought_history_;
    size_t max_history_size_;
    
    // Insight repository
    std::vector<Insight> insight_repository_;
    
    // Statistics
    uint64_t total_thoughts_generated_;
    uint64_t total_insights_generated_;
    uint64_t total_recursive_analyses_;
    
    // Internal methods
    std::vector<float> abstract_to_higher_level(
        const std::vector<float>& state,
        ThoughtLevel current_level,
        ThoughtLevel target_level
    );
    
    float compute_insight_value(const HigherOrderThought& thought);
    
    float compute_novelty(const Insight& insight);
    
    bool is_coherent(
        const HigherOrderThought& thought,
        float threshold = 0.5f
    );
    
    void prune_history();
    
    std::vector<float> encode_thought_symbolic(
        const std::string& content,
        ThoughtLevel level
    );
};

} // namespace rros

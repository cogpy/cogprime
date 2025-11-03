#include "meta_cognitive_synthesis.hpp"
#include "higher_order_thinking.hpp"
#include "cognitive_strategy_selector.hpp"
#include "meta_knowledge_representation.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

// MetaCognitiveState implementation
MetaCognitiveState::MetaCognitiveState()
    : self_awareness(0.5f)
    , process_understanding(0.5f)
    , strategic_alignment(0.5f)
    , recursive_depth(0.0f)
    , timestamp(std::chrono::high_resolution_clock::now())
{}

// CognitiveProcess implementation
CognitiveProcess::CognitiveProcess()
    : efficiency(0.5f)
    , relevance(0.5f)
{}

CognitiveProcess::CognitiveProcess(const std::string& proc_name)
    : name(proc_name)
    , efficiency(0.5f)
    , relevance(0.5f)
{}

// MetaCognitiveStrategy implementation
MetaCognitiveStrategy::MetaCognitiveStrategy()
    : effectiveness(0.5f)
    , applicability(0.5f)
{}

MetaCognitiveStrategy::MetaCognitiveStrategy(const std::string& strategy_name)
    : name(strategy_name)
    , effectiveness(0.5f)
    , applicability(0.5f)
{}

// MetaCognitiveSynthesis implementation
MetaCognitiveSynthesis::MetaCognitiveSynthesis(
    const std::unordered_map<std::string, float>& config
)
    : config_(config)
    , total_syntheses_(0)
    , successful_adaptations_(0)
{
    learning_state_.learning_rate = config.count("learning_rate") ? 
                                    config.at("learning_rate") : 0.01f;
    learning_state_.total_experiences = 0;
    
    initialize_components();
}

MetaCognitiveSynthesis::~MetaCognitiveSynthesis() {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
}

void MetaCognitiveSynthesis::initialize_components() {
    // Initialize sub-components
    monitor_ = std::make_unique<MetaCognitiveMonitor>(config_);
    higher_order_thinking_ = std::make_unique<HigherOrderThinking>(config_);
    strategy_selector_ = std::make_unique<CognitiveStrategySelector>(config_);
    knowledge_rep_ = std::make_unique<MetaKnowledgeRepresentation>(config_);
    
    // Initialize default strategies
    MetaCognitiveStrategy reflective("reflective");
    reflective.description = "Reflective analysis of cognitive processes";
    reflective.effectiveness = 0.6f;
    strategy_repository_.push_back(reflective);
    
    MetaCognitiveStrategy analytical("analytical");
    analytical.description = "Analytical decomposition of problems";
    analytical.effectiveness = 0.7f;
    strategy_repository_.push_back(analytical);
    
    MetaCognitiveStrategy integrative("integrative");
    integrative.description = "Integrative synthesis of insights";
    integrative.effectiveness = 0.65f;
    strategy_repository_.push_back(integrative);
    
    MetaCognitiveStrategy adaptive("adaptive");
    adaptive.description = "Adaptive strategy modification";
    adaptive.effectiveness = 0.5f;
    strategy_repository_.push_back(adaptive);
}

std::unordered_map<std::string, float> MetaCognitiveSynthesis::think_about_thinking(
    const CognitiveProcess& process,
    uint32_t depth
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    total_syntheses_++;
    
    std::unordered_map<std::string, float> insights;
    
    // Generate higher-order thought about the process
    auto hot = higher_order_thinking_->generate_higher_order_thought(
        process.state,
        static_cast<ThoughtLevel>(std::min(depth, 3u))
    );
    
    // Recursive analysis if depth > 1
    if (depth > 1) {
        auto thought_stack = higher_order_thinking_->recursive_analysis(hot, depth);
        auto generated_insights = higher_order_thinking_->generate_insights(thought_stack);
        
        // Convert insights to metric form
        for (size_t i = 0; i < generated_insights.size(); i++) {
            insights["insight_" + std::to_string(i) + "_confidence"] = 
                generated_insights[i].confidence;
            insights["insight_" + std::to_string(i) + "_novelty"] = 
                generated_insights[i].novelty;
        }
    }
    
    // Evaluate process characteristics
    insights["coherence"] = higher_order_thinking_->evaluate_coherence(hot);
    insights["insight_value"] = hot.insight_value;
    insights["recursive_depth"] = static_cast<float>(depth);
    insights["process_efficiency"] = process.efficiency;
    insights["process_relevance"] = process.relevance;
    
    // Update current state
    current_state_.recursive_depth = static_cast<float>(depth);
    evolve_meta_knowledge(insights);
    
    return insights;
}

MetaCognitiveState MetaCognitiveSynthesis::assess_self_awareness() {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    update_self_awareness();
    current_state_.process_understanding = compute_process_understanding();
    current_state_.strategic_alignment = compute_strategic_alignment();
    current_state_.timestamp = std::chrono::high_resolution_clock::now();
    
    return current_state_;
}

std::unordered_map<std::string, float> MetaCognitiveSynthesis::introspect(
    uint32_t lookback_window
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    // Get introspection from monitor
    auto monitor_insights = monitor_->introspect_state(lookback_window);
    
    // Add synthesis-level introspection
    monitor_insights["self_awareness"] = current_state_.self_awareness;
    monitor_insights["process_understanding"] = current_state_.process_understanding;
    monitor_insights["strategic_alignment"] = current_state_.strategic_alignment;
    monitor_insights["total_syntheses"] = static_cast<float>(total_syntheses_);
    monitor_insights["successful_adaptations"] = static_cast<float>(successful_adaptations_);
    
    // Knowledge coherence
    monitor_insights["knowledge_coherence"] = knowledge_rep_->compute_coherence();
    
    return monitor_insights;
}

std::unordered_map<std::string, float> MetaCognitiveSynthesis::analyze_process(
    const CognitiveProcess& process
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    std::unordered_map<std::string, float> analysis;
    
    // Basic metrics
    analysis["efficiency"] = process.efficiency;
    analysis["relevance"] = process.relevance;
    
    // Higher-order analysis
    auto reflection = higher_order_thinking_->reflect_on_process(process.properties);
    for (const auto& [key, value] : reflection) {
        analysis["reflection_" + key] = value;
    }
    
    // Strategy recommendation
    SelectionContext ctx;
    ctx.cognitive_state = process.properties;
    ctx.state_vector = process.state;
    
    auto recommendations = strategy_selector_->recommend_strategies(ctx, 3);
    for (size_t i = 0; i < recommendations.size(); i++) {
        analysis["recommended_strategy_" + std::to_string(i)] = recommendations[i].second;
    }
    
    // Identify optimization opportunities
    if (process.efficiency < 0.6f) {
        analysis["optimization_needed"] = 1.0f;
        analysis["optimization_priority"] = 1.0f - process.efficiency;
    } else {
        analysis["optimization_needed"] = 0.0f;
        analysis["optimization_priority"] = 0.0f;
    }
    
    return analysis;
}

std::pair<MetaCognitiveStrategy, float> MetaCognitiveSynthesis::select_strategy(
    const std::unordered_map<std::string, float>& context,
    const std::vector<MetaCognitiveStrategy>& available_strategies
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    SelectionContext sel_ctx;
    sel_ctx.cognitive_state = context;
    
    // Use strategy selector
    return strategy_selector_->select_strategy(sel_ctx, available_strategies);
}

MetaCognitiveStrategy MetaCognitiveSynthesis::optimize_strategy(
    const MetaCognitiveStrategy& strategy,
    const std::unordered_map<std::string, float>& performance_feedback
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    SelectionContext ctx;
    ctx.cognitive_state = performance_feedback;
    
    // Extract performance history
    std::vector<float> perf_history;
    for (const auto& [key, value] : performance_feedback) {
        if (key.find("performance") != std::string::npos) {
            perf_history.push_back(value);
        }
    }
    
    return strategy_selector_->optimize_strategy(strategy, ctx, perf_history);
}

std::unordered_map<std::string, float> MetaCognitiveSynthesis::recursive_process(
    const std::vector<float>& initial_state,
    uint32_t max_depth
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    std::vector<std::unordered_map<std::string, float>> insight_sets;
    
    // Generate thoughts at increasing levels
    auto hot = higher_order_thinking_->generate_higher_order_thought(
        initial_state,
        ThoughtLevel::FIRST_ORDER
    );
    
    auto thought_stack = higher_order_thinking_->recursive_analysis(hot, max_depth);
    auto insights = higher_order_thinking_->generate_insights(thought_stack);
    
    // Collect insights at each level
    for (const auto& insight : insights) {
        std::unordered_map<std::string, float> level_insights;
        level_insights["confidence"] = insight.confidence;
        level_insights["novelty"] = insight.novelty;
        for (const auto& [key, value] : insight.attributes) {
            level_insights[key] = value;
        }
        insight_sets.push_back(level_insights);
    }
    
    // Synthesize across levels
    auto synthesis_vector = synthesize_insights(insight_sets);
    
    std::unordered_map<std::string, float> result;
    result["max_depth_reached"] = static_cast<float>(max_depth);
    result["total_insights"] = static_cast<float>(insights.size());
    result["average_confidence"] = 0.0f;
    result["average_novelty"] = 0.0f;
    
    for (const auto& insight : insights) {
        result["average_confidence"] += insight.confidence;
        result["average_novelty"] += insight.novelty;
    }
    
    if (!insights.empty()) {
        result["average_confidence"] /= insights.size();
        result["average_novelty"] /= insights.size();
    }
    
    // Store synthesis result
    for (size_t i = 0; i < std::min(synthesis_vector.size(), size_t(10)); i++) {
        result["synthesis_" + std::to_string(i)] = synthesis_vector[i];
    }
    
    return result;
}

void MetaCognitiveSynthesis::store_meta_knowledge(
    const std::unordered_map<std::string, float>& knowledge,
    const std::string& context
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    knowledge_rep_->store_knowledge(knowledge, context);
}

std::unordered_map<std::string, float> MetaCognitiveSynthesis::retrieve_meta_knowledge(
    const std::unordered_map<std::string, float>& query
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    KnowledgeQuery kq;
    kq.query_type = "similarity";
    kq.max_results = 5;
    
    // Convert query to vector (simplified)
    for (const auto& [key, value] : query) {
        kq.query_vector.push_back(value);
    }
    
    auto results = knowledge_rep_->retrieve_knowledge(kq);
    
    std::unordered_map<std::string, float> retrieved;
    for (size_t i = 0; i < results.size(); i++) {
        retrieved["result_" + std::to_string(i) + "_confidence"] = results[i].confidence;
        retrieved["result_" + std::to_string(i) + "_utility"] = results[i].utility;
    }
    
    return retrieved;
}

void MetaCognitiveSynthesis::learn_from_experience(
    const CognitiveProcess& experience,
    float outcome
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    learning_state_.total_experiences++;
    
    // Update experience weights
    std::string key = experience.name + "_" + experience.type;
    if (learning_state_.experience_weights.find(key) == 
        learning_state_.experience_weights.end()) {
        learning_state_.experience_weights[key] = outcome;
    } else {
        float& weight = learning_state_.experience_weights[key];
        weight = (1.0f - learning_state_.learning_rate) * weight + 
                 learning_state_.learning_rate * outcome;
    }
    
    // Store as meta-knowledge
    std::unordered_map<std::string, float> knowledge;
    knowledge["efficiency"] = experience.efficiency;
    knowledge["relevance"] = experience.relevance;
    knowledge["outcome"] = outcome;
    
    knowledge_rep_->store_knowledge(knowledge, experience.type);
}

void MetaCognitiveSynthesis::adapt_capabilities(
    const std::unordered_map<std::string, float>& performance_metrics
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    // Analyze performance trends
    float avg_performance = 0.0f;
    int count = 0;
    for (const auto& [key, value] : performance_metrics) {
        if (key.find("performance") != std::string::npos) {
            avg_performance += value;
            count++;
        }
    }
    
    if (count > 0) {
        avg_performance /= count;
        
        // Adapt learning rate based on performance
        if (avg_performance > 0.7f) {
            // Performing well, reduce learning rate
            learning_state_.learning_rate *= 0.95f;
            successful_adaptations_++;
        } else if (avg_performance < 0.4f) {
            // Performing poorly, increase learning rate
            learning_state_.learning_rate *= 1.05f;
        }
        
        // Clamp learning rate
        learning_state_.learning_rate = std::max(0.001f, 
                                                  std::min(0.1f, learning_state_.learning_rate));
    }
    
    // Update self-awareness based on adaptation
    update_self_awareness();
}

MetaCognitiveState MetaCognitiveSynthesis::get_state() const {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    return current_state_;
}

void MetaCognitiveSynthesis::reset() {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    current_state_ = MetaCognitiveState();
    registered_processes_.clear();
    learning_state_.experience_weights.clear();
    learning_state_.total_experiences = 0;
    total_syntheses_ = 0;
    successful_adaptations_ = 0;
    
    monitor_->reset();
    higher_order_thinking_->reset();
    strategy_selector_->reset();
    knowledge_rep_->reset();
}

std::unordered_map<std::string, float> MetaCognitiveSynthesis::get_statistics() const {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    std::unordered_map<std::string, float> stats;
    stats["total_syntheses"] = static_cast<float>(total_syntheses_);
    stats["successful_adaptations"] = static_cast<float>(successful_adaptations_);
    stats["registered_processes"] = static_cast<float>(registered_processes_.size());
    stats["total_experiences"] = static_cast<float>(learning_state_.total_experiences);
    stats["learning_rate"] = learning_state_.learning_rate;
    stats["self_awareness"] = current_state_.self_awareness;
    stats["process_understanding"] = current_state_.process_understanding;
    stats["strategic_alignment"] = current_state_.strategic_alignment;
    
    return stats;
}

void MetaCognitiveSynthesis::register_process(const CognitiveProcess& process) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    registered_processes_[process.name] = process;
}

void MetaCognitiveSynthesis::update_process_state(
    const std::string& process_name,
    const std::vector<float>& new_state
) {
    std::lock_guard<std::mutex> lock(synthesis_mutex_);
    
    auto it = registered_processes_.find(process_name);
    if (it != registered_processes_.end()) {
        it->second.state = new_state;
    }
}

// Private methods

void MetaCognitiveSynthesis::update_self_awareness() {
    // Self-awareness based on:
    // 1. Coherence of meta-knowledge
    // 2. Success rate of adaptations
    // 3. Quality of introspections
    
    float knowledge_coherence = knowledge_rep_->compute_coherence();
    
    float adaptation_rate = total_syntheses_ > 0 ? 
        static_cast<float>(successful_adaptations_) / static_cast<float>(total_syntheses_) : 0.5f;
    
    float awareness_score = monitor_->get_awareness_score();
    
    current_state_.self_awareness = 0.4f * knowledge_coherence + 
                                    0.3f * adaptation_rate + 
                                    0.3f * awareness_score;
    
    current_state_.self_awareness = std::max(0.0f, std::min(1.0f, current_state_.self_awareness));
}

float MetaCognitiveSynthesis::compute_process_understanding() {
    if (registered_processes_.empty()) {
        return 0.5f;
    }
    
    float total_understanding = 0.0f;
    for (const auto& [name, process] : registered_processes_) {
        // Understanding based on efficiency and knowledge about process
        total_understanding += process.efficiency;
    }
    
    return total_understanding / registered_processes_.size();
}

float MetaCognitiveSynthesis::compute_strategic_alignment() {
    // Alignment based on strategy effectiveness
    float total_effectiveness = 0.0f;
    for (const auto& strategy : strategy_repository_) {
        total_effectiveness += strategy.effectiveness;
    }
    
    if (strategy_repository_.empty()) {
        return 0.5f;
    }
    
    return total_effectiveness / strategy_repository_.size();
}

void MetaCognitiveSynthesis::evolve_meta_knowledge(
    const std::unordered_map<std::string, float>& new_insights
) {
    // Add insights to meta-knowledge vector
    for (const auto& [key, value] : new_insights) {
        current_state_.meta_knowledge_vector.push_back(value);
    }
    
    // Limit vector size
    if (current_state_.meta_knowledge_vector.size() > 100) {
        current_state_.meta_knowledge_vector.erase(
            current_state_.meta_knowledge_vector.begin(),
            current_state_.meta_knowledge_vector.begin() + 10
        );
    }
    
    // Update insights map
    for (const auto& [key, value] : new_insights) {
        current_state_.insights[key] = value;
    }
}

std::vector<float> MetaCognitiveSynthesis::synthesize_insights(
    const std::vector<std::unordered_map<std::string, float>>& insight_sets
) {
    if (insight_sets.empty()) {
        return {};
    }
    
    std::vector<float> synthesis;
    
    // Collect all values
    for (const auto& insight_set : insight_sets) {
        for (const auto& [key, value] : insight_set) {
            synthesis.push_back(value);
        }
    }
    
    // Compute statistics
    if (!synthesis.empty()) {
        float mean = std::accumulate(synthesis.begin(), synthesis.end(), 0.0f) / synthesis.size();
        synthesis.push_back(mean);
        
        float max_val = *std::max_element(synthesis.begin(), synthesis.end());
        synthesis.push_back(max_val);
    }
    
    return synthesis;
}

} // namespace rros

#include "higher_order_thinking.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace rros {

// HigherOrderThought implementation
HigherOrderThought::HigherOrderThought()
    : level(ThoughtLevel::FIRST_ORDER)
    , coherence(0.5f)
    , insight_value(0.0f)
{}

HigherOrderThought::HigherOrderThought(ThoughtLevel lvl)
    : level(lvl)
    , coherence(0.5f)
    , insight_value(0.0f)
{}

// Insight implementation
Insight::Insight()
    : confidence(0.5f)
    , novelty(0.5f)
{}

// HigherOrderThinking implementation
HigherOrderThinking::HigherOrderThinking(
    const std::unordered_map<std::string, float>& config
)
    : config_(config)
    , max_history_size_(static_cast<size_t>(
        config.count("max_thought_history") ? config.at("max_thought_history") : 500
      ))
    , total_thoughts_generated_(0)
    , total_insights_generated_(0)
    , total_recursive_analyses_(0)
{}

HigherOrderThinking::~HigherOrderThinking() {
    thought_history_.clear();
    insight_repository_.clear();
}

HigherOrderThought HigherOrderThinking::generate_higher_order_thought(
    const std::vector<float>& first_order_state,
    ThoughtLevel target_level
) {
    total_thoughts_generated_++;
    
    HigherOrderThought thought(target_level);
    
    // Abstract state to target level
    thought.state = abstract_to_higher_level(
        first_order_state,
        ThoughtLevel::FIRST_ORDER,
        target_level
    );
    
    // Generate symbolic content
    std::stringstream content;
    content << "Thought at level " << static_cast<int>(target_level) << ": ";
    content << "abstraction of " << first_order_state.size() << " features";
    thought.content = content.str();
    
    // Compute coherence
    thought.coherence = evaluate_coherence(thought);
    
    // Compute insight value
    thought.insight_value = compute_insight_value(thought);
    
    // Add to history
    thought_history_.push_back(thought);
    if (thought_history_.size() > max_history_size_) {
        prune_history();
    }
    
    return thought;
}

std::vector<HigherOrderThought> HigherOrderThinking::recursive_analysis(
    const HigherOrderThought& thought,
    uint32_t depth
) {
    total_recursive_analyses_++;
    
    std::vector<HigherOrderThought> thought_stack;
    thought_stack.push_back(thought);
    
    HigherOrderThought current = thought;
    
    for (uint32_t level = 1; level < depth; level++) {
        // Generate next level thought
        ThoughtLevel next_level = static_cast<ThoughtLevel>(
            std::min(static_cast<int>(current.level) + 1, 
                     static_cast<int>(ThoughtLevel::META_LEVEL))
        );
        
        HigherOrderThought next_thought(next_level);
        
        // Abstract current thought state to next level
        next_thought.state = abstract_to_higher_level(
            current.state,
            current.level,
            next_level
        );
        
        // Reference to lower level
        next_thought.references.push_back(current.content);
        
        // Generate content
        std::stringstream content;
        content << "Meta-analysis (level " << static_cast<int>(next_level) << "): ";
        content << "reflection on '" << current.content << "'";
        next_thought.content = content.str();
        
        next_thought.coherence = evaluate_coherence(next_thought);
        next_thought.insight_value = compute_insight_value(next_thought);
        
        thought_stack.push_back(next_thought);
        current = next_thought;
    }
    
    return thought_stack;
}

std::vector<Insight> HigherOrderThinking::generate_insights(
    const std::vector<HigherOrderThought>& thoughts
) {
    std::vector<Insight> insights;
    
    for (const auto& thought : thoughts) {
        // Generate insight if thought has sufficient quality
        if (thought.insight_value > 0.3f && thought.coherence > 0.4f) {
            Insight insight;
            insight.type = "higher_order";
            insight.description = thought.content;
            insight.confidence = thought.coherence;
            insight.novelty = compute_novelty(insight);
            insight.features = thought.state;
            
            // Add attributes
            insight.attributes["level"] = static_cast<float>(thought.level);
            insight.attributes["coherence"] = thought.coherence;
            insight.attributes["value"] = thought.insight_value;
            
            insights.push_back(insight);
            insight_repository_.push_back(insight);
            total_insights_generated_++;
        }
    }
    
    return insights;
}

float HigherOrderThinking::evaluate_coherence(const HigherOrderThought& thought) {
    if (thought.state.empty()) {
        return 0.0f;
    }
    
    // Coherence based on:
    // 1. Internal consistency of state vector
    // 2. Relation to thought history
    
    // Compute variance of state (lower variance = more coherent)
    float mean = std::accumulate(thought.state.begin(), thought.state.end(), 0.0f) / 
                 thought.state.size();
    
    float variance = 0.0f;
    for (float val : thought.state) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= thought.state.size();
    
    float internal_coherence = 1.0f / (1.0f + std::sqrt(variance));
    
    // Compare with recent thought history
    float historical_coherence = 0.5f;
    if (!thought_history_.empty() && thought_history_.size() > 5) {
        size_t start = thought_history_.size() > 10 ? thought_history_.size() - 10 : 0;
        float similarity_sum = 0.0f;
        int count = 0;
        
        for (size_t i = start; i < thought_history_.size(); i++) {
            const auto& hist_thought = thought_history_[i];
            if (hist_thought.level == thought.level && !hist_thought.state.empty()) {
                // Compute cosine similarity
                float dot = 0.0f;
                float norm1 = 0.0f;
                float norm2 = 0.0f;
                
                size_t min_size = std::min(thought.state.size(), hist_thought.state.size());
                for (size_t j = 0; j < min_size; j++) {
                    dot += thought.state[j] * hist_thought.state[j];
                    norm1 += thought.state[j] * thought.state[j];
                    norm2 += hist_thought.state[j] * hist_thought.state[j];
                }
                
                if (norm1 > 0 && norm2 > 0) {
                    float similarity = dot / (std::sqrt(norm1) * std::sqrt(norm2));
                    similarity_sum += similarity;
                    count++;
                }
            }
        }
        
        if (count > 0) {
            historical_coherence = similarity_sum / count;
        }
    }
    
    return 0.6f * internal_coherence + 0.4f * historical_coherence;
}

HigherOrderThought HigherOrderThinking::synthesize_thoughts(
    const std::vector<HigherOrderThought>& thoughts
) {
    if (thoughts.empty()) {
        return HigherOrderThought();
    }
    
    // Find highest level
    ThoughtLevel max_level = ThoughtLevel::FIRST_ORDER;
    for (const auto& thought : thoughts) {
        if (static_cast<int>(thought.level) > static_cast<int>(max_level)) {
            max_level = thought.level;
        }
    }
    
    HigherOrderThought synthesis(max_level);
    
    // Combine state vectors
    std::vector<float> combined_state;
    for (const auto& thought : thoughts) {
        for (float val : thought.state) {
            combined_state.push_back(val);
        }
    }
    
    // Reduce dimensionality via averaging
    size_t target_size = 32;  // Fixed synthesis size
    synthesis.state.resize(target_size, 0.0f);
    
    for (size_t i = 0; i < combined_state.size(); i++) {
        synthesis.state[i % target_size] += combined_state[i];
    }
    
    for (float& val : synthesis.state) {
        val /= (combined_state.size() / target_size + 1);
    }
    
    // Synthesize content
    synthesis.content = "Synthesis of " + std::to_string(thoughts.size()) + " thoughts";
    
    // Average coherence
    float total_coherence = 0.0f;
    for (const auto& thought : thoughts) {
        total_coherence += thought.coherence;
    }
    synthesis.coherence = total_coherence / thoughts.size();
    
    synthesis.insight_value = compute_insight_value(synthesis);
    
    return synthesis;
}

std::unordered_map<std::string, float> HigherOrderThinking::detect_thought_patterns(
    const std::deque<HigherOrderThought>& thought_sequence
) {
    std::unordered_map<std::string, float> patterns;
    
    if (thought_sequence.size() < 3) {
        return patterns;
    }
    
    // Detect increasing abstraction
    int abstraction_increases = 0;
    for (size_t i = 1; i < thought_sequence.size(); i++) {
        if (static_cast<int>(thought_sequence[i].level) > 
            static_cast<int>(thought_sequence[i-1].level)) {
            abstraction_increases++;
        }
    }
    patterns["increasing_abstraction"] = 
        static_cast<float>(abstraction_increases) / (thought_sequence.size() - 1);
    
    // Detect coherence trends
    int coherence_increases = 0;
    for (size_t i = 1; i < thought_sequence.size(); i++) {
        if (thought_sequence[i].coherence > thought_sequence[i-1].coherence) {
            coherence_increases++;
        }
    }
    patterns["improving_coherence"] = 
        static_cast<float>(coherence_increases) / (thought_sequence.size() - 1);
    
    // Detect insight generation
    int insights = 0;
    for (const auto& thought : thought_sequence) {
        if (thought.insight_value > 0.5f) {
            insights++;
        }
    }
    patterns["insight_density"] = static_cast<float>(insights) / thought_sequence.size();
    
    return patterns;
}

float HigherOrderThinking::assess_understanding_depth(
    const std::vector<float>& subject,
    const std::vector<HigherOrderThought>& thought_history
) {
    if (thought_history.empty()) {
        return 0.0f;
    }
    
    // Understanding depth based on:
    // 1. Number of abstraction levels reached
    // 2. Coherence of thoughts
    // 3. Insight generation rate
    
    ThoughtLevel max_level = ThoughtLevel::FIRST_ORDER;
    float total_coherence = 0.0f;
    float total_insight = 0.0f;
    
    for (const auto& thought : thought_history) {
        if (static_cast<int>(thought.level) > static_cast<int>(max_level)) {
            max_level = thought.level;
        }
        total_coherence += thought.coherence;
        total_insight += thought.insight_value;
    }
    
    float avg_coherence = total_coherence / thought_history.size();
    float avg_insight = total_insight / thought_history.size();
    
    float abstraction_depth = static_cast<float>(max_level) / 
                              static_cast<float>(ThoughtLevel::META_LEVEL);
    
    return 0.4f * abstraction_depth + 0.3f * avg_coherence + 0.3f * avg_insight;
}

std::unordered_map<std::string, float> HigherOrderThinking::reflect_on_process(
    const std::unordered_map<std::string, float>& cognitive_process
) {
    std::unordered_map<std::string, float> reflection;
    
    // Analyze process characteristics
    float process_complexity = 0.0f;
    float process_quality = 0.0f;
    int count = 0;
    
    for (const auto& [key, value] : cognitive_process) {
        process_quality += value;
        process_complexity += std::abs(value - 0.5f);
        count++;
    }
    
    if (count > 0) {
        process_quality /= count;
        process_complexity /= count;
    }
    
    reflection["process_quality"] = process_quality;
    reflection["process_complexity"] = process_complexity;
    reflection["process_dimensions"] = static_cast<float>(count);
    
    // Generate meta-insight
    if (process_quality > 0.7f) {
        reflection["meta_insight"] = 0.8f;  // High quality recognized
    } else if (process_quality < 0.3f) {
        reflection["meta_insight"] = 0.6f;  // Low quality recognized
    } else {
        reflection["meta_insight"] = 0.5f;  // Moderate quality
    }
    
    return reflection;
}

std::unordered_map<std::string, float> HigherOrderThinking::get_statistics() const {
    std::unordered_map<std::string, float> stats;
    stats["total_thoughts_generated"] = static_cast<float>(total_thoughts_generated_);
    stats["total_insights_generated"] = static_cast<float>(total_insights_generated_);
    stats["total_recursive_analyses"] = static_cast<float>(total_recursive_analyses_);
    stats["thought_history_size"] = static_cast<float>(thought_history_.size());
    stats["insight_repository_size"] = static_cast<float>(insight_repository_.size());
    
    return stats;
}

void HigherOrderThinking::reset() {
    thought_history_.clear();
    insight_repository_.clear();
    total_thoughts_generated_ = 0;
    total_insights_generated_ = 0;
    total_recursive_analyses_ = 0;
}

// Private methods

std::vector<float> HigherOrderThinking::abstract_to_higher_level(
    const std::vector<float>& state,
    ThoughtLevel current_level,
    ThoughtLevel target_level
) {
    if (state.empty()) {
        return {};
    }
    
    int level_diff = static_cast<int>(target_level) - static_cast<int>(current_level);
    if (level_diff <= 0) {
        return state;
    }
    
    std::vector<float> abstracted = state;
    
    // Each level of abstraction reduces dimensionality and increases aggregation
    for (int i = 0; i < level_diff; i++) {
        std::vector<float> next_level;
        size_t chunk_size = std::max(size_t(2), abstracted.size() / 4);
        
        for (size_t j = 0; j < abstracted.size(); j += chunk_size) {
            float chunk_sum = 0.0f;
            size_t end = std::min(j + chunk_size, abstracted.size());
            
            for (size_t k = j; k < end; k++) {
                chunk_sum += abstracted[k];
            }
            
            next_level.push_back(chunk_sum / (end - j));
        }
        
        abstracted = next_level;
    }
    
    return abstracted;
}

float HigherOrderThinking::compute_insight_value(const HigherOrderThought& thought) {
    // Insight value based on:
    // 1. Abstraction level (higher = more insight potential)
    // 2. Coherence (must be coherent to be insightful)
    // 3. Novelty compared to history
    
    float level_factor = static_cast<float>(thought.level) / 
                        static_cast<float>(ThoughtLevel::META_LEVEL);
    
    float novelty = 0.5f;
    if (!thought_history_.empty() && thought_history_.size() > 3) {
        // Compare with recent history
        float max_similarity = 0.0f;
        size_t start = thought_history_.size() > 10 ? thought_history_.size() - 10 : 0;
        
        for (size_t i = start; i < thought_history_.size(); i++) {
            const auto& hist = thought_history_[i];
            if (!hist.state.empty() && !thought.state.empty()) {
                float similarity = 0.0f;
                size_t min_size = std::min(thought.state.size(), hist.state.size());
                
                for (size_t j = 0; j < min_size; j++) {
                    similarity += std::abs(thought.state[j] - hist.state[j]);
                }
                similarity /= min_size;
                max_similarity = std::max(max_similarity, 1.0f - similarity);
            }
        }
        
        novelty = 1.0f - max_similarity;
    }
    
    return 0.3f * level_factor + 0.4f * thought.coherence + 0.3f * novelty;
}

float HigherOrderThinking::compute_novelty(const Insight& insight) {
    // Compare with existing insights
    if (insight_repository_.empty()) {
        return 1.0f;
    }
    
    float max_similarity = 0.0f;
    for (const auto& existing : insight_repository_) {
        if (existing.type == insight.type && !existing.features.empty() && 
            !insight.features.empty()) {
            // Compute similarity
            float similarity = 0.0f;
            size_t min_size = std::min(existing.features.size(), insight.features.size());
            
            for (size_t i = 0; i < min_size; i++) {
                similarity += std::abs(existing.features[i] - insight.features[i]);
            }
            similarity /= min_size;
            
            max_similarity = std::max(max_similarity, 1.0f - similarity);
        }
    }
    
    return 1.0f - max_similarity;
}

bool HigherOrderThinking::is_coherent(
    const HigherOrderThought& thought,
    float threshold
) {
    return thought.coherence >= threshold;
}

void HigherOrderThinking::prune_history() {
    // Remove oldest 20%
    size_t remove_count = max_history_size_ / 5;
    if (thought_history_.size() > remove_count) {
        thought_history_.erase(
            thought_history_.begin(),
            thought_history_.begin() + remove_count
        );
    }
}

std::vector<float> HigherOrderThinking::encode_thought_symbolic(
    const std::string& content,
    ThoughtLevel level
) {
    // Simple encoding: hash content and level into vector
    std::vector<float> encoding;
    encoding.push_back(static_cast<float>(level));
    encoding.push_back(static_cast<float>(content.length()));
    
    // Add character statistics
    float char_sum = 0.0f;
    for (char c : content) {
        char_sum += static_cast<float>(c);
    }
    encoding.push_back(char_sum / content.length());
    
    return encoding;
}

} // namespace rros

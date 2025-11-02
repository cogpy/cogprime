#include "contextual_meaning.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {
namespace meaning {

namespace {
    float vector_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
        
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return (denom > 1e-8f) ? (dot / denom) : 0.0f;
    }
}

struct ContextualMeaning::Impl {
    std::unordered_map<std::string, ContextFrame> context_frames;
    std::unordered_map<std::string, float> config;
    
    // Context transition tracking
    std::unordered_map<std::string, 
        std::unordered_map<std::string, float>> transition_probabilities;
    std::unordered_map<std::string, 
        std::unordered_map<std::string, size_t>> transition_counts;
    
    // Context history
    std::vector<std::string> context_history;
    size_t max_history;
    
    // Statistics
    size_t total_interpretations;
    size_t total_context_switches;
    size_t total_adaptations;
    
    Impl() : max_history(100), total_interpretations(0), 
            total_context_switches(0), total_adaptations(0) {
        config["context_blend_threshold"] = 0.5f;
        config["adaptation_rate"] = 0.1f;
        config["min_context_activation"] = 0.3f;
        config["context_decay_rate"] = 0.95f;
    }
};

ContextualMeaning::ContextualMeaning(
    std::shared_ptr<SemanticRepresentation> semantic_system,
    const std::unordered_map<std::string, float>& config
) : impl_(std::make_unique<Impl>()),
    semantic_system_(semantic_system) {
    for (const auto& [key, value] : config) {
        impl_->config[key] = value;
    }
}

ContextualMeaning::~ContextualMeaning() = default;

void ContextualMeaning::add_context_frame(const ContextFrame& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frame.frame_id.empty()) return;
    
    impl_->context_frames[frame.frame_id] = frame;
    impl_->context_frames[frame.frame_id].timestamp = 
        std::chrono::system_clock::now();
}

ContextFrame* ContextualMeaning::get_context_frame(const std::string& frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->context_frames.find(frame_id);
    return (it != impl_->context_frames.end()) ? &(it->second) : nullptr;
}

void ContextualMeaning::update_context(
    const std::string& frame_id,
    const std::unordered_map<std::string, float>& updates
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->context_frames.find(frame_id);
    if (it != impl_->context_frames.end()) {
        for (const auto& [key, value] : updates) {
            it->second.parameters[key] = value;
        }
        it->second.timestamp = std::chrono::system_clock::now();
    }
}

void ContextualMeaning::activate_context(const std::string& frame_id, float activation) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->context_frames.find(frame_id);
    if (it != impl_->context_frames.end()) {
        it->second.activation = std::clamp(activation, 0.0f, 1.0f);
        
        // Track context switch
        if (!impl_->context_history.empty() && 
            impl_->context_history.back() != frame_id) {
            update_transition_probabilities(impl_->context_history.back(), frame_id);
            impl_->total_context_switches++;
        }
        
        // Add to history
        impl_->context_history.push_back(frame_id);
        if (impl_->context_history.size() > impl_->max_history) {
            impl_->context_history.erase(impl_->context_history.begin());
        }
    }
}

void ContextualMeaning::deactivate_context(const std::string& frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->context_frames.find(frame_id);
    if (it != impl_->context_frames.end()) {
        it->second.activation = 0.0f;
    }
}

SituatedMeaning ContextualMeaning::interpret_in_context(
    const std::string& concept_id,
    const std::string& context_id
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    SituatedMeaning meaning;
    meaning.concept_id = concept_id;
    meaning.context_id = context_id;
    
    auto ctx_it = impl_->context_frames.find(context_id);
    if (ctx_it == impl_->context_frames.end()) {
        return meaning;
    }
    
    // Get base concept from semantic system
    auto* concept = semantic_system_->get_concept(concept_id);
    if (!concept) {
        return meaning;
    }
    
    // Apply contextual modulation to embedding
    meaning.contextualized_embedding = apply_context_modulation(
        concept->embedding, ctx_it->second
    );
    
    // Compute applicability based on context-concept alignment
    meaning.applicability = 0.5f; // Base applicability
    for (const auto& salient_concept : ctx_it->second.salient_concepts) {
        if (salient_concept == concept_id) {
            meaning.applicability = 1.0f;
            break;
        }
        float sim = semantic_system_->compute_similarity(concept_id, salient_concept);
        meaning.applicability = std::max(meaning.applicability, sim * 0.8f);
    }
    
    // Specificity based on how unique this interpretation is to the context
    meaning.specificity = ctx_it->second.activation * 0.7f + 0.3f;
    
    // Confidence combines activation and applicability
    meaning.confidence = (ctx_it->second.activation + meaning.applicability) / 2.0f;
    
    // Extract situated features from context parameters
    for (const auto& [key, value] : ctx_it->second.parameters) {
        meaning.situated_features[key] = value;
    }
    
    impl_->total_interpretations++;
    return meaning;
}

SituatedMeaning ContextualMeaning::interpret_in_multi_context(
    const std::string& concept_id,
    const std::vector<std::string>& context_ids
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    SituatedMeaning combined_meaning;
    combined_meaning.concept_id = concept_id;
    combined_meaning.context_id = "multi_context";
    
    if (context_ids.empty()) return combined_meaning;
    
    // Collect all context frames
    std::vector<ContextFrame> contexts;
    for (const auto& id : context_ids) {
        auto it = impl_->context_frames.find(id);
        if (it != impl_->context_frames.end()) {
            contexts.push_back(it->second);
        }
    }
    
    if (contexts.empty()) return combined_meaning;
    
    // Get base concept
    auto* concept = semantic_system_->get_concept(concept_id);
    if (!concept) return combined_meaning;
    
    // Initialize combined embedding
    size_t embedding_dim = concept->embedding.size();
    combined_meaning.contextualized_embedding.resize(embedding_dim, 0.0f);
    
    float total_weight = 0.0f;
    for (const auto& ctx : contexts) {
        auto modulated = apply_context_modulation(concept->embedding, ctx);
        float weight = ctx.activation * ctx.relevance;
        
        for (size_t i = 0; i < embedding_dim; ++i) {
            combined_meaning.contextualized_embedding[i] += modulated[i] * weight;
        }
        total_weight += weight;
    }
    
    // Normalize
    if (total_weight > 1e-6f) {
        for (float& v : combined_meaning.contextualized_embedding) {
            v /= total_weight;
        }
    }
    
    // Compute aggregated metrics
    combined_meaning.applicability = 0.0f;
    combined_meaning.specificity = 0.0f;
    for (const auto& ctx : contexts) {
        combined_meaning.applicability += ctx.activation / contexts.size();
        combined_meaning.specificity += ctx.relevance / contexts.size();
    }
    combined_meaning.confidence = (combined_meaning.applicability + 
                                   combined_meaning.specificity) / 2.0f;
    
    return combined_meaning;
}

std::vector<SituatedMeaning> ContextualMeaning::generate_interpretations(
    const std::string& concept_id,
    size_t max_interpretations
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<SituatedMeaning> interpretations;
    
    // Generate interpretation for each active context
    std::vector<std::pair<float, std::string>> contexts_by_activation;
    for (const auto& [id, frame] : impl_->context_frames) {
        if (frame.activation >= impl_->config["min_context_activation"]) {
            contexts_by_activation.push_back({frame.activation, id});
        }
    }
    
    // Sort by activation
    std::sort(contexts_by_activation.begin(), contexts_by_activation.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Generate top interpretations
    for (size_t i = 0; i < std::min(max_interpretations, contexts_by_activation.size()); ++i) {
        SituatedMeaning meaning;
        meaning.concept_id = concept_id;
        meaning.context_id = contexts_by_activation[i].second;
        
        auto* concept = semantic_system_->get_concept(concept_id);
        if (concept) {
            auto& ctx = impl_->context_frames[contexts_by_activation[i].second];
            meaning.contextualized_embedding = apply_context_modulation(
                concept->embedding, ctx
            );
            meaning.applicability = ctx.activation;
            meaning.confidence = ctx.relevance;
        }
        
        interpretations.push_back(meaning);
    }
    
    return interpretations;
}

std::vector<float> ContextualMeaning::contextualize_embedding(
    const std::vector<float>& base_embedding,
    const std::vector<std::string>& context_ids
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (base_embedding.empty() || context_ids.empty()) {
        return base_embedding;
    }
    
    std::vector<float> result = base_embedding;
    float total_weight = 0.0f;
    
    for (const auto& ctx_id : context_ids) {
        auto it = impl_->context_frames.find(ctx_id);
        if (it != impl_->context_frames.end()) {
            auto modulated = apply_context_modulation(base_embedding, it->second);
            float weight = it->second.activation;
            
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] += modulated[i] * weight;
            }
            total_weight += weight;
        }
    }
    
    // Normalize
    if (total_weight > 1e-6f) {
        for (float& v : result) {
            v /= (1.0f + total_weight);
        }
    }
    
    return result;
}

float ContextualMeaning::compute_contextual_similarity(
    const std::string& concept1,
    const std::string& concept2,
    const std::string& context_id
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto ctx_it = impl_->context_frames.find(context_id);
    if (ctx_it == impl_->context_frames.end()) {
        return semantic_system_->compute_similarity(concept1, concept2);
    }
    
    auto* c1 = semantic_system_->get_concept(concept1);
    auto* c2 = semantic_system_->get_concept(concept2);
    
    if (!c1 || !c2) return 0.0f;
    
    // Contextualize both embeddings
    auto emb1 = apply_context_modulation(c1->embedding, ctx_it->second);
    auto emb2 = apply_context_modulation(c2->embedding, ctx_it->second);
    
    return vector_similarity(emb1, emb2);
}

void ContextualMeaning::integrate_sensory_context(
    const std::string& context_id,
    const std::vector<float>& sensory_input,
    const std::string& modality
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->context_frames.find(context_id);
    if (it != impl_->context_frames.end()) {
        // Add sensory features to context parameters
        for (size_t i = 0; i < sensory_input.size(); ++i) {
            std::string key = modality + "_" + std::to_string(i);
            it->second.parameters[key] = sensory_input[i];
        }
    }
}

void ContextualMeaning::integrate_motor_context(
    const std::string& context_id,
    const std::vector<float>& motor_state
) {
    integrate_sensory_context(context_id, motor_state, "motor");
}

void ContextualMeaning::integrate_emotional_context(
    const std::string& context_id,
    float valence,
    float arousal
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->context_frames.find(context_id);
    if (it != impl_->context_frames.end()) {
        it->second.parameters["valence"] = valence;
        it->second.parameters["arousal"] = arousal;
        it->second.type = ContextType::EMOTIONAL;
    }
}

void ContextualMeaning::switch_context(
    const std::string& from_context,
    const std::string& to_context,
    float transition_speed
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto from_it = impl_->context_frames.find(from_context);
    auto to_it = impl_->context_frames.find(to_context);
    
    if (from_it != impl_->context_frames.end() && 
        to_it != impl_->context_frames.end()) {
        // Gradual transition
        from_it->second.activation *= (1.0f - transition_speed);
        to_it->second.activation = std::min(1.0f, 
            to_it->second.activation + transition_speed);
        
        update_transition_probabilities(from_context, to_context);
    }
}

ContextFrame ContextualMeaning::blend_contexts(
    const std::vector<std::string>& context_ids,
    const std::vector<float>& weights
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ContextFrame blended;
    blended.frame_id = "blended_" + std::to_string(impl_->context_frames.size());
    
    if (context_ids.empty()) return blended;
    
    // Use equal weights if not provided
    std::vector<float> w = weights;
    if (w.empty()) {
        w.resize(context_ids.size(), 1.0f / context_ids.size());
    }
    
    // Blend parameters
    for (size_t i = 0; i < context_ids.size(); ++i) {
        auto it = impl_->context_frames.find(context_ids[i]);
        if (it != impl_->context_frames.end()) {
            for (const auto& [key, value] : it->second.parameters) {
                blended.parameters[key] = blended.parameters[key] + value * w[i];
            }
            blended.activation += it->second.activation * w[i];
            blended.relevance += it->second.relevance * w[i];
            
            // Merge salient concepts
            for (const auto& concept : it->second.salient_concepts) {
                blended.salient_concepts.push_back(concept);
            }
        }
    }
    
    return blended;
}

void ContextualMeaning::update_context_relevance(
    const std::vector<float>& current_state
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [id, frame] : impl_->context_frames) {
        // Simple relevance update based on parameter matching
        float match_score = 0.0f;
        size_t count = 0;
        
        for (size_t i = 0; i < current_state.size(); ++i) {
            std::string key = "state_" + std::to_string(i);
            if (frame.parameters.count(key)) {
                float diff = std::abs(current_state[i] - frame.parameters[key]);
                match_score += 1.0f - std::min(1.0f, diff);
                count++;
            }
        }
        
        if (count > 0) {
            frame.relevance = match_score / count;
        }
    }
}

std::vector<std::string> ContextualMeaning::get_active_contexts(
    float min_activation
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> active;
    for (const auto& [id, frame] : impl_->context_frames) {
        if (frame.activation >= min_activation) {
            active.push_back(id);
        }
    }
    return active;
}

std::string ContextualMeaning::get_dominant_context() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string dominant;
    float max_activation = 0.0f;
    
    for (const auto& [id, frame] : impl_->context_frames) {
        if (frame.activation > max_activation) {
            max_activation = frame.activation;
            dominant = id;
        }
    }
    
    return dominant;
}

std::vector<std::string> ContextualMeaning::predict_next_contexts(
    const std::string& current_context,
    size_t top_k
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::pair<float, std::string>> predictions;
    
    auto it = impl_->transition_probabilities.find(current_context);
    if (it != impl_->transition_probabilities.end()) {
        for (const auto& [next_ctx, prob] : it->second) {
            predictions.push_back({prob, next_ctx});
        }
    }
    
    std::sort(predictions.begin(), predictions.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<std::string> result;
    for (size_t i = 0; i < std::min(top_k, predictions.size()); ++i) {
        result.push_back(predictions[i].second);
    }
    
    return result;
}

float ContextualMeaning::estimate_context_transition_probability(
    const std::string& from_context,
    const std::string& to_context
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->transition_probabilities.find(from_context);
    if (it != impl_->transition_probabilities.end()) {
        auto prob_it = it->second.find(to_context);
        if (prob_it != it->second.end()) {
            return prob_it->second;
        }
    }
    
    return 0.0f;
}

void ContextualMeaning::adapt_to_context(
    const std::string& context_id,
    const std::vector<float>& feedback
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->context_frames.find(context_id);
    if (it != impl_->context_frames.end()) {
        float learning_rate = impl_->config["adaptation_rate"];
        
        for (size_t i = 0; i < feedback.size(); ++i) {
            std::string key = "adapt_" + std::to_string(i);
            float current = it->second.parameters.count(key) ? 
                           it->second.parameters[key] : 0.0f;
            it->second.parameters[key] = current * (1.0f - learning_rate) + 
                                        feedback[i] * learning_rate;
        }
        
        impl_->total_adaptations++;
    }
}

void ContextualMeaning::learn_context_associations(
    const std::string& context_id,
    const std::vector<std::string>& associated_concepts
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->context_frames.find(context_id);
    if (it != impl_->context_frames.end()) {
        // Add to salient concepts (avoiding duplicates)
        for (const auto& concept : associated_concepts) {
            if (std::find(it->second.salient_concepts.begin(),
                         it->second.salient_concepts.end(), concept) ==
                it->second.salient_concepts.end()) {
                it->second.salient_concepts.push_back(concept);
            }
        }
    }
}

float ContextualMeaning::compute_context_coherence(
    const std::vector<std::string>& context_ids
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (context_ids.size() < 2) return 1.0f;
    
    float total_compatibility = 0.0f;
    size_t count = 0;
    
    for (size_t i = 0; i < context_ids.size(); ++i) {
        for (size_t j = i + 1; j < context_ids.size(); ++j) {
            auto it1 = impl_->context_frames.find(context_ids[i]);
            auto it2 = impl_->context_frames.find(context_ids[j]);
            
            if (it1 != impl_->context_frames.end() && 
                it2 != impl_->context_frames.end()) {
                total_compatibility += compute_context_compatibility(
                    it1->second, it2->second
                );
                count++;
            }
        }
    }
    
    return count > 0 ? total_compatibility / count : 0.0f;
}

std::unordered_map<std::string, float> ContextualMeaning::analyze_context_influence(
    const std::string& concept_id
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, float> influence;
    
    for (const auto& [id, frame] : impl_->context_frames) {
        float inf = 0.0f;
        
        // Check if concept is salient in this context
        for (const auto& salient : frame.salient_concepts) {
            if (salient == concept_id) {
                inf = frame.activation * frame.relevance;
                break;
            }
        }
        
        if (inf > 0.0f) {
            influence[id] = inf;
        }
    }
    
    return influence;
}

std::unordered_map<std::string, float> ContextualMeaning::get_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return {
        {"total_contexts", static_cast<float>(impl_->context_frames.size())},
        {"total_interpretations", static_cast<float>(impl_->total_interpretations)},
        {"total_switches", static_cast<float>(impl_->total_context_switches)},
        {"total_adaptations", static_cast<float>(impl_->total_adaptations)}
    };
}

void ContextualMeaning::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->context_frames.clear();
    impl_->context_history.clear();
    impl_->transition_probabilities.clear();
    impl_->transition_counts.clear();
}

float ContextualMeaning::compute_context_compatibility(
    const ContextFrame& c1,
    const ContextFrame& c2
) const {
    // Type compatibility
    float type_compat = (c1.type == c2.type) ? 1.0f : 0.5f;
    
    // Parameter overlap
    float param_compat = 0.0f;
    size_t common_params = 0;
    for (const auto& [key, val1] : c1.parameters) {
        if (c2.parameters.count(key)) {
            float diff = std::abs(val1 - c2.parameters.at(key));
            param_compat += 1.0f - std::min(1.0f, diff);
            common_params++;
        }
    }
    if (common_params > 0) {
        param_compat /= common_params;
    }
    
    return 0.4f * type_compat + 0.6f * param_compat;
}

std::vector<float> ContextualMeaning::apply_context_modulation(
    const std::vector<float>& embedding,
    const ContextFrame& context
) const {
    if (embedding.empty()) return embedding;
    
    std::vector<float> modulated = embedding;
    
    // Apply simple multiplicative modulation based on context activation
    float modulation_strength = context.activation * context.relevance;
    
    for (float& v : modulated) {
        v *= (1.0f + 0.3f * modulation_strength);
    }
    
    return modulated;
}

void ContextualMeaning::update_transition_probabilities(
    const std::string& from_context,
    const std::string& to_context
) {
    // Update counts
    impl_->transition_counts[from_context][to_context]++;
    
    // Recompute probabilities
    size_t total_transitions = 0;
    for (const auto& [to, count] : impl_->transition_counts[from_context]) {
        total_transitions += count;
    }
    
    if (total_transitions > 0) {
        for (const auto& [to, count] : impl_->transition_counts[from_context]) {
            impl_->transition_probabilities[from_context][to] = 
                static_cast<float>(count) / total_transitions;
        }
    }
}

} // namespace meaning
} // namespace rros

#include "symbolic_subsymbolic_bridge.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {
namespace meaning {

struct SymbolicSubsymbolicBridge::Impl {
    std::unordered_map<std::string, SymbolGrounding> groundings;
    std::unordered_map<std::string, float> config;
    
    size_t total_groundings;
    size_t total_translations;
    
    Impl() : total_groundings(0), total_translations(0) {
        config["grounding_threshold"] = 0.5f;
        config["symbol_activation_threshold"] = 0.3f;
    }
};

SymbolicSubsymbolicBridge::SymbolicSubsymbolicBridge(
    std::shared_ptr<SemanticRepresentation> semantic_system,
    const std::unordered_map<std::string, float>& config
) : impl_(std::make_unique<Impl>()),
    semantic_system_(semantic_system) {
    for (const auto& [key, value] : config) {
        impl_->config[key] = value;
    }
}

SymbolicSubsymbolicBridge::~SymbolicSubsymbolicBridge() = default;

SymbolGrounding SymbolicSubsymbolicBridge::ground_symbol(
    const std::string& symbol,
    const std::vector<float>& sensory_experience
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    SymbolGrounding grounding;
    grounding.symbol = symbol;
    grounding.grounded_embedding = sensory_experience;
    grounding.grounding_strength = 1.0f;
    grounding.grounding_experiences.push_back("experience_" + 
        std::to_string(impl_->total_groundings));
    
    impl_->groundings[symbol] = grounding;
    impl_->total_groundings++;
    
    return grounding;
}

void SymbolicSubsymbolicBridge::update_grounding(
    const std::string& symbol,
    const std::vector<float>& new_experience,
    float learning_rate
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->groundings.find(symbol);
    if (it != impl_->groundings.end()) {
        auto& grounding = it->second;
        
        // Update embedding with exponential moving average
        for (size_t i = 0; i < std::min(grounding.grounded_embedding.size(), 
                                        new_experience.size()); ++i) {
            grounding.grounded_embedding[i] = 
                grounding.grounded_embedding[i] * (1.0f - learning_rate) +
                new_experience[i] * learning_rate;
        }
        
        grounding.grounding_strength = std::min(1.0f, 
            grounding.grounding_strength + learning_rate * 0.1f);
        grounding.grounding_experiences.push_back("update_" + 
            std::to_string(grounding.grounding_experiences.size()));
    }
}

float SymbolicSubsymbolicBridge::get_grounding_strength(
    const std::string& symbol
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->groundings.find(symbol);
    return (it != impl_->groundings.end()) ? it->second.grounding_strength : 0.0f;
}

std::string SymbolicSubsymbolicBridge::vectorize_to_symbol(
    const std::vector<float>& vector
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find nearest grounded symbol
    std::string nearest_symbol;
    float max_similarity = -1.0f;
    
    for (const auto& [symbol, grounding] : impl_->groundings) {
        if (grounding.grounded_embedding.size() != vector.size()) continue;
        
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (size_t i = 0; i < vector.size(); ++i) {
            dot += vector[i] * grounding.grounded_embedding[i];
            norm1 += vector[i] * vector[i];
            norm2 += grounding.grounded_embedding[i] * grounding.grounded_embedding[i];
        }
        
        float denom = std::sqrt(norm1) * std::sqrt(norm2);
        float similarity = (denom > 1e-8f) ? (dot / denom) : 0.0f;
        
        if (similarity > max_similarity) {
            max_similarity = similarity;
            nearest_symbol = symbol;
        }
    }
    
    impl_->total_translations++;
    return nearest_symbol;
}

std::vector<float> SymbolicSubsymbolicBridge::symbolize_to_vector(
    const std::string& symbol
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = impl_->groundings.find(symbol);
    if (it != impl_->groundings.end()) {
        impl_->total_translations++;
        return it->second.grounded_embedding;
    }
    
    // Fallback: get from semantic system
    auto* concept = semantic_system_->get_concept(symbol);
    if (concept) {
        return concept->embedding;
    }
    
    return {};
}

std::vector<std::string> SymbolicSubsymbolicBridge::vector_to_symbols(
    const std::vector<float>& vector,
    size_t max_symbols,
    float min_activation
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::pair<float, std::string>> activations;
    
    // Compute activation for all grounded symbols
    for (const auto& [symbol, grounding] : impl_->groundings) {
        if (grounding.grounded_embedding.size() != vector.size()) continue;
        
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (size_t i = 0; i < vector.size(); ++i) {
            dot += vector[i] * grounding.grounded_embedding[i];
            norm1 += vector[i] * vector[i];
            norm2 += grounding.grounded_embedding[i] * grounding.grounded_embedding[i];
        }
        
        float denom = std::sqrt(norm1) * std::sqrt(norm2);
        float activation = (denom > 1e-8f) ? (dot / denom) : 0.0f;
        
        if (activation >= min_activation) {
            activations.push_back({activation, symbol});
        }
    }
    
    // Sort by activation
    std::sort(activations.begin(), activations.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return top symbols
    std::vector<std::string> symbols;
    for (size_t i = 0; i < std::min(max_symbols, activations.size()); ++i) {
        symbols.push_back(activations[i].second);
    }
    
    return symbols;
}

std::vector<float> SymbolicSubsymbolicBridge::symbolic_to_distributed(
    const std::vector<std::string>& symbolic_expression
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (symbolic_expression.empty()) return {};
    
    // Compose vectors from all symbols
    std::vector<std::vector<float>> embeddings;
    for (const auto& symbol : symbolic_expression) {
        auto it = impl_->groundings.find(symbol);
        if (it != impl_->groundings.end()) {
            embeddings.push_back(it->second.grounded_embedding);
        }
    }
    
    if (embeddings.empty()) return {};
    
    // Average composition
    size_t dim = embeddings[0].size();
    std::vector<float> result(dim, 0.0f);
    
    for (const auto& emb : embeddings) {
        for (size_t i = 0; i < std::min(dim, emb.size()); ++i) {
            result[i] += emb[i];
        }
    }
    
    for (float& v : result) {
        v /= embeddings.size();
    }
    
    return result;
}

std::vector<std::string> SymbolicSubsymbolicBridge::distributed_to_symbolic(
    const std::vector<float>& distributed_representation,
    size_t max_symbols
) {
    return vector_to_symbols(distributed_representation, max_symbols, 
                            impl_->config["symbol_activation_threshold"]);
}

std::vector<float> SymbolicSubsymbolicBridge::compose_symbols(
    const std::vector<std::string>& symbols,
    const std::string& operator_type
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (symbols.empty()) return {};
    
    std::vector<std::vector<float>> embeddings;
    for (const auto& symbol : symbols) {
        auto it = impl_->groundings.find(symbol);
        if (it != impl_->groundings.end()) {
            embeddings.push_back(it->second.grounded_embedding);
        }
    }
    
    if (embeddings.empty()) return {};
    
    size_t dim = embeddings[0].size();
    std::vector<float> result(dim, 0.0f);
    
    if (operator_type == "conjunction") {
        // Element-wise product (AND)
        for (size_t i = 0; i < dim; ++i) {
            result[i] = 1.0f;
            for (const auto& emb : embeddings) {
                if (i < emb.size()) result[i] *= emb[i];
            }
        }
    } else if (operator_type == "disjunction") {
        // Element-wise max (OR)
        for (const auto& emb : embeddings) {
            for (size_t i = 0; i < std::min(dim, emb.size()); ++i) {
                result[i] = std::max(result[i], emb[i]);
            }
        }
    } else {
        // Default: average
        for (const auto& emb : embeddings) {
            for (size_t i = 0; i < std::min(dim, emb.size()); ++i) {
                result[i] += emb[i];
            }
        }
        for (float& v : result) {
            v /= embeddings.size();
        }
    }
    
    return result;
}

std::string SymbolicSubsymbolicBridge::decompose_vector(
    const std::vector<float>& complex_vector
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find best matching symbol
    return vectorize_to_symbol(complex_vector);
}

std::unordered_map<std::string, float> SymbolicSubsymbolicBridge::get_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return {
        {"total_groundings", static_cast<float>(impl_->groundings.size())},
        {"total_translations", static_cast<float>(impl_->total_translations)}
    };
}

void SymbolicSubsymbolicBridge::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->groundings.clear();
}

} // namespace meaning
} // namespace rros

#pragma once

#include "semantic_representation.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

namespace rros {
namespace meaning {

/**
 * @brief Symbol grounding representation
 */
struct SymbolGrounding {
    std::string symbol;
    std::vector<float> grounded_embedding;
    std::vector<std::string> grounding_experiences;
    float grounding_strength;
    
    SymbolGrounding() : grounding_strength(0.0f) {}
};

/**
 * @brief Symbolic-subsymbolic bridge for hybrid reasoning
 * 
 * Enables translation between symbolic representations and
 * distributed subsymbolic embeddings
 */
class SymbolicSubsymbolicBridge {
public:
    explicit SymbolicSubsymbolicBridge(
        std::shared_ptr<SemanticRepresentation> semantic_system,
        const std::unordered_map<std::string, float>& config = {}
    );
    ~SymbolicSubsymbolicBridge();
    
    // Symbol grounding
    SymbolGrounding ground_symbol(
        const std::string& symbol,
        const std::vector<float>& sensory_experience
    );
    
    void update_grounding(
        const std::string& symbol,
        const std::vector<float>& new_experience,
        float learning_rate = 0.1f
    );
    
    float get_grounding_strength(const std::string& symbol) const;
    
    // Vector-to-symbol mapping
    std::string vectorize_to_symbol(const std::vector<float>& vector);
    std::vector<float> symbolize_to_vector(const std::string& symbol);
    
    std::vector<std::string> vector_to_symbols(
        const std::vector<float>& vector,
        size_t max_symbols = 5,
        float min_activation = 0.3f
    );
    
    // Hybrid reasoning
    std::vector<float> symbolic_to_distributed(
        const std::vector<std::string>& symbolic_expression
    );
    
    std::vector<std::string> distributed_to_symbolic(
        const std::vector<float>& distributed_representation,
        size_t max_symbols = 10
    );
    
    // Composition operations
    std::vector<float> compose_symbols(
        const std::vector<std::string>& symbols,
        const std::string& operator_type = "conjunction"
    );
    
    std::string decompose_vector(
        const std::vector<float>& complex_vector
    );
    
    // Statistics
    std::unordered_map<std::string, float> get_statistics() const;
    void clear();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::shared_ptr<SemanticRepresentation> semantic_system_;
    mutable std::mutex mutex_;
};

} // namespace meaning
} // namespace rros

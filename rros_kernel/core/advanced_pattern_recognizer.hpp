#pragma once

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace rros {

/**
 * @brief Pattern types for advanced cognitive pattern recognition
 * 
 * Implements multi-scale hierarchical pattern detection for enhanced
 * reasoning capabilities, targeting 30% improvement over baseline.
 */
enum class PatternType {
    CAUSAL,        // Cause-effect relationships
    ANALOGICAL,    // Similarity-based patterns
    HIERARCHICAL,  // Nested structure patterns
    TEMPORAL,      // Time-based sequences
    UNKNOWN        // Unclassified patterns
};

/**
 * @brief Result of pattern recognition analysis
 */
struct PatternRecognitionResult {
    PatternType dominant_type;
    std::array<float, 4> type_confidences;  // Confidence for each pattern type
    std::vector<float> local_features;      // Local pattern features
    std::vector<float> global_features;     // Global pattern features
    float pattern_strength;                 // Overall pattern strength [0, 1]
    float novelty_score;                    // How novel the pattern is [0, 1]
};

/**
 * @brief Advanced Pattern Recognizer for enhanced cognitive processing
 * 
 * Implements hierarchical multi-scale pattern detection with:
 * - Local and global feature extraction
 * - Pattern type classification (causal, analogical, hierarchical, temporal)
 * - Confidence estimation
 * - Novelty detection for cognitive flexibility
 * 
 * Performance target: <10μs per pattern recognition
 */
class AdvancedPatternRecognizer {
public:
    /**
     * @brief Construct pattern recognizer with specified dimensionality
     * @param feature_dim Dimensionality of input features (default: 512)
     */
    explicit AdvancedPatternRecognizer(size_t feature_dim = 512);
    
    /**
     * @brief Recognize patterns in input data
     * @param input Input feature vector
     * @return Pattern recognition result with type and confidence
     */
    PatternRecognitionResult recognize(const std::vector<float>& input);
    
    /**
     * @brief Extract local patterns from input
     * @param input Input feature vector
     * @return Local pattern features
     */
    std::vector<float> extract_local_patterns(const std::vector<float>& input);
    
    /**
     * @brief Extract global patterns from input
     * @param input Input feature vector
     * @return Global pattern features
     */
    std::vector<float> extract_global_patterns(const std::vector<float>& input);
    
    /**
     * @brief Classify pattern type with confidence scores
     * @param combined_features Combined local and global features
     * @return Array of confidence scores for each pattern type
     */
    std::array<float, 4> classify_pattern_type(const std::vector<float>& combined_features);
    
    /**
     * @brief Calculate novelty score for pattern
     * @param pattern_features Pattern feature vector
     * @return Novelty score [0, 1]
     */
    float calculate_novelty(const std::vector<float>& pattern_features);
    
    /**
     * @brief Get recognition statistics
     * @return Map of statistics (total_patterns, avg_confidence, etc.)
     */
    std::unordered_map<std::string, float> get_statistics() const;
    
    /**
     * @brief Reset pattern history for novelty detection
     */
    void reset();

private:
    size_t feature_dim_;
    size_t local_dim_;
    size_t global_dim_;
    
    // Pattern history for novelty detection
    std::vector<std::vector<float>> pattern_history_;
    static constexpr size_t MAX_HISTORY_SIZE = 100;
    
    // Statistics tracking
    size_t total_patterns_recognized_;
    std::array<size_t, 4> pattern_type_counts_;
    float cumulative_confidence_;
    
    // Helper functions for neural network-like transformations
    std::vector<float> dense_transform(const std::vector<float>& input, 
                                       size_t output_dim,
                                       const std::string& activation = "relu");
    
    float relu(float x) const { return std::max(0.0f, x); }
    float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
    float tanh_activation(float x) const { return std::tanh(x); }
    
    std::vector<float> apply_activation(const std::vector<float>& input, 
                                        const std::string& activation);
    
    std::vector<float> softmax(const std::vector<float>& input);
};

} // namespace rros

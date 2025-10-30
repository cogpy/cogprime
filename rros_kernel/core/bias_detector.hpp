#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
#include <deque>

namespace rros {

/**
 * @brief Common cognitive biases to detect
 */
enum class BiasType {
    CONFIRMATION_BIAS,       // Tendency to confirm existing beliefs
    ANCHORING_BIAS,          // Over-reliance on first information
    AVAILABILITY_BIAS,       // Overweighting recent/available information
    RECENCY_BIAS,           // Overweighting recent events
    OVERCONFIDENCE_BIAS,    // Overestimating accuracy
    UNDERCONFIDENCE_BIAS,   // Underestimating accuracy
    NEGATIVITY_BIAS,        // Overweighting negative information
    PRIMACY_BIAS,           // Overweighting initial information
    CLUSTERING_ILLUSION,    // Seeing patterns in random data
    REGRESSION_TO_MEAN      // Ignoring regression to mean
};

/**
 * @brief Bias detection result
 */
struct BiasDetection {
    BiasType bias_type;
    float strength;              // Bias strength [0.0, 1.0]
    float confidence;            // Detection confidence [0.0, 1.0]
    std::string description;
    std::vector<float> evidence; // Supporting evidence
    
    BiasDetection();
    BiasDetection(BiasType type, float str, float conf, const std::string& desc);
};

/**
 * @brief Bias correction strategy
 */
struct BiasCorrection {
    BiasType bias_type;
    float correction_factor;                    // How much to adjust
    std::unordered_map<std::string, float> parameters;
    
    BiasCorrection();
};

/**
 * @brief Cognitive bias detection and correction system
 * 
 * Monitors cognitive processing for common biases and provides
 * correction mechanisms. Implements bias awareness from Episode 11
 * (Self-Deception) and Episode 42 (Intelligence & Rationality).
 */
class BiasDetector {
public:
    /**
     * @brief Constructor initializes detection system
     * @param config Configuration parameters
     */
    explicit BiasDetector(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~BiasDetector();
    
    /**
     * @brief Detect biases in cognitive processing
     * @param decisions Recent decision sequence
     * @param confidences Confidence estimates for decisions
     * @param outcomes Actual outcomes (if available)
     * @return Detected biases with strength and confidence
     */
    std::vector<BiasDetection> detect_biases(
        const std::vector<std::vector<float>>& decisions,
        const std::vector<float>& confidences,
        const std::vector<float>& outcomes = {}
    );
    
    /**
     * @brief Generate correction for detected bias
     * @param detection Bias detection result
     * @return Correction strategy to apply
     */
    BiasCorrection generate_correction(const BiasDetection& detection);
    
    /**
     * @brief Apply bias correction to decision
     * @param decision Original decision
     * @param correction Correction to apply
     * @return Corrected decision
     */
    std::vector<float> apply_correction(
        const std::vector<float>& decision,
        const BiasCorrection& correction
    );
    
    /**
     * @brief Monitor for confirmation bias
     * @param decisions Sequence of decisions
     * @param prior_beliefs Prior belief state
     * @return Confirmation bias detection
     */
    BiasDetection detect_confirmation_bias(
        const std::vector<std::vector<float>>& decisions,
        const std::vector<float>& prior_beliefs
    );
    
    /**
     * @brief Monitor for overconfidence bias
     * @param confidences Confidence estimates
     * @param accuracies Actual accuracies
     * @return Overconfidence bias detection
     */
    BiasDetection detect_overconfidence(
        const std::vector<float>& confidences,
        const std::vector<float>& accuracies
    );
    
    /**
     * @brief Monitor for recency bias
     * @param decisions Temporal sequence of decisions
     * @param weights Applied weights to decisions
     * @return Recency bias detection
     */
    BiasDetection detect_recency_bias(
        const std::vector<std::vector<float>>& decisions,
        const std::vector<float>& weights
    );
    
    /**
     * @brief Get bias detection statistics
     * @return Current bias statistics
     */
    std::unordered_map<std::string, float> get_bias_statistics() const;
    
    /**
     * @brief Reset detection state
     */
    void reset();

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Detection thresholds
    float confirmation_threshold_;
    float overconfidence_threshold_;
    float recency_threshold_;
    
    // Bias history
    std::deque<BiasDetection> detection_history_;
    size_t max_history_size_;
    
    // Bias occurrence counts
    std::unordered_map<BiasType, uint64_t> bias_counts_;
    
    // Thread safety
    mutable std::mutex detector_mutex_;
    
    // Statistics
    uint64_t total_detections_;
    uint64_t total_corrections_;
    
    // Internal methods
    float compute_decision_consistency(const std::vector<std::vector<float>>& decisions) const;
    float compute_temporal_weighting_bias(const std::vector<float>& weights) const;
    float compute_confidence_calibration_error(
        const std::vector<float>& confidences,
        const std::vector<float>& accuracies
    ) const;
    BiasDetection detect_anchoring_bias(const std::vector<std::vector<float>>& decisions);
    BiasDetection detect_availability_bias(const std::vector<std::vector<float>>& decisions);
    BiasDetection detect_clustering_illusion(const std::vector<std::vector<float>>& decisions);
};

} // namespace rros

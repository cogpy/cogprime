#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>

namespace rros {

/**
 * @brief Confidence estimation methods
 */
enum class ConfidenceMethod {
    ENSEMBLE_VARIANCE,      // Based on variance across ensemble predictions
    PREDICTION_ENTROPY,     // Based on entropy of prediction distribution
    FEATURE_QUALITY,        // Based on input feature quality
    HISTORICAL_ACCURACY,    // Based on historical performance
    HYBRID                  // Combination of multiple methods
};

/**
 * @brief Confidence estimation result
 */
struct ConfidenceEstimate {
    float confidence;                           // Overall confidence [0.0, 1.0]
    float epistemic_uncertainty;                // Model uncertainty
    float aleatoric_uncertainty;                // Data uncertainty
    std::unordered_map<std::string, float> factors; // Contributing factors
    ConfidenceMethod method_used;
    
    ConfidenceEstimate();
};

/**
 * @brief Confidence calibration data
 */
struct CalibrationData {
    std::vector<float> predicted_confidences;
    std::vector<float> actual_accuracies;
    float calibration_error;
    uint64_t sample_count;
    
    CalibrationData();
};

/**
 * @brief Confidence estimation system for cognitive outputs
 * 
 * Provides well-calibrated confidence estimates for decisions,
 * predictions, and cognitive outputs. Tracks calibration quality
 * and adapts estimation methods over time.
 */
class ConfidenceEstimator {
public:
    /**
     * @brief Constructor initializes estimation system
     * @param config Configuration parameters
     */
    explicit ConfidenceEstimator(const std::unordered_map<std::string, float>& config = {});
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~ConfidenceEstimator();
    
    /**
     * @brief Estimate confidence for a cognitive output
     * @param output Output vector to estimate confidence for
     * @param context Contextual information
     * @param method Confidence estimation method to use
     * @return Confidence estimate with uncertainty breakdown
     */
    ConfidenceEstimate estimate_confidence(
        const std::vector<float>& output,
        const std::unordered_map<std::string, float>& context = {},
        ConfidenceMethod method = ConfidenceMethod::HYBRID
    );
    
    /**
     * @brief Update calibration data with actual outcome
     * @param predicted_confidence Previously predicted confidence
     * @param was_correct Whether the prediction was correct
     */
    void update_calibration(float predicted_confidence, bool was_correct);
    
    /**
     * @brief Get current calibration quality
     * @return Calibration error (lower is better)
     */
    float get_calibration_error() const;
    
    /**
     * @brief Get calibration curve data for analysis
     * @param num_bins Number of bins for calibration curve
     * @return Calibration data structure
     */
    CalibrationData get_calibration_data(uint32_t num_bins = 10) const;
    
    /**
     * @brief Adjust confidence based on calibration
     * @param raw_confidence Raw confidence estimate
     * @return Calibrated confidence estimate
     */
    float calibrate_confidence(float raw_confidence);
    
    /**
     * @brief Reset calibration state
     */
    void reset();
    
    /**
     * @brief Get estimation statistics
     * @return Current estimation statistics
     */
    std::unordered_map<std::string, float> get_statistics() const;

private:
    // Configuration
    std::unordered_map<std::string, float> config_;
    
    // Calibration tracking
    std::vector<std::pair<float, bool>> calibration_history_;
    size_t max_calibration_history_;
    
    // Calibration curve (binned)
    std::vector<float> bin_confidences_;
    std::vector<float> bin_accuracies_;
    std::vector<uint64_t> bin_counts_;
    uint32_t num_bins_;
    
    // Historical accuracy by confidence level
    std::unordered_map<int, std::pair<uint64_t, uint64_t>> accuracy_by_confidence_;
    
    // Thread safety
    mutable std::mutex estimator_mutex_;
    
    // Statistics
    uint64_t total_estimates_;
    uint64_t total_calibration_updates_;
    
    // Internal methods
    float estimate_ensemble_variance(const std::vector<float>& output) const;
    float estimate_prediction_entropy(const std::vector<float>& output) const;
    float estimate_feature_quality(const std::unordered_map<std::string, float>& context) const;
    float estimate_historical_accuracy(const std::vector<float>& output) const;
    float estimate_hybrid(
        const std::vector<float>& output,
        const std::unordered_map<std::string, float>& context
    );
    void update_calibration_bins(float predicted_confidence, bool was_correct);
    float compute_calibration_error() const;
};

} // namespace rros

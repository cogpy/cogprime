#include "confidence_estimator.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

// ConfidenceEstimate implementation
ConfidenceEstimate::ConfidenceEstimate()
    : confidence(0.5f)
    , epistemic_uncertainty(0.5f)
    , aleatoric_uncertainty(0.5f)
    , method_used(ConfidenceMethod::HYBRID)
{}

// CalibrationData implementation
CalibrationData::CalibrationData()
    : calibration_error(0.0f)
    , sample_count(0)
{}

// ConfidenceEstimator implementation
ConfidenceEstimator::ConfidenceEstimator(
    const std::unordered_map<std::string, float>& config
)
    : config_(config)
    , max_calibration_history_(static_cast<size_t>(
        config.count("max_calibration_history") ? config.at("max_calibration_history") : 1000
      ))
    , num_bins_(static_cast<uint32_t>(
        config.count("calibration_bins") ? config.at("calibration_bins") : 10
      ))
    , total_estimates_(0)
    , total_calibration_updates_(0)
{
    calibration_history_.clear();
    bin_confidences_.resize(num_bins_, 0.0f);
    bin_accuracies_.resize(num_bins_, 0.0f);
    bin_counts_.resize(num_bins_, 0);
}

ConfidenceEstimator::~ConfidenceEstimator() {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    calibration_history_.clear();
}

ConfidenceEstimate ConfidenceEstimator::estimate_confidence(
    const std::vector<float>& output,
    const std::unordered_map<std::string, float>& context,
    ConfidenceMethod method
) {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    
    ConfidenceEstimate estimate;
    estimate.method_used = method;
    
    float raw_confidence = 0.0f;
    
    switch (method) {
        case ConfidenceMethod::ENSEMBLE_VARIANCE:
            raw_confidence = estimate_ensemble_variance(output);
            estimate.factors["ensemble_variance"] = raw_confidence;
            break;
            
        case ConfidenceMethod::PREDICTION_ENTROPY:
            raw_confidence = estimate_prediction_entropy(output);
            estimate.factors["prediction_entropy"] = raw_confidence;
            break;
            
        case ConfidenceMethod::FEATURE_QUALITY:
            raw_confidence = estimate_feature_quality(context);
            estimate.factors["feature_quality"] = raw_confidence;
            break;
            
        case ConfidenceMethod::HISTORICAL_ACCURACY:
            raw_confidence = estimate_historical_accuracy(output);
            estimate.factors["historical_accuracy"] = raw_confidence;
            break;
            
        case ConfidenceMethod::HYBRID:
        default:
            raw_confidence = estimate_hybrid(output, context);
            estimate.factors["hybrid_score"] = raw_confidence;
            break;
    }
    
    // Apply calibration
    estimate.confidence = calibrate_confidence(raw_confidence);
    
    // Estimate epistemic uncertainty (model uncertainty)
    estimate.epistemic_uncertainty = estimate_ensemble_variance(output);
    
    // Estimate aleatoric uncertainty (data uncertainty)
    float feature_quality = estimate_feature_quality(context);
    estimate.aleatoric_uncertainty = 1.0f - feature_quality;
    
    total_estimates_++;
    
    return estimate;
}

void ConfidenceEstimator::update_calibration(float predicted_confidence, bool was_correct) {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    
    calibration_history_.push_back({predicted_confidence, was_correct});
    
    // Prune history if needed
    if (calibration_history_.size() > max_calibration_history_) {
        calibration_history_.erase(
            calibration_history_.begin(),
            calibration_history_.begin() + max_calibration_history_ / 5
        );
    }
    
    // Update calibration bins
    update_calibration_bins(predicted_confidence, was_correct);
    
    // Update accuracy tracking by confidence level
    int confidence_bucket = static_cast<int>(predicted_confidence * 10);
    confidence_bucket = std::max(0, std::min(10, confidence_bucket));
    
    if (accuracy_by_confidence_.find(confidence_bucket) == accuracy_by_confidence_.end()) {
        accuracy_by_confidence_[confidence_bucket] = {0, 0};
    }
    
    accuracy_by_confidence_[confidence_bucket].first++; // Total count
    if (was_correct) {
        accuracy_by_confidence_[confidence_bucket].second++; // Correct count
    }
    
    total_calibration_updates_++;
}

float ConfidenceEstimator::get_calibration_error() const {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    return compute_calibration_error();
}

CalibrationData ConfidenceEstimator::get_calibration_data(uint32_t num_bins) const {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    
    CalibrationData data;
    data.calibration_error = compute_calibration_error();
    data.sample_count = calibration_history_.size();
    
    // Create binned calibration curve
    std::vector<std::vector<bool>> bins(num_bins);
    std::vector<float> bin_centers(num_bins);
    
    for (size_t i = 0; i < num_bins; i++) {
        bin_centers[i] = (i + 0.5f) / num_bins;
    }
    
    for (const auto& entry : calibration_history_) {
        int bin_idx = static_cast<int>(entry.first * num_bins);
        bin_idx = std::max(0, std::min(static_cast<int>(num_bins) - 1, bin_idx));
        bins[bin_idx].push_back(entry.second);
    }
    
    for (size_t i = 0; i < num_bins; i++) {
        data.predicted_confidences.push_back(bin_centers[i]);
        
        if (bins[i].empty()) {
            data.actual_accuracies.push_back(0.0f);
        } else {
            float accuracy = 0.0f;
            for (bool correct : bins[i]) {
                if (correct) accuracy += 1.0f;
            }
            accuracy /= bins[i].size();
            data.actual_accuracies.push_back(accuracy);
        }
    }
    
    return data;
}

float ConfidenceEstimator::calibrate_confidence(float raw_confidence) {
    // Use calibration curve to adjust confidence
    // For now, use simple isotonic regression approximation
    
    if (calibration_history_.size() < 50) {
        return raw_confidence; // Not enough data for calibration
    }
    
    int bin_idx = static_cast<int>(raw_confidence * num_bins_);
    bin_idx = std::max(0, std::min(static_cast<int>(num_bins_) - 1, bin_idx));
    
    if (bin_counts_[bin_idx] > 0) {
        // Use observed accuracy in this bin as calibrated confidence
        return bin_accuracies_[bin_idx];
    }
    
    return raw_confidence;
}

void ConfidenceEstimator::reset() {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    
    calibration_history_.clear();
    accuracy_by_confidence_.clear();
    
    std::fill(bin_confidences_.begin(), bin_confidences_.end(), 0.0f);
    std::fill(bin_accuracies_.begin(), bin_accuracies_.end(), 0.0f);
    std::fill(bin_counts_.begin(), bin_counts_.end(), 0);
    
    total_estimates_ = 0;
    total_calibration_updates_ = 0;
}

std::unordered_map<std::string, float> ConfidenceEstimator::get_statistics() const {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    
    std::unordered_map<std::string, float> stats;
    stats["total_estimates"] = static_cast<float>(total_estimates_);
    stats["total_calibration_updates"] = static_cast<float>(total_calibration_updates_);
    stats["calibration_error"] = compute_calibration_error();
    stats["calibration_history_size"] = static_cast<float>(calibration_history_.size());
    
    return stats;
}

// Private methods

float ConfidenceEstimator::estimate_ensemble_variance(const std::vector<float>& output) const {
    if (output.empty()) {
        return 0.5f;
    }
    
    // Estimate confidence based on variance in output
    float mean = std::accumulate(output.begin(), output.end(), 0.0f) / output.size();
    
    float variance = 0.0f;
    for (float val : output) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= output.size();
    
    // Lower variance = higher confidence
    float confidence = 1.0f / (1.0f + std::sqrt(variance));
    
    return confidence;
}

float ConfidenceEstimator::estimate_prediction_entropy(const std::vector<float>& output) const {
    if (output.empty()) {
        return 0.5f;
    }
    
    // Normalize output to probability distribution
    float sum = std::accumulate(output.begin(), output.end(), 0.0f);
    if (sum < 1e-6f) {
        return 0.5f;
    }
    
    std::vector<float> probs;
    for (float val : output) {
        probs.push_back(val / sum);
    }
    
    // Compute entropy
    float entropy = 0.0f;
    for (float p : probs) {
        if (p > 1e-6f) {
            entropy -= p * std::log2(p);
        }
    }
    
    // Normalize by max entropy
    float max_entropy = std::log2(probs.size());
    float normalized_entropy = max_entropy > 0 ? entropy / max_entropy : 0.0f;
    
    // Lower entropy = higher confidence
    float confidence = 1.0f - normalized_entropy;
    
    return confidence;
}

float ConfidenceEstimator::estimate_feature_quality(
    const std::unordered_map<std::string, float>& context
) const {
    if (context.empty()) {
        return 0.5f;
    }
    
    // Estimate confidence based on quality indicators in context
    float quality = 0.5f;
    
    if (context.count("input_quality")) {
        quality = context.at("input_quality");
    } else if (context.count("feature_completeness")) {
        quality = context.at("feature_completeness");
    } else {
        // Default: higher values in context suggest higher quality
        float sum = 0.0f;
        for (const auto& entry : context) {
            sum += std::abs(entry.second);
        }
        quality = std::min(1.0f, sum / context.size());
    }
    
    return quality;
}

float ConfidenceEstimator::estimate_historical_accuracy(const std::vector<float>& output) const {
    if (output.empty() || calibration_history_.size() < 10) {
        return 0.5f;
    }
    
    // Use recent historical accuracy as confidence
    size_t lookback = std::min(size_t(50), calibration_history_.size());
    
    uint64_t correct_count = 0;
    for (size_t i = calibration_history_.size() - lookback; i < calibration_history_.size(); i++) {
        if (calibration_history_[i].second) {
            correct_count++;
        }
    }
    
    float accuracy = static_cast<float>(correct_count) / static_cast<float>(lookback);
    
    return accuracy;
}

float ConfidenceEstimator::estimate_hybrid(
    const std::vector<float>& output,
    const std::unordered_map<std::string, float>& context
) {
    // Combine multiple estimation methods
    float variance_conf = estimate_ensemble_variance(output);
    float entropy_conf = estimate_prediction_entropy(output);
    float feature_conf = estimate_feature_quality(context);
    float historical_conf = estimate_historical_accuracy(output);
    
    // Weighted combination
    float hybrid = 0.3f * variance_conf +
                   0.3f * entropy_conf +
                   0.2f * feature_conf +
                   0.2f * historical_conf;
    
    return hybrid;
}

void ConfidenceEstimator::update_calibration_bins(float predicted_confidence, bool was_correct) {
    int bin_idx = static_cast<int>(predicted_confidence * num_bins_);
    bin_idx = std::max(0, std::min(static_cast<int>(num_bins_) - 1, bin_idx));
    
    // Update running average for this bin
    float alpha = 0.1f;
    bin_confidences_[bin_idx] = (1.0f - alpha) * bin_confidences_[bin_idx] +
                                alpha * predicted_confidence;
    
    float accuracy = was_correct ? 1.0f : 0.0f;
    bin_accuracies_[bin_idx] = (1.0f - alpha) * bin_accuracies_[bin_idx] +
                               alpha * accuracy;
    
    bin_counts_[bin_idx]++;
}

float ConfidenceEstimator::compute_calibration_error() const {
    if (calibration_history_.empty()) {
        return 0.0f;
    }
    
    // Compute Expected Calibration Error (ECE)
    float total_error = 0.0f;
    uint64_t total_samples = 0;
    
    for (size_t i = 0; i < num_bins_; i++) {
        if (bin_counts_[i] > 0) {
            float error = std::abs(bin_confidences_[i] - bin_accuracies_[i]);
            total_error += error * bin_counts_[i];
            total_samples += bin_counts_[i];
        }
    }
    
    if (total_samples == 0) {
        return 0.0f;
    }
    
    return total_error / total_samples;
}

} // namespace rros

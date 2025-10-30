#include "bias_detector.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rros {

// BiasDetection implementation
BiasDetection::BiasDetection()
    : bias_type(BiasType::CONFIRMATION_BIAS)
    , strength(0.0f)
    , confidence(0.0f)
{}

BiasDetection::BiasDetection(BiasType type, float str, float conf, const std::string& desc)
    : bias_type(type)
    , strength(str)
    , confidence(conf)
    , description(desc)
{}

// BiasCorrection implementation
BiasCorrection::BiasCorrection()
    : bias_type(BiasType::CONFIRMATION_BIAS)
    , correction_factor(0.0f)
{}

// BiasDetector implementation
BiasDetector::BiasDetector(const std::unordered_map<std::string, float>& config)
    : config_(config)
    , confirmation_threshold_(config.count("confirmation_threshold") ? 
                             config.at("confirmation_threshold") : 0.7f)
    , overconfidence_threshold_(config.count("overconfidence_threshold") ?
                               config.at("overconfidence_threshold") : 0.15f)
    , recency_threshold_(config.count("recency_threshold") ?
                        config.at("recency_threshold") : 0.6f)
    , max_history_size_(static_cast<size_t>(
        config.count("max_history_size") ? config.at("max_history_size") : 500
      ))
    , total_detections_(0)
    , total_corrections_(0)
{
    detection_history_.clear();
}

BiasDetector::~BiasDetector() {
    std::lock_guard<std::mutex> lock(detector_mutex_);
    detection_history_.clear();
}

std::vector<BiasDetection> BiasDetector::detect_biases(
    const std::vector<std::vector<float>>& decisions,
    const std::vector<float>& confidences,
    const std::vector<float>& outcomes
) {
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    std::vector<BiasDetection> detections;
    
    if (decisions.empty()) {
        return detections;
    }
    
    // Detect various biases
    
    // 1. Overconfidence bias (requires outcomes)
    if (!outcomes.empty() && confidences.size() == outcomes.size()) {
        BiasDetection overconf = detect_overconfidence(confidences, outcomes);
        if (overconf.strength > overconfidence_threshold_) {
            detections.push_back(overconf);
            detection_history_.push_back(overconf);
            bias_counts_[BiasType::OVERCONFIDENCE_BIAS]++;
        }
    }
    
    // 2. Recency bias
    if (decisions.size() > 5) {
        std::vector<float> uniform_weights(decisions.size(), 1.0f / decisions.size());
        BiasDetection recency = detect_recency_bias(decisions, uniform_weights);
        if (recency.strength > recency_threshold_) {
            detections.push_back(recency);
            detection_history_.push_back(recency);
            bias_counts_[BiasType::RECENCY_BIAS]++;
        }
    }
    
    // 3. Anchoring bias
    BiasDetection anchoring = detect_anchoring_bias(decisions);
    if (anchoring.strength > 0.5f) {
        detections.push_back(anchoring);
        detection_history_.push_back(anchoring);
        bias_counts_[BiasType::ANCHORING_BIAS]++;
    }
    
    // 4. Clustering illusion
    if (decisions.size() > 10) {
        BiasDetection clustering = detect_clustering_illusion(decisions);
        if (clustering.strength > 0.6f) {
            detections.push_back(clustering);
            detection_history_.push_back(clustering);
            bias_counts_[BiasType::CLUSTERING_ILLUSION]++;
        }
    }
    
    total_detections_ += detections.size();
    
    // Prune history if needed
    if (detection_history_.size() > max_history_size_) {
        detection_history_.erase(
            detection_history_.begin(),
            detection_history_.begin() + max_history_size_ / 5
        );
    }
    
    return detections;
}

BiasCorrection BiasDetector::generate_correction(const BiasDetection& detection) {
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    BiasCorrection correction;
    correction.bias_type = detection.bias_type;
    
    switch (detection.bias_type) {
        case BiasType::OVERCONFIDENCE_BIAS:
            // Reduce confidence estimates
            correction.correction_factor = -detection.strength * 0.2f;
            correction.parameters["confidence_scale"] = 1.0f - detection.strength * 0.3f;
            break;
            
        case BiasType::UNDERCONFIDENCE_BIAS:
            // Increase confidence estimates
            correction.correction_factor = detection.strength * 0.2f;
            correction.parameters["confidence_scale"] = 1.0f + detection.strength * 0.3f;
            break;
            
        case BiasType::RECENCY_BIAS:
            // Redistribute temporal weights more uniformly
            correction.correction_factor = detection.strength;
            correction.parameters["temporal_decay"] = 0.95f + detection.strength * 0.04f;
            break;
            
        case BiasType::CONFIRMATION_BIAS:
            // Increase weight on contradicting evidence
            correction.correction_factor = detection.strength;
            correction.parameters["contradiction_boost"] = 1.0f + detection.strength;
            break;
            
        case BiasType::ANCHORING_BIAS:
            // Reduce influence of initial information
            correction.correction_factor = detection.strength;
            correction.parameters["anchor_weight"] = 0.5f - detection.strength * 0.3f;
            break;
            
        default:
            correction.correction_factor = 0.0f;
            break;
    }
    
    total_corrections_++;
    
    return correction;
}

std::vector<float> BiasDetector::apply_correction(
    const std::vector<float>& decision,
    const BiasCorrection& correction
) {
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    std::vector<float> corrected = decision;
    
    switch (correction.bias_type) {
        case BiasType::OVERCONFIDENCE_BIAS:
        case BiasType::UNDERCONFIDENCE_BIAS:
            // Scale decision values toward mean
            {
                float mean = std::accumulate(decision.begin(), decision.end(), 0.0f) / decision.size();
                float scale = correction.parameters.count("confidence_scale") ?
                             correction.parameters.at("confidence_scale") : 1.0f;
                
                for (size_t i = 0; i < corrected.size(); i++) {
                    corrected[i] = mean + (decision[i] - mean) * scale;
                }
            }
            break;
            
        case BiasType::ANCHORING_BIAS:
            // Reduce influence of first element
            if (!corrected.empty() && correction.parameters.count("anchor_weight")) {
                float anchor_weight = correction.parameters.at("anchor_weight");
                float mean = std::accumulate(decision.begin() + 1, decision.end(), 0.0f) / 
                            (decision.size() - 1);
                corrected[0] = anchor_weight * decision[0] + (1.0f - anchor_weight) * mean;
            }
            break;
            
        default:
            // No specific correction
            break;
    }
    
    return corrected;
}

BiasDetection BiasDetector::detect_confirmation_bias(
    const std::vector<std::vector<float>>& decisions,
    const std::vector<float>& prior_beliefs
) {
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    if (decisions.empty() || prior_beliefs.empty()) {
        return BiasDetection(BiasType::CONFIRMATION_BIAS, 0.0f, 0.0f, 
                           "Insufficient data for confirmation bias detection");
    }
    
    // Measure how much decisions align with prior beliefs
    float alignment_score = 0.0f;
    int alignment_count = 0;
    
    for (const auto& decision : decisions) {
        size_t min_size = std::min(decision.size(), prior_beliefs.size());
        
        for (size_t i = 0; i < min_size; i++) {
            // Check if decision reinforces prior belief
            if ((decision[i] > 0 && prior_beliefs[i] > 0) ||
                (decision[i] < 0 && prior_beliefs[i] < 0)) {
                alignment_score += 1.0f;
            }
            alignment_count++;
        }
    }
    
    float bias_strength = alignment_count > 0 ? alignment_score / alignment_count : 0.0f;
    
    // High alignment suggests confirmation bias
    if (bias_strength > confirmation_threshold_) {
        return BiasDetection(
            BiasType::CONFIRMATION_BIAS,
            bias_strength,
            0.8f,
            "Decisions strongly align with prior beliefs"
        );
    }
    
    return BiasDetection(BiasType::CONFIRMATION_BIAS, bias_strength, 0.5f, 
                        "Normal belief updating");
}

BiasDetection BiasDetector::detect_overconfidence(
    const std::vector<float>& confidences,
    const std::vector<float>& accuracies
) {
    if (confidences.size() != accuracies.size() || confidences.empty()) {
        return BiasDetection(BiasType::OVERCONFIDENCE_BIAS, 0.0f, 0.0f,
                           "Mismatched confidence and accuracy data");
    }
    
    float calibration_error = compute_confidence_calibration_error(confidences, accuracies);
    
    // Check if confidences systematically exceed accuracies
    float confidence_mean = std::accumulate(confidences.begin(), confidences.end(), 0.0f) / 
                           confidences.size();
    float accuracy_mean = std::accumulate(accuracies.begin(), accuracies.end(), 0.0f) /
                         accuracies.size();
    
    float overconfidence_margin = confidence_mean - accuracy_mean;
    
    if (overconfidence_margin > overconfidence_threshold_) {
        return BiasDetection(
            BiasType::OVERCONFIDENCE_BIAS,
            calibration_error,
            0.85f,
            "Confidence systematically exceeds accuracy"
        );
    } else if (overconfidence_margin < -overconfidence_threshold_) {
        return BiasDetection(
            BiasType::UNDERCONFIDENCE_BIAS,
            calibration_error,
            0.85f,
            "Confidence systematically below accuracy"
        );
    }
    
    return BiasDetection(BiasType::OVERCONFIDENCE_BIAS, calibration_error, 0.5f,
                        "Well-calibrated confidence");
}

BiasDetection BiasDetector::detect_recency_bias(
    const std::vector<std::vector<float>>& decisions,
    const std::vector<float>& weights
) {
    if (decisions.size() != weights.size() || decisions.size() < 3) {
        return BiasDetection(BiasType::RECENCY_BIAS, 0.0f, 0.0f,
                           "Insufficient data for recency bias detection");
    }
    
    float temporal_bias = compute_temporal_weighting_bias(weights);
    
    if (temporal_bias > recency_threshold_) {
        return BiasDetection(
            BiasType::RECENCY_BIAS,
            temporal_bias,
            0.75f,
            "Recent information overweighted"
        );
    }
    
    return BiasDetection(BiasType::RECENCY_BIAS, temporal_bias, 0.5f,
                        "Balanced temporal weighting");
}

std::unordered_map<std::string, float> BiasDetector::get_bias_statistics() const {
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    std::unordered_map<std::string, float> stats;
    stats["total_detections"] = static_cast<float>(total_detections_);
    stats["total_corrections"] = static_cast<float>(total_corrections_);
    stats["detection_history_size"] = static_cast<float>(detection_history_.size());
    
    stats["confirmation_bias_count"] = static_cast<float>(
        bias_counts_.count(BiasType::CONFIRMATION_BIAS) ?
        bias_counts_.at(BiasType::CONFIRMATION_BIAS) : 0
    );
    stats["overconfidence_count"] = static_cast<float>(
        bias_counts_.count(BiasType::OVERCONFIDENCE_BIAS) ?
        bias_counts_.at(BiasType::OVERCONFIDENCE_BIAS) : 0
    );
    stats["recency_bias_count"] = static_cast<float>(
        bias_counts_.count(BiasType::RECENCY_BIAS) ?
        bias_counts_.at(BiasType::RECENCY_BIAS) : 0
    );
    
    return stats;
}

void BiasDetector::reset() {
    std::lock_guard<std::mutex> lock(detector_mutex_);
    
    detection_history_.clear();
    bias_counts_.clear();
    total_detections_ = 0;
    total_corrections_ = 0;
}

// Private methods

float BiasDetector::compute_decision_consistency(
    const std::vector<std::vector<float>>& decisions
) const {
    if (decisions.size() < 2) {
        return 1.0f;
    }
    
    // Measure consistency across decisions
    float total_similarity = 0.0f;
    int comparison_count = 0;
    
    for (size_t i = 1; i < decisions.size(); i++) {
        const auto& prev = decisions[i-1];
        const auto& curr = decisions[i];
        
        size_t min_size = std::min(prev.size(), curr.size());
        float similarity = 0.0f;
        
        for (size_t j = 0; j < min_size; j++) {
            similarity += 1.0f - std::abs(curr[j] - prev[j]);
        }
        
        if (min_size > 0) {
            similarity /= min_size;
            total_similarity += similarity;
            comparison_count++;
        }
    }
    
    return comparison_count > 0 ? total_similarity / comparison_count : 1.0f;
}

float BiasDetector::compute_temporal_weighting_bias(
    const std::vector<float>& weights
) const {
    if (weights.size() < 3) {
        return 0.0f;
    }
    
    // Check if recent items have disproportionately high weights
    size_t recent_count = weights.size() / 3;
    float recent_weight = 0.0f;
    float old_weight = 0.0f;
    
    for (size_t i = 0; i < weights.size(); i++) {
        if (i >= weights.size() - recent_count) {
            recent_weight += weights[i];
        } else if (i < recent_count) {
            old_weight += weights[i];
        }
    }
    
    float expected_recent = static_cast<float>(recent_count) / weights.size();
    float actual_recent = recent_weight;
    
    // Positive bias indicates recency bias
    float bias = actual_recent - expected_recent;
    
    return std::max(0.0f, bias);
}

float BiasDetector::compute_confidence_calibration_error(
    const std::vector<float>& confidences,
    const std::vector<float>& accuracies
) const {
    if (confidences.size() != accuracies.size() || confidences.empty()) {
        return 0.0f;
    }
    
    float total_error = 0.0f;
    for (size_t i = 0; i < confidences.size(); i++) {
        total_error += std::abs(confidences[i] - accuracies[i]);
    }
    
    return total_error / confidences.size();
}

BiasDetection BiasDetector::detect_anchoring_bias(
    const std::vector<std::vector<float>>& decisions
) {
    if (decisions.size() < 3) {
        return BiasDetection(BiasType::ANCHORING_BIAS, 0.0f, 0.0f,
                           "Insufficient data");
    }
    
    // Check if first decision has disproportionate influence
    const auto& first_decision = decisions[0];
    float anchor_influence = 0.0f;
    int comparison_count = 0;
    
    for (size_t i = 1; i < decisions.size(); i++) {
        size_t min_size = std::min(first_decision.size(), decisions[i].size());
        
        for (size_t j = 0; j < min_size; j++) {
            float similarity = 1.0f - std::abs(decisions[i][j] - first_decision[j]);
            anchor_influence += similarity;
            comparison_count++;
        }
    }
    
    float bias_strength = comparison_count > 0 ? anchor_influence / comparison_count : 0.0f;
    
    if (bias_strength > 0.7f) {
        return BiasDetection(
            BiasType::ANCHORING_BIAS,
            bias_strength,
            0.7f,
            "Decisions show strong anchoring to initial value"
        );
    }
    
    return BiasDetection(BiasType::ANCHORING_BIAS, bias_strength, 0.5f,
                        "Normal decision independence");
}

BiasDetection BiasDetector::detect_availability_bias(
    const std::vector<std::vector<float>>& decisions
) {
    // Placeholder implementation
    return BiasDetection(BiasType::AVAILABILITY_BIAS, 0.0f, 0.3f,
                        "Availability bias detection not yet implemented");
}

BiasDetection BiasDetector::detect_clustering_illusion(
    const std::vector<std::vector<float>>& decisions
) {
    if (decisions.size() < 10) {
        return BiasDetection(BiasType::CLUSTERING_ILLUSION, 0.0f, 0.0f,
                           "Insufficient data");
    }
    
    // Detect if system sees patterns in what should be random
    float consistency = compute_decision_consistency(decisions);
    
    // High consistency in supposedly independent decisions suggests clustering illusion
    if (consistency > 0.8f) {
        return BiasDetection(
            BiasType::CLUSTERING_ILLUSION,
            consistency,
            0.6f,
            "Possible pattern detection in random data"
        );
    }
    
    return BiasDetection(BiasType::CLUSTERING_ILLUSION, consistency, 0.4f,
                        "Normal pattern recognition");
}

} // namespace rros

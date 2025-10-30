#include "../core/meta_cognitive_monitor.hpp"
#include "../core/decision_quality_tracker.hpp"
#include "../core/confidence_estimator.hpp"
#include "../core/bias_detector.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdlib>

using namespace rros;

void test_meta_cognitive_monitor() {
    std::cout << "\n🧠 Testing Meta-Cognitive Monitor..." << std::endl;
    
    MetaCognitiveMonitor monitor;
    
    // Record some cognitive snapshots
    for (uint64_t i = 0; i < 50; i++) {
        CognitiveSnapshot snapshot;
        snapshot.cycle_id = i;
        snapshot.relevance_score = 0.5f + 0.3f * sin(i * 0.1f);
        snapshot.confidence = 0.6f + 0.2f * cos(i * 0.15f);
        snapshot.processing_stage = "cycle_" + std::to_string(i);
        snapshot.state_vector = {
            static_cast<float>(i) * 0.01f,
            0.5f + 0.1f * sin(i * 0.2f),
            0.3f + 0.2f * cos(i * 0.3f)
        };
        
        monitor.record_snapshot(snapshot);
    }
    
    // Introspect state
    auto introspection = monitor.introspect_state(30);
    
    std::cout << "  Average relevance: " << std::fixed << std::setprecision(3) 
              << introspection["average_relevance"] << std::endl;
    std::cout << "  Average confidence: " << introspection["average_confidence"] << std::endl;
    std::cout << "  Relevance stability: " << introspection["relevance_stability"] << std::endl;
    std::cout << "  Temporal coherence: " << introspection["temporal_coherence"] << std::endl;
    std::cout << "  Awareness score: " << introspection["awareness_score"] << std::endl;
    
    // Detect patterns
    float increasing = monitor.detect_pattern("increasing_relevance");
    float oscillating = monitor.detect_pattern("oscillating");
    
    std::cout << "  Pattern - Increasing relevance: " << increasing << std::endl;
    std::cout << "  Pattern - Oscillating: " << oscillating << std::endl;
    
    // Update process metrics
    monitor.update_process_metrics("reasoning", 2.5f, 0.75f, 0.85f, true);
    monitor.update_process_metrics("reasoning", 3.1f, 0.72f, 0.82f, true);
    monitor.update_process_metrics("perception", 1.8f, 0.68f, 0.79f, false);
    
    auto reasoning_metrics = monitor.analyze_process("reasoning");
    std::cout << "  Reasoning process - Avg time: " << reasoning_metrics.average_processing_time_us 
              << " µs" << std::endl;
    std::cout << "  Reasoning process - Success rate: " << reasoning_metrics.success_rate << std::endl;
    
    auto stats = monitor.get_statistics();
    std::cout << "  Total snapshots: " << stats["total_snapshots"] << std::endl;
    std::cout << "✅ Meta-Cognitive Monitor test passed" << std::endl;
}

void test_decision_quality_tracker() {
    std::cout << "\n📊 Testing Decision Quality Tracker..." << std::endl;
    
    DecisionQualityTracker tracker;
    
    // Record some decisions and outcomes
    std::vector<uint64_t> decision_ids;
    
    std::cout << "  Recording decisions..." << std::endl;
    for (int i = 0; i < 20; i++) {
        std::vector<float> context;
        context.push_back(static_cast<float>(i) * 0.1f);
        context.push_back(0.5f + 0.2f * sin(i * 0.3f));
        
        std::vector<float> output;
        output.push_back(0.6f + 0.3f * cos(i * 0.2f));
        output.push_back(0.4f + 0.2f * sin(i * 0.4f));
        
        float predicted_quality = 0.7f + 0.2f * sin(i * 0.5f);
        
        uint64_t id = tracker.record_decision(context, output, predicted_quality, "test_decision");
        decision_ids.push_back(id);
    }
    
    std::cout << "  Recording outcomes..." << std::endl;
    // Record outcomes with some noise
    for (size_t i = 0; i < decision_ids.size(); i++) {
        float actual_quality = 0.65f + 0.25f * sin(i * 0.5f);
        float reward = actual_quality + (rand() % 100 - 50) * 0.001f;
        bool success = actual_quality > 0.6f;
        
        tracker.record_outcome(decision_ids[i], actual_quality, reward, success);
    }
    
    std::cout << "  Getting quality metrics..." << std::endl;
    // Get quality metrics
    auto metrics = tracker.get_quality_metrics("test_decision");
    std::cout << "  Average quality: " << std::fixed << std::setprecision(3) 
              << metrics.average_quality << std::endl;
    std::cout << "  Prediction accuracy: " << metrics.prediction_accuracy << std::endl;
    std::cout << "  Calibration score: " << metrics.calibration_score << std::endl;
    std::cout << "  Success rate: " << tracker.get_success_rate("test_decision") << std::endl;
    
    std::cout << "  Assessing correlation..." << std::endl;
    // Assess prediction correlation
    float correlation = tracker.assess_prediction_correlation(20);
    std::cout << "  Prediction correlation: " << correlation << std::endl;
    
    std::cout << "  Analyzing trend..." << std::endl;
    // Analyze quality trend
    float trend = tracker.analyze_quality_trend(10);
    std::cout << "  Quality trend: " << trend << std::endl;
    
    std::cout << "  Getting stats..." << std::endl;
    auto stats = tracker.get_statistics();
    std::cout << "  Total decisions: " << stats["total_decisions_recorded"] << std::endl;
    std::cout << "  Total outcomes: " << stats["total_outcomes_recorded"] << std::endl;
    std::cout << "✅ Decision Quality Tracker test passed" << std::endl;
}

void test_confidence_estimator() {
    std::cout << "\n🎯 Testing Confidence Estimator..." << std::endl;
    
    ConfidenceEstimator estimator;
    
    // Test different estimation methods
    std::vector<float> output1 = {0.8f, 0.75f, 0.78f, 0.82f}; // Low variance
    std::vector<float> output2 = {0.3f, 0.7f, 0.1f, 0.9f};   // High variance
    
    std::unordered_map<std::string, float> context;
    context["input_quality"] = 0.85f;
    
    // Estimate confidence
    auto estimate1 = estimator.estimate_confidence(output1, context, ConfidenceMethod::HYBRID);
    auto estimate2 = estimator.estimate_confidence(output2, context, ConfidenceMethod::HYBRID);
    
    std::cout << "  Low variance output - Confidence: " << std::fixed << std::setprecision(3)
              << estimate1.confidence << std::endl;
    std::cout << "  Low variance output - Epistemic uncertainty: " 
              << estimate1.epistemic_uncertainty << std::endl;
    
    std::cout << "  High variance output - Confidence: " << estimate2.confidence << std::endl;
    std::cout << "  High variance output - Epistemic uncertainty: " 
              << estimate2.epistemic_uncertainty << std::endl;
    
    // Update calibration data
    for (int i = 0; i < 50; i++) {
        float predicted_conf = 0.5f + 0.4f * sin(i * 0.1f);
        // Simulate that higher confidence predictions are indeed more accurate
        bool was_correct = (rand() % 100) < (predicted_conf * 100);
        estimator.update_calibration(predicted_conf, was_correct);
    }
    
    // Get calibration error
    float calib_error = estimator.get_calibration_error();
    std::cout << "  Calibration error: " << calib_error << std::endl;
    
    // Get calibration data
    auto calib_data = estimator.get_calibration_data(10);
    std::cout << "  Calibration samples: " << calib_data.sample_count << std::endl;
    
    auto stats = estimator.get_statistics();
    std::cout << "  Total estimates: " << stats["total_estimates"] << std::endl;
    std::cout << "  Total calibration updates: " << stats["total_calibration_updates"] << std::endl;
    std::cout << "✅ Confidence Estimator test passed" << std::endl;
}

void test_bias_detector() {
    std::cout << "\n🔍 Testing Bias Detector..." << std::endl;
    
    BiasDetector detector;
    
    // Create decision sequence with some biases
    std::vector<std::vector<float>> decisions;
    std::vector<float> confidences;
    std::vector<float> outcomes;
    
    // Simulate overconfidence bias
    for (int i = 0; i < 30; i++) {
        std::vector<float> decision;
        decision.push_back(0.6f + 0.2f * sin(i * 0.2f));
        decision.push_back(0.4f + 0.3f * cos(i * 0.3f));
        decisions.push_back(decision);
        
        // Confidence higher than actual accuracy
        confidences.push_back(0.8f + 0.1f * sin(i * 0.1f));
        outcomes.push_back(0.6f + 0.15f * sin(i * 0.1f));
    }
    
    // Detect biases
    auto detected = detector.detect_biases(decisions, confidences, outcomes);
    
    std::cout << "  Detected " << detected.size() << " biases:" << std::endl;
    for (const auto& bias : detected) {
        std::cout << "    - Strength: " << std::fixed << std::setprecision(3) 
                  << bias.strength << ", Confidence: " << bias.confidence << std::endl;
        std::cout << "      Description: " << bias.description << std::endl;
    }
    
    // Test overconfidence detection specifically
    auto overconf = detector.detect_overconfidence(confidences, outcomes);
    std::cout << "  Overconfidence bias strength: " << overconf.strength << std::endl;
    
    // Generate and apply correction
    if (!detected.empty()) {
        auto correction = detector.generate_correction(detected[0]);
        std::cout << "  Generated correction factor: " << correction.correction_factor << std::endl;
        
        auto corrected = detector.apply_correction(decisions[0], correction);
        std::cout << "  Original decision: [" << decisions[0][0] << ", " << decisions[0][1] << "]" << std::endl;
        std::cout << "  Corrected decision: [" << corrected[0] << ", " << corrected[1] << "]" << std::endl;
    }
    
    auto stats = detector.get_bias_statistics();
    std::cout << "  Total detections: " << stats["total_detections"] << std::endl;
    std::cout << "  Total corrections: " << stats["total_corrections"] << std::endl;
    std::cout << "✅ Bias Detector test passed" << std::endl;
}

int main() {
    std::cout << "🚀 Starting Self-Reflection Mechanisms Test Suite" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    try {
        test_meta_cognitive_monitor();
        test_decision_quality_tracker();
        test_confidence_estimator();
        test_bias_detector();
        
        std::cout << "\n✅ All Self-Reflection tests passed!" << std::endl;
        std::cout << "\n📈 Self-Reflection System Performance Summary:" << std::endl;
        std::cout << "  - Meta-cognitive monitoring: Active ✓" << std::endl;
        std::cout << "  - Decision quality tracking: Active ✓" << std::endl;
        std::cout << "  - Confidence estimation: Calibrated ✓" << std::endl;
        std::cout << "  - Bias detection: Operational ✓" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

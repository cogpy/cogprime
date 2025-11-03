/**
 * @file system_validator.cpp
 * @brief Implementation of system-wide validation framework
 */

#include "system_validator.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>

namespace rros {

// ============================================================================
// RangeValidator Implementation
// ============================================================================

RangeValidator::RangeValidator(const std::string& name, const std::string& key,
                               float min_val, float max_val)
    : name_(name), key_(key), min_value_(min_val), max_value_(max_val) {}

ValidationResult RangeValidator::validate(
    const std::unordered_map<std::string, float>& data) {
    
    ValidationResult result;
    result.validator_name = name_;
    
    auto it = data.find(key_);
    if (it == data.end()) {
        result.passed = false;
        result.message = "Required key '" + key_ + "' not found in data";
        result.severity = 1.0f;
        return result;
    }
    
    float value = it->second;
    result.metrics[key_] = value;
    
    if (value < min_value_ || value > max_value_) {
        result.passed = false;
        result.message = key_ + " = " + std::to_string(value) + 
                        " is outside valid range [" + std::to_string(min_value_) + 
                        ", " + std::to_string(max_value_) + "]";
        result.severity = 0.8f;
    } else {
        result.passed = true;
        result.message = key_ + " is within valid range";
        result.severity = 0.0f;
    }
    
    return result;
}

std::string RangeValidator::get_name() const {
    return name_;
}

// ============================================================================
// ConsistencyValidator Implementation
// ============================================================================

ConsistencyValidator::ConsistencyValidator(
    const std::string& name,
    const std::vector<std::string>& keys,
    std::function<bool(const std::vector<float>&)> check)
    : name_(name), keys_(keys), check_func_(check) {}

ValidationResult ConsistencyValidator::validate(
    const std::unordered_map<std::string, float>& data) {
    
    ValidationResult result;
    result.validator_name = name_;
    
    std::vector<float> values;
    for (const auto& key : keys_) {
        auto it = data.find(key);
        if (it == data.end()) {
            result.passed = false;
            result.message = "Required key '" + key + "' not found";
            result.severity = 1.0f;
            return result;
        }
        values.push_back(it->second);
        result.metrics[key] = it->second;
    }
    
    result.passed = check_func_(values);
    result.message = result.passed ? "Consistency check passed" : "Consistency check failed";
    result.severity = result.passed ? 0.0f : 0.7f;
    
    return result;
}

std::string ConsistencyValidator::get_name() const {
    return name_;
}

// ============================================================================
// PerformanceValidator Implementation
// ============================================================================

PerformanceValidator::PerformanceValidator(const std::string& name,
                                         const std::string& metric,
                                         float threshold,
                                         bool higher_better)
    : name_(name), metric_key_(metric), threshold_(threshold),
      higher_is_better_(higher_better) {}

ValidationResult PerformanceValidator::validate(
    const std::unordered_map<std::string, float>& data) {
    
    ValidationResult result;
    result.validator_name = name_;
    
    auto it = data.find(metric_key_);
    if (it == data.end()) {
        result.passed = false;
        result.message = "Performance metric '" + metric_key_ + "' not found";
        result.severity = 1.0f;
        return result;
    }
    
    float value = it->second;
    result.metrics[metric_key_] = value;
    
    bool meets_requirement = higher_is_better_ ? 
        (value >= threshold_) : (value <= threshold_);
    
    result.passed = meets_requirement;
    result.message = metric_key_ + " = " + std::to_string(value) + 
                    (meets_requirement ? " meets " : " fails ") +
                    "requirement (" + 
                    (higher_is_better_ ? ">= " : "<= ") + 
                    std::to_string(threshold_) + ")";
    result.severity = meets_requirement ? 0.0f : 0.6f;
    
    return result;
}

std::string PerformanceValidator::get_name() const {
    return name_;
}

// ============================================================================
// SystemValidator Implementation
// ============================================================================

SystemValidator::SystemValidator() {
    // Add default validators for common system properties
    
    // Range validators
    add_validator(std::make_shared<RangeValidator>(
        "Relevance Range", "global_relevance", 0.0f, 1.0f));
    add_validator(std::make_shared<RangeValidator>(
        "Confidence Range", "confidence", 0.0f, 1.0f));
    add_validator(std::make_shared<RangeValidator>(
        "Attention Budget", "attention_budget", 0.0f, 10.0f));
    
    // Performance validators
    add_validator(std::make_shared<PerformanceValidator>(
        "Processing Speed", "processing_time_us", 10000.0f, false));
    add_validator(std::make_shared<PerformanceValidator>(
        "Cache Hit Rate", "cache_hit_rate", 0.4f, true));
    
    // Consistency validators
    // Note: Probability normalization validator commented out as 'uncertainty' 
    // metric is not yet implemented in the kernel. This can be enabled when
    // the kernel provides uncertainty estimates.
    /*
    add_validator(std::make_shared<ConsistencyValidator>(
        "Probability Normalization",
        std::vector<std::string>{"confidence", "uncertainty"},
        [](const std::vector<float>& vals) {
            if (vals.size() != 2) return false;
            float sum = vals[0] + vals[1];
            return std::abs(sum - 1.0f) < 0.1f; // Allow small margin
        }
    ));
    */
}

void SystemValidator::add_validator(std::shared_ptr<Validator> validator) {
    validators_.push_back(validator);
}

ValidationReport SystemValidator::validate_system(
    const std::unordered_map<std::string, float>& data) {
    
    ValidationReport report;
    report.total_validations = static_cast<int>(validators_.size());
    report.passed_validations = 0;
    report.failed_validations = 0;
    
    for (auto& validator : validators_) {
        ValidationResult result = validator->validate(data);
        report.results.push_back(result);
        
        if (result.passed) {
            report.passed_validations++;
        } else {
            report.failed_validations++;
        }
        
        // Aggregate metrics
        for (const auto& [key, value] : result.metrics) {
            report.summary_metrics[key] = value;
        }
    }
    
    report.all_passed = (report.failed_validations == 0);
    history_.push_back(report);
    
    return report;
}

ValidationReport SystemValidator::validate_continuous(
    const std::vector<std::unordered_map<std::string, float>>& data_sequence) {
    
    ValidationReport aggregate_report;
    aggregate_report.total_validations = 0;
    aggregate_report.passed_validations = 0;
    aggregate_report.failed_validations = 0;
    
    for (const auto& data : data_sequence) {
        ValidationReport cycle_report = validate_system(data);
        
        aggregate_report.total_validations += cycle_report.total_validations;
        aggregate_report.passed_validations += cycle_report.passed_validations;
        aggregate_report.failed_validations += cycle_report.failed_validations;
        
        // Collect all results
        aggregate_report.results.insert(
            aggregate_report.results.end(),
            cycle_report.results.begin(),
            cycle_report.results.end()
        );
    }
    
    aggregate_report.all_passed = (aggregate_report.failed_validations == 0);
    
    return aggregate_report;
}

const std::vector<ValidationReport>& SystemValidator::get_history() const {
    return history_;
}

void SystemValidator::clear_history() {
    history_.clear();
}

std::string SystemValidator::generate_report_summary(const ValidationReport& report) const {
    std::ostringstream oss;
    
    oss << "=== Validation Report ===\n";
    oss << "Total Validations: " << report.total_validations << "\n";
    oss << "Passed: " << report.passed_validations << "\n";
    oss << "Failed: " << report.failed_validations << "\n";
    oss << "Overall Status: " << (report.all_passed ? "PASSED" : "FAILED") << "\n\n";
    
    if (report.failed_validations > 0) {
        oss << "Failed Validations:\n";
        for (const auto& result : report.results) {
            if (!result.passed) {
                oss << "  - " << result.validator_name << ": " 
                    << result.message << "\n";
            }
        }
    }
    
    return oss.str();
}

namespace {
    // Validation severity thresholds
    constexpr float SEVERITY_CRITICAL = 0.8f;
    constexpr float SEVERITY_WARNING = 0.5f;
    constexpr float SEVERITY_INFO = 0.0f;
}

bool SystemValidator::is_system_healthy(
    const std::unordered_map<std::string, float>& data) {
    
    ValidationReport report = validate_system(data);
    
    // System is healthy if all critical validators pass
    for (const auto& result : report.results) {
        if (!result.passed && result.severity >= SEVERITY_CRITICAL) {
            return false; // Critical validation failed
        }
    }
    
    return true;
}

// ============================================================================
// IntegrationValidator Implementation
// ============================================================================

void IntegrationValidator::register_module(
    const std::string& name,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) {
    
    ModuleInterface module;
    module.name = name;
    module.input_keys = inputs;
    module.output_keys = outputs;
    modules_.push_back(module);
}

ValidationResult IntegrationValidator::validate_module_integration(
    const std::unordered_map<std::string, float>& input_data,
    const std::unordered_map<std::string, float>& output_data) {
    
    ValidationResult result;
    result.validator_name = "Module Integration";
    result.passed = true;
    
    // Check that all registered modules have their required data
    for (const auto& module : modules_) {
        // Verify inputs
        for (const auto& input_key : module.input_keys) {
            if (input_data.find(input_key) == input_data.end()) {
                result.passed = false;
                result.message += "Module '" + module.name + 
                                "' missing input '" + input_key + "'; ";
            }
        }
        
        // Verify outputs
        for (const auto& output_key : module.output_keys) {
            if (output_data.find(output_key) == output_data.end()) {
                result.passed = false;
                result.message += "Module '" + module.name + 
                                "' missing output '" + output_key + "'; ";
            }
        }
    }
    
    if (result.passed) {
        result.message = "All module interfaces validated successfully";
        result.severity = 0.0f;
    } else {
        result.severity = 0.9f;
    }
    
    return result;
}

bool IntegrationValidator::check_data_integrity(
    const std::unordered_map<std::string, float>& input_data,
    const std::unordered_map<std::string, float>& output_data) {
    
    // Check for NaN or infinite values in output
    for (const auto& [key, value] : output_data) {
        if (std::isnan(value) || std::isinf(value)) {
            return false;
        }
    }
    
    // Verify output has at least as much information as input
    // (or transformation is intentional)
    return output_data.size() > 0;
}

// ============================================================================
// ScenarioValidator Implementation
// ============================================================================

void ScenarioValidator::add_scenario(const Scenario& scenario) {
    scenarios_.push_back(scenario);
}

ValidationResult ScenarioValidator::validate_scenario(
    const std::string& scenario_name,
    const std::vector<std::unordered_map<std::string, float>>& results) {
    
    ValidationResult result;
    result.validator_name = "Scenario: " + scenario_name;
    
    // Find scenario
    auto it = std::find_if(scenarios_.begin(), scenarios_.end(),
        [&scenario_name](const Scenario& s) { return s.name == scenario_name; });
    
    if (it == scenarios_.end()) {
        result.passed = false;
        result.message = "Scenario not found: " + scenario_name;
        result.severity = 1.0f;
        return result;
    }
    
    // Validate using scenario's success criteria
    result.passed = it->success_criteria(results);
    result.message = result.passed ? 
        "Scenario completed successfully" : 
        "Scenario failed to meet success criteria";
    result.severity = result.passed ? 0.0f : 0.8f;
    
    return result;
}

std::vector<ValidationResult> ScenarioValidator::validate_all_scenarios(
    std::function<std::unordered_map<std::string, float>(const std::vector<float>&)> processor) {
    
    std::vector<ValidationResult> results;
    
    for (const auto& scenario : scenarios_) {
        std::vector<std::unordered_map<std::string, float>> scenario_results;
        
        // Process each input in the scenario
        for (const auto& input : scenario.input_sequence) {
            auto output = processor(input);
            scenario_results.push_back(output);
        }
        
        // Validate scenario
        ValidationResult result = validate_scenario(scenario.name, scenario_results);
        results.push_back(result);
    }
    
    return results;
}

} // namespace rros

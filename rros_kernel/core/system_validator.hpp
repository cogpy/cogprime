/**
 * @file system_validator.hpp
 * @brief System-wide validation and verification framework
 * 
 * Provides comprehensive validation procedures for ensuring correct
 * operation of the RROS cognitive kernel under diverse conditions.
 */

#ifndef RROS_SYSTEM_VALIDATOR_HPP
#define RROS_SYSTEM_VALIDATOR_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>

namespace rros {

/**
 * @brief Validation result structure
 */
struct ValidationResult {
    bool passed;
    std::string validator_name;
    std::string message;
    float severity;  // 0.0 = info, 0.5 = warning, 1.0 = critical
    std::unordered_map<std::string, float> metrics;
};

/**
 * @brief Validation report aggregating multiple results
 */
struct ValidationReport {
    bool all_passed;
    int total_validations;
    int passed_validations;
    int failed_validations;
    std::vector<ValidationResult> results;
    std::unordered_map<std::string, float> summary_metrics;
};

/**
 * @brief Base validator interface
 */
class Validator {
public:
    virtual ~Validator() = default;
    
    /**
     * @brief Execute validation check
     * @param data Validation input data
     * @return Validation result
     */
    virtual ValidationResult validate(const std::unordered_map<std::string, float>& data) = 0;
    
    /**
     * @brief Get validator name
     */
    virtual std::string get_name() const = 0;
};

/**
 * @brief Validates data ranges and bounds
 */
class RangeValidator : public Validator {
private:
    std::string name_;
    std::string key_;
    float min_value_;
    float max_value_;
    
public:
    RangeValidator(const std::string& name, const std::string& key, 
                   float min_val, float max_val);
    
    ValidationResult validate(const std::unordered_map<std::string, float>& data) override;
    std::string get_name() const override;
};

/**
 * @brief Validates consistency between related metrics
 */
class ConsistencyValidator : public Validator {
private:
    std::string name_;
    std::vector<std::string> keys_;
    std::function<bool(const std::vector<float>&)> check_func_;
    
public:
    ConsistencyValidator(const std::string& name,
                        const std::vector<std::string>& keys,
                        std::function<bool(const std::vector<float>&)> check);
    
    ValidationResult validate(const std::unordered_map<std::string, float>& data) override;
    std::string get_name() const override;
};

/**
 * @brief Validates performance requirements
 */
class PerformanceValidator : public Validator {
private:
    std::string name_;
    std::string metric_key_;
    float threshold_;
    bool higher_is_better_;
    
public:
    PerformanceValidator(const std::string& name, const std::string& metric,
                        float threshold, bool higher_better = true);
    
    ValidationResult validate(const std::unordered_map<std::string, float>& data) override;
    std::string get_name() const override;
};

/**
 * @brief System-wide validation coordinator
 */
class SystemValidator {
private:
    std::vector<std::shared_ptr<Validator>> validators_;
    std::vector<ValidationReport> history_;
    
public:
    /**
     * @brief Constructor with default validators
     */
    SystemValidator();
    
    /**
     * @brief Add custom validator
     */
    void add_validator(std::shared_ptr<Validator> validator);
    
    /**
     * @brief Run all validators on provided data
     * @param data System metrics and state data
     * @return Comprehensive validation report
     */
    ValidationReport validate_system(const std::unordered_map<std::string, float>& data);
    
    /**
     * @brief Run continuous validation across multiple cycles
     * @param data_sequence Sequence of system states to validate
     * @return Aggregated validation report
     */
    ValidationReport validate_continuous(
        const std::vector<std::unordered_map<std::string, float>>& data_sequence);
    
    /**
     * @brief Get validation history
     */
    const std::vector<ValidationReport>& get_history() const;
    
    /**
     * @brief Clear validation history
     */
    void clear_history();
    
    /**
     * @brief Generate validation report summary
     */
    std::string generate_report_summary(const ValidationReport& report) const;
    
    /**
     * @brief Check if system meets all critical requirements
     */
    bool is_system_healthy(const std::unordered_map<std::string, float>& data);
};

/**
 * @brief Integration validator for module communication
 */
class IntegrationValidator {
private:
    struct ModuleInterface {
        std::string name;
        std::vector<std::string> input_keys;
        std::vector<std::string> output_keys;
    };
    
    std::vector<ModuleInterface> modules_;
    
public:
    /**
     * @brief Register module interface
     */
    void register_module(const std::string& name,
                        const std::vector<std::string>& inputs,
                        const std::vector<std::string>& outputs);
    
    /**
     * @brief Validate data flow between modules
     * @param input_data Input to module pipeline
     * @param output_data Output from module pipeline
     * @return Validation result
     */
    ValidationResult validate_module_integration(
        const std::unordered_map<std::string, float>& input_data,
        const std::unordered_map<std::string, float>& output_data);
    
    /**
     * @brief Check for data loss in pipeline
     */
    bool check_data_integrity(
        const std::unordered_map<std::string, float>& input_data,
        const std::unordered_map<std::string, float>& output_data);
};

/**
 * @brief End-to-end scenario validator
 */
class ScenarioValidator {
public:
    struct Scenario {
        std::string name;
        std::vector<std::vector<float>> input_sequence;
        std::function<bool(const std::vector<std::unordered_map<std::string, float>>&)> success_criteria;
    };
    
private:
    std::vector<Scenario> scenarios_;
    
public:
    /**
     * @brief Add validation scenario
     */
    void add_scenario(const Scenario& scenario);
    
    /**
     * @brief Run scenario validation
     * @param scenario_name Name of scenario to run
     * @param results Output results from cognitive processing
     * @return Validation result
     */
    ValidationResult validate_scenario(
        const std::string& scenario_name,
        const std::vector<std::unordered_map<std::string, float>>& results);
    
    /**
     * @brief Run all scenarios
     */
    std::vector<ValidationResult> validate_all_scenarios(
        std::function<std::unordered_map<std::string, float>(const std::vector<float>&)> processor);
};

} // namespace rros

#endif // RROS_SYSTEM_VALIDATOR_HPP

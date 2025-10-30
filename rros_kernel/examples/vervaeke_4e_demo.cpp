#include "../core/rros_kernel.hpp"
#include "../core/vervaeke_4e.hpp"
#include <iostream>
#include <iomanip>

using namespace rros;
using namespace rros::vervaeke;

/**
 * @brief Example integration of Vervaeke 4E Framework with RROS Kernel
 * 
 * Demonstrates how the 4E cognition components enhance the existing
 * RROS cognitive kernel with embodied, embedded, enacted, and extended
 * processing capabilities.
 */

void demonstrate_full_integration() {
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "FULL 4E FRAMEWORK INTEGRATION" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    const int NUM_DEMO_CYCLES = 3;
    
    std::unordered_map<std::string, float> config;
    config["body_schema_lr"] = 0.1f;
    config["adaptation_strength"] = 0.7f;
    config["novelty_weight"] = 0.3f;
    
    Vervaeke4EFramework framework(config);
    
    std::cout << "Running complete 4E cognitive cycles..." << std::endl;
    
    for (int cycle = 0; cycle < NUM_DEMO_CYCLES; ++cycle) {
        std::cout << "\n--- Cycle " << (cycle + 1) << " ---" << std::endl;
        
        std::vector<float> sensory = {
            0.2f + cycle * 0.1f, 
            0.4f + cycle * 0.1f, 
            0.6f + cycle * 0.1f
        };
        
        std::vector<float> environment = {
            0.3f + cycle * 0.05f,
            0.5f + cycle * 0.05f,
            0.7f + cycle * 0.05f
        };
        
        std::vector<float> goals = {
            0.5f + cycle * 0.1f,
            0.7f + cycle * 0.1f,
            0.9f - cycle * 0.1f
        };
        
        auto state = framework.process_4e_cycle(sensory, environment, goals);
        
        std::cout << "Overall coherence: " << std::fixed << std::setprecision(3)
                  << state.overall_coherence << std::endl;
        std::cout << "Wisdom measure: " << state.wisdom_measure << std::endl;
        std::cout << "Meaning connectivity: " << state.meaning_connectivity << std::endl;
    }
    
    float wisdom = framework.compute_wisdom_measure();
    float meaning = framework.assess_meaning_connectivity();
    
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "Final Wisdom Measure: " << std::fixed << std::setprecision(3) 
              << wisdom << std::endl;
    std::cout << "Final Meaning Connectivity: " << meaning << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  VERVAEKE 4E COGNITION FRAMEWORK                       ║" << std::endl;
    std::cout << "║  Integration Demo                                      ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\n";
    
    demonstrate_full_integration();
    
    std::cout << "\n";
    std::cout << "✅ Embodied: Sensorimotor integration" << std::endl;
    std::cout << "✅ Embedded: Context awareness" << std::endl;
    std::cout << "✅ Enacted: Active perception" << std::endl;
    std::cout << "✅ Extended: Cognitive extension" << std::endl;
    std::cout << "✅ Salience: Relevance navigation" << std::endl;
    std::cout << "✅ Perspectival: Viewpoint integration" << std::endl;
    std::cout << "✅ Participatory: Agent-world coupling" << std::endl;
    std::cout << "✅ Wisdom: Relevance realization" << std::endl;
    std::cout << "✅ Meaning: Mind-world reconnection" << std::endl;
    std::cout << "\n";
    
    return 0;
}

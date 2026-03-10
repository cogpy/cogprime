// test_metahuman_dna.cpp — Tests for MetaHuman DNA Cognitive Bridge
// Compile: g++ -std=c++11 -I../include -o test_metahuman_dna test_metahuman_dna.cpp && ./test_metahuman_dna
#include "cog/prime/metahuman_dna.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace cog::prime;

void test_facs_state() {
    FACSState facs;
    // All AUs should be zero
    for (uint8_t i = 0; i < AU_COUNT; ++i) {
        assert(facs.get(static_cast<ActionUnit>(i)) == 0.0f);
    }
    // Set and get
    facs.set(AU12, 0.8f);
    assert(std::abs(facs.get(AU12) - 0.8f) < 0.001f);
    // Clamping
    facs.set(AU6, 1.5f);
    assert(facs.get(AU6) == 1.0f);
    facs.set(AU4, -0.5f);
    assert(facs.get(AU4) == 0.0f);
    // Add
    facs.set(AU12, 0.5f);
    facs.add(AU12, 0.3f);
    assert(std::abs(facs.get(AU12) - 0.8f) < 0.001f);
    // Reset
    facs.reset();
    assert(facs.get(AU12) == 0.0f);
    std::cout << "  [PASS] FACSState\n";
}

void test_morph_targets() {
    FACSState facs;
    facs.set(AU12, 0.8f);
    facs.set(AU6, 0.6f);
    auto targets = facs.to_morph_targets();
    assert(targets["CTRL_mouth_cornerPull"] == 0.8f);
    assert(targets["CTRL_cheek_raise"] == 0.6f);
    std::cout << "  [PASS] MorphTargets\n";
}

void test_lorenz_attractor() {
    LorenzAttractor la;
    // Step 1000 times
    float nx, ny, nz;
    for (int i = 0; i < 1000; ++i) {
        la.step(nx, ny, nz);
        assert(!std::isnan(nx) && !std::isnan(ny) && !std::isnan(nz));
    }
    assert(la.is_healthy());
    // Lyapunov should be positive after many steps
    for (int i = 0; i < 4000; ++i) la.step(nx, ny, nz);
    assert(la.lyapunov_exponent() > 0.0);
    std::cout << "  [PASS] LorenzAttractor (Lyapunov=" << la.lyapunov_exponent() << ")\n";
}

void test_lorenz_micro_expressions() {
    LorenzAttractor la;
    FACSState facs;
    for (int i = 0; i < 100; ++i) { float x,y,z; la.step(x,y,z); }
    la.apply_micro_expressions(facs);
    auto targets = facs.to_morph_targets();
    assert(!targets.empty());
    std::cout << "  [PASS] LorenzMicroExpressions\n";
}

void test_endocrine_mapper() {
    EndocrineExpressionMapper mapper;
    FACSState facs;
    float conc[HORMONE_COUNT] = {};
    conc[DopaminePhasic] = 0.9f;
    mapper.map_to_facs(conc, facs);
    assert(facs.get(AU12) > 0.5f); // Smile
    assert(facs.get(AU6) > 0.3f);  // Cheek raise
    std::cout << "  [PASS] EndocrineMapper (dopamine→smile)\n";
}

void test_cortisol_stress() {
    EndocrineExpressionMapper mapper;
    FACSState facs;
    float conc[HORMONE_COUNT] = {};
    conc[Cortisol] = 0.8f;
    mapper.map_to_facs(conc, facs);
    assert(facs.get(AU4) > 0.5f);  // Brow lowerer
    assert(facs.get(AU15) > 0.2f); // Lip corner depress
    std::cout << "  [PASS] CortisolStress\n";
}

void test_cognitive_mode_preset() {
    FACSState facs;
    cognitive_mode_preset(MODE_SOCIAL, facs);
    assert(facs.get(AU6) > 0.5f);  // Cheek raise
    assert(facs.get(AU12) > 0.4f); // Smile
    std::cout << "  [PASS] CognitiveModePreset\n";
}

void test_aesthetic_parameters() {
    AestheticParameters ap;
    FACSState facs;
    facs.set(AU4, 0.8f); // High stress
    ap.confidence_posture = 0.9f;
    ap.apply_to_facs(facs);
    assert(facs.get(AU4) < 0.8f);  // Reduced by confidence
    assert(facs.get(AU17) > 0.1f); // Chin raise added
    std::cout << "  [PASS] AestheticParameters\n";
}

void test_dna_bridge() {
    DNACognitiveBridge bridge;
    auto frame = bridge.update_simple(0.7f, 0.5f, 0.3f);
    assert(frame.frame == 1);
    assert(!frame.morph_targets.empty());
    assert(!frame.material_params.empty());
    // Multiple frames
    for (int i = 0; i < 99; ++i) {
        frame = bridge.update_simple(0.5f, 0.5f, 0.3f);
    }
    assert(frame.frame == 100);
    assert(bridge.is_healthy());
    std::cout << "  [PASS] DNACognitiveBridge (100 frames)\n";
}

void test_dna_bridge_with_endocrine() {
    DNACognitiveBridge bridge;
    float conc[HORMONE_COUNT] = {};
    conc[DopaminePhasic] = 0.8f;
    conc[Oxytocin] = 0.6f;
    auto frame = bridge.update(conc, MODE_SOCIAL, 0.3f, 0.7f, 0.5f);
    assert(frame.morph_targets.count("CTRL_mouth_cornerPull") > 0);
    assert(frame.morph_targets["CTRL_mouth_cornerPull"] > 0.1f);
    std::cout << "  [PASS] DNACognitiveBridge with endocrine\n";
}

int main() {
    std::cout << "MetaHuman DNA Cognitive Bridge — C++ Tests\n";
    std::cout << "==========================================\n";
    test_facs_state();
    test_morph_targets();
    test_lorenz_attractor();
    test_lorenz_micro_expressions();
    test_endocrine_mapper();
    test_cortisol_stress();
    test_cognitive_mode_preset();
    test_aesthetic_parameters();
    test_dna_bridge();
    test_dna_bridge_with_endocrine();
    std::cout << "==========================================\n";
    std::cout << "All tests passed!\n";
    return 0;
}

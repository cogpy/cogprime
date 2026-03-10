// cog/prime/metahuman_dna.hpp — MetaHuman DNA Cognitive Bridge
// FACS Action Units → MetaHuman CTRL_ Morph Targets
// Endocrine-driven expression, Lorenz chaotic micro-expressions, SuperHotGirl aesthetics
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
//
// Architecture (echo-angel composition):
//   Endocrine State ──┐
//                     ├→ FACS AUs → MetaHuman CTRL_ Morph Targets
//   Cognitive State ──┘      ↕                    ↕
//                     Chaotic Dynamics      Aesthetic Parameters
//                     (Lorenz Attractor)    (SuperHotGirl)
//
#ifndef COG_PRIME_METAHUMAN_DNA_HPP
#define COG_PRIME_METAHUMAN_DNA_HPP

#include "../core/core.hpp"
#include <cmath>
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <unordered_map>

namespace cog { namespace prime {

// ─────────────────────────────────────────────────────────────────────────────
// FACS Action Unit Enumeration (Ekman & Friesen)
// ─────────────────────────────────────────────────────────────────────────────
enum ActionUnit : uint8_t {
    AU1  = 0,  // Inner Brow Raise
    AU2  = 1,  // Outer Brow Raise
    AU4  = 2,  // Brow Lowerer
    AU5  = 3,  // Upper Lid Raise
    AU6  = 4,  // Cheek Raise
    AU7  = 5,  // Lid Tightener
    AU9  = 6,  // Nose Wrinkle
    AU10 = 7,  // Upper Lip Raise
    AU12 = 8,  // Lip Corner Pull (Smile)
    AU14 = 9,  // Dimpler
    AU15 = 10, // Lip Corner Depress
    AU17 = 11, // Chin Raise
    AU20 = 12, // Lip Stretch
    AU23 = 13, // Lip Tightener
    AU25 = 14, // Lips Part
    AU26 = 15, // Jaw Drop
    AU28 = 16, // Lip Suck
    AU43 = 17, // Eyes Closed
    AU45 = 18, // Blink
    AU46 = 19, // Wink
    AU_COUNT = 20
};

inline const char* au_name(ActionUnit au) {
    static const char* names[] = {
        "Inner Brow Raise", "Outer Brow Raise", "Brow Lowerer",
        "Upper Lid Raise", "Cheek Raise", "Lid Tightener",
        "Nose Wrinkle", "Upper Lip Raise", "Lip Corner Pull",
        "Dimpler", "Lip Corner Depress", "Chin Raise",
        "Lip Stretch", "Lip Tightener", "Lips Part",
        "Jaw Drop", "Lip Suck", "Eyes Closed", "Blink", "Wink"
    };
    return (au < AU_COUNT) ? names[au] : "Unknown";
}

// MetaHuman CTRL_ morph target names per AU
inline const char* au_morph_target(ActionUnit au) {
    static const char* targets[] = {
        "CTRL_brow_inner_UP", "CTRL_brow_outer_UP", "CTRL_brow_down",
        "CTRL_eye_upperLid_UP", "CTRL_cheek_raise", "CTRL_eye_squint",
        "CTRL_nose_wrinkle", "CTRL_mouth_upperLip_UP", "CTRL_mouth_cornerPull",
        "CTRL_mouth_dimple", "CTRL_mouth_cornerDepress", "CTRL_chin_raise",
        "CTRL_mouth_stretch", "CTRL_mouth_tighten", "CTRL_mouth_lipsPart",
        "CTRL_jaw_open", "CTRL_mouth_lipSuck", "CTRL_eye_blink",
        "CTRL_eye_blink", "CTRL_eye_blink_L"
    };
    return (au < AU_COUNT) ? targets[au] : "";
}

// ─────────────────────────────────────────────────────────────────────────────
// FACSState — Thread-safe AU activation storage
// ─────────────────────────────────────────────────────────────────────────────
class FACSState {
public:
    FACSState() { reset(); }

    void set(ActionUnit au, float v) {
        if (au < AU_COUNT) act_[au] = clamp01(v);
    }
    float get(ActionUnit au) const {
        return (au < AU_COUNT) ? act_[au] : 0.0f;
    }
    void add(ActionUnit au, float v) {
        if (au < AU_COUNT) act_[au] = clamp01(act_[au] + v);
    }
    void reset() { act_.fill(0.0f); }

    // Convert to MetaHuman morph target map
    std::unordered_map<std::string, float> to_morph_targets() const {
        std::unordered_map<std::string, float> targets;
        for (uint8_t i = 0; i < AU_COUNT; ++i) {
            if (act_[i] > 0.001f) {
                std::string key = au_morph_target(static_cast<ActionUnit>(i));
                targets[key] += act_[i];
                if (targets[key] > 1.0f) targets[key] = 1.0f;
            }
        }
        return targets;
    }

    const std::array<float, AU_COUNT>& raw() const { return act_; }

private:
    std::array<float, AU_COUNT> act_;
    static float clamp01(float v) { return (v < 0.0f) ? 0.0f : (v > 1.0f) ? 1.0f : v; }
};

// ─────────────────────────────────────────────────────────────────────────────
// LorenzAttractor — Chaotic micro-expression dynamics (RK4 integration)
// ─────────────────────────────────────────────────────────────────────────────
class LorenzAttractor {
public:
    float sigma, rho, beta, dt, chaos_intensity;
    float x, y, z;

    LorenzAttractor()
        : sigma(10.0f), rho(28.0f), beta(8.0f/3.0f), dt(0.01f),
          chaos_intensity(0.15f), x(1.0f), y(1.0f), z(1.0f),
          px_(1.0001f), py_(1.0f), pz_(1.0f),
          lyap_sum_(0.0), lyap_steps_(0) {}

    // Step one timestep (RK4), return normalized (x,y,z) in ~[-1,1]
    void step(float& nx, float& ny, float& nz) {
        rk4(x, y, z);
        rk4(px_, py_, pz_);
        update_lyapunov();
        nx = x / 20.0f;
        ny = y / 30.0f;
        nz = (z - 25.0f) / 25.0f;
    }

    // Step with variable delta time
    void step_delta(float delta_time, float& nx, float& ny, float& nz) {
        int steps = static_cast<int>(std::ceil(delta_time / dt));
        if (steps < 1) steps = 1;
        if (steps > 1000) steps = 1000;
        for (int i = 0; i < steps; ++i) {
            rk4(x, y, z);
            rk4(px_, py_, pz_);
        }
        update_lyapunov();
        nx = x / 20.0f;
        ny = y / 30.0f;
        nz = (z - 25.0f) / 25.0f;
    }

    // Apply chaotic micro-expressions to FACS state
    void apply_micro_expressions(FACSState& facs) {
        float nx, ny, nz;
        step(nx, ny, nz);
        float ci = chaos_intensity;
        facs.add(AU1,  nx * ci * 0.3f);
        facs.add(AU7,  ny * ci * 0.2f);
        facs.add(AU12, nz * ci * 0.15f);
        facs.add(AU9,  (nx + ny) * ci * 0.1f);
        facs.add(AU26, nz * ci * 0.05f);
    }

    double lyapunov_exponent() const {
        return (lyap_steps_ > 0) ? lyap_sum_ / lyap_steps_ / dt : 0.0;
    }

    bool is_healthy() const {
        return !std::isnan(x) && !std::isinf(x) &&
               !std::isnan(y) && !std::isinf(y) &&
               !std::isnan(z) && !std::isinf(z) &&
               std::abs(x) < 100.0f && std::abs(y) < 100.0f && std::abs(z) < 100.0f;
    }

    void reset() {
        x = y = z = 1.0f;
        px_ = 1.0001f; py_ = pz_ = 1.0f;
        lyap_sum_ = 0; lyap_steps_ = 0;
    }

private:
    float px_, py_, pz_;
    double lyap_sum_;
    int lyap_steps_;

    void derivatives(float ax, float ay, float az, float& dx, float& dy, float& dz) const {
        dx = sigma * (ay - ax);
        dy = ax * (rho - az) - ay;
        dz = ax * ay - beta * az;
    }

    void rk4(float& ax, float& ay, float& az) {
        float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
        derivatives(ax, ay, az, k1x, k1y, k1z);
        derivatives(ax + 0.5f*dt*k1x, ay + 0.5f*dt*k1y, az + 0.5f*dt*k1z, k2x, k2y, k2z);
        derivatives(ax + 0.5f*dt*k2x, ay + 0.5f*dt*k2y, az + 0.5f*dt*k2z, k3x, k3y, k3z);
        derivatives(ax + dt*k3x, ay + dt*k3y, az + dt*k3z, k4x, k4y, k4z);
        ax += (dt / 6.0f) * (k1x + 2*k2x + 2*k3x + k4x);
        ay += (dt / 6.0f) * (k1y + 2*k2y + 2*k3y + k4y);
        az += (dt / 6.0f) * (k1z + 2*k2z + 2*k3z + k4z);
    }

    void update_lyapunov() {
        float dx = px_ - x, dy = py_ - y, dz = pz_ - z;
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist > 1e-10f) {
            lyap_sum_ += std::log(dist / 1e-4);
            lyap_steps_++;
            float scale = 1e-4f / dist;
            px_ = x + dx * scale;
            py_ = y + dy * scale;
            pz_ = z + dz * scale;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Hormone Enumeration (mirrors Go endocrine system)
// ─────────────────────────────────────────────────────────────────────────────
enum Hormone : uint8_t {
    Cortisol = 0,
    DopaminePhasic,
    DopamineTonic,
    Serotonin,
    Norepinephrine,
    Oxytocin,
    Melatonin,
    ThyroxineT4,
    CytokineIL6,
    Anandamide,
    HORMONE_COUNT
};

// ─────────────────────────────────────────────────────────────────────────────
// EndocrineExpressionMapper — Hormone → FACS mapping
// ─────────────────────────────────────────────────────────────────────────────
class EndocrineExpressionMapper {
public:
    struct AUMapping { ActionUnit au; float scale; };

    EndocrineExpressionMapper() {
        // Cortisol → worry/stress
        maps_[Cortisol]       = {{AU4, 0.8f}, {AU1, 0.5f}, {AU15, 0.4f}};
        // Dopamine (phasic) → reward smile
        maps_[DopaminePhasic] = {{AU12, 0.9f}, {AU6, 0.7f}};
        // Dopamine (tonic) → baseline contentment
        maps_[DopamineTonic]  = {{AU12, 0.3f}};
        // Serotonin → warm contentment
        maps_[Serotonin]      = {{AU6, 0.4f}, {AU12, 0.3f}};
        // Norepinephrine → alertness
        maps_[Norepinephrine] = {{AU5, 0.8f}, {AU7, 0.5f}, {AU20, 0.3f}};
        // Oxytocin → social warmth
        maps_[Oxytocin]       = {{AU6, 0.6f}, {AU12, 0.5f}, {AU25, 0.3f}};
        // Melatonin → drowsiness
        maps_[Melatonin]      = {{AU43, 0.7f}, {AU7, 0.4f}};
        // Cytokine IL-6 → sickness
        maps_[CytokineIL6]    = {{AU4, 0.5f}, {AU10, 0.4f}};
        // Anandamide → bliss
        maps_[Anandamide]     = {{AU6, 0.5f}, {AU25, 0.3f}};
    }

    void map_to_facs(const float conc[HORMONE_COUNT], FACSState& facs) const {
        for (uint8_t h = 0; h < HORMONE_COUNT; ++h) {
            auto it = maps_.find(static_cast<Hormone>(h));
            if (it == maps_.end()) continue;
            for (const auto& m : it->second) {
                facs.add(m.au, conc[h] * m.scale);
            }
        }
    }

private:
    std::unordered_map<uint8_t, std::vector<AUMapping>> maps_;
};

// ─────────────────────────────────────────────────────────────────────────────
// CognitiveMode — Emergent behavioral modes
// ─────────────────────────────────────────────────────────────────────────────
enum CognitiveMode : uint8_t {
    MODE_EXPLORE = 0, MODE_EXPLOIT, MODE_FIGHT, MODE_FLIGHT,
    MODE_FREEZE, MODE_FLOW, MODE_SOCIAL, MODE_REST,
    MODE_CREATIVE, MODE_ANALYTICAL, MODE_COUNT
};

inline void cognitive_mode_preset(CognitiveMode mode, FACSState& facs) {
    switch (mode) {
    case MODE_EXPLORE:    facs.add(AU5, 0.5f); facs.add(AU25, 0.4f); facs.add(AU1, 0.3f); break;
    case MODE_EXPLOIT:    facs.add(AU4, 0.4f); facs.add(AU7, 0.3f); break;
    case MODE_FIGHT:      facs.add(AU1, 0.6f); facs.add(AU4, 0.7f); facs.add(AU5, 0.5f); facs.add(AU20, 0.6f); break;
    case MODE_FLIGHT:     facs.add(AU1, 0.7f); facs.add(AU5, 0.6f); facs.add(AU20, 0.4f); break;
    case MODE_FREEZE:     facs.add(AU5, 0.8f); facs.add(AU7, 0.6f); facs.add(AU20, 0.3f); break;
    case MODE_FLOW:       facs.add(AU6, 0.4f); facs.add(AU12, 0.3f); facs.add(AU7, 0.2f); break;
    case MODE_SOCIAL:     facs.add(AU6, 0.7f); facs.add(AU12, 0.6f); facs.add(AU25, 0.3f); break;
    case MODE_REST:       facs.add(AU43, 0.5f); facs.add(AU7, 0.3f); break;
    case MODE_CREATIVE:   facs.add(AU1, 0.4f); facs.add(AU2, 0.3f); facs.add(AU6, 0.5f); facs.add(AU12, 0.4f); break;
    case MODE_ANALYTICAL: facs.add(AU4, 0.5f); facs.add(AU7, 0.4f); break;
    default: break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AestheticParameters — SuperHotGirl aesthetic modulation
// ─────────────────────────────────────────────────────────────────────────────
struct AestheticParameters {
    float confidence_posture;
    float charisma;
    float eye_sparkle;
    float graceful_movement;
    float emissive_glow;

    AestheticParameters()
        : confidence_posture(0.7f), charisma(0.6f), eye_sparkle(0.5f),
          graceful_movement(0.6f), emissive_glow(0.15f) {}

    void apply_to_facs(FACSState& facs) const {
        if (confidence_posture > 0.5f) {
            float boost = (confidence_posture - 0.5f) * 2.0f;
            facs.add(AU17, boost * 0.3f);
            facs.add(AU2, boost * 0.2f);
            float cur4 = facs.get(AU4);
            facs.set(AU4, cur4 * (1.0f - boost * 0.3f));
            float cur15 = facs.get(AU15);
            facs.set(AU15, cur15 * (1.0f - boost * 0.4f));
        }
        if (charisma > 0.3f) {
            float boost = (charisma - 0.3f) / 0.7f;
            facs.add(AU12, boost * 0.2f);
            facs.add(AU6, boost * 0.15f);
        }
        if (eye_sparkle > 0.3f) {
            float boost = (eye_sparkle - 0.3f) / 0.7f;
            facs.add(AU5, boost * 0.1f);
        }
    }

    std::unordered_map<std::string, float> material_parameters() const {
        return {
            {"EyeSparkleIntensity", eye_sparkle},
            {"SkinGlowIntensity", emissive_glow},
            {"IrisSpecular", eye_sparkle * 2.0f},
            {"SkinSSSBoost", emissive_glow * 1.5f},
            {"MotionSmoothing", graceful_movement}
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ExpressionFrame — One frame of pipeline output
// ─────────────────────────────────────────────────────────────────────────────
struct ExpressionFrame {
    uint64_t frame;
    std::unordered_map<std::string, float> morph_targets;
    std::unordered_map<std::string, float> material_params;
    double lyapunov_exponent;
    CognitiveMode cognitive_mode;
};

// ─────────────────────────────────────────────────────────────────────────────
// DNACognitiveBridge — Main orchestrator
// ─────────────────────────────────────────────────────────────────────────────
class DNACognitiveBridge {
public:
    FACSState facs;
    LorenzAttractor attractor;
    AestheticParameters aesthetic;
    EndocrineExpressionMapper mapper;
    bool enable_chaos;
    bool enable_aesthetic;
    float smoothing_factor;

    DNACognitiveBridge()
        : enable_chaos(true), enable_aesthetic(true),
          smoothing_factor(0.3f), frame_count_(0) {}

    // Full pipeline update
    ExpressionFrame update(
        const float hormone_conc[HORMONE_COUNT],
        CognitiveMode mode,
        float cognitive_load,
        float valence,
        float arousal,
        float delta_time = 0.016f)
    {
        frame_count_++;
        facs.reset();

        // 1. Endocrine → FACS
        mapper.map_to_facs(hormone_conc, facs);

        // 2. Cognitive → FACS
        cognitive_mode_preset(mode, facs);
        facs.add(AU4, cognitive_load * 0.6f);
        facs.add(AU7, cognitive_load * 0.4f);

        // 3. Valence/Arousal → FACS
        if (valence > 0) {
            facs.add(AU6, valence * 0.6f);
            facs.add(AU12, valence * 0.7f);
        } else {
            facs.add(AU15, -valence * 0.5f);
            facs.add(AU4, -valence * 0.3f);
        }
        facs.add(AU5, arousal * 0.5f);
        facs.add(AU25, arousal * 0.3f);

        // 4. Chaotic micro-expressions
        if (enable_chaos) {
            float nx, ny, nz;
            attractor.step_delta(delta_time, nx, ny, nz);
            float ci = attractor.chaos_intensity;
            facs.add(AU1, nx * ci * 0.3f);
            facs.add(AU7, ny * ci * 0.2f);
            facs.add(AU12, nz * ci * 0.15f);
        }

        // 5. Aesthetic bias
        if (enable_aesthetic) {
            aesthetic.apply_to_facs(facs);
        }

        // 6. Convert to morph targets with smoothing
        auto targets = facs.to_morph_targets();
        if (smoothing_factor > 0.0f && !last_targets_.empty()) {
            for (auto& kv : targets) {
                auto it = last_targets_.find(kv.first);
                if (it != last_targets_.end()) {
                    kv.second = it->second * smoothing_factor +
                                kv.second * (1.0f - smoothing_factor);
                }
            }
        }
        last_targets_ = targets;

        // 7. Material parameters
        auto mat = aesthetic.material_parameters();
        mat["EyeSparkleIntensity"] += hormone_conc[DopaminePhasic] * 0.3f;
        mat["SkinGlowIntensity"] += hormone_conc[Serotonin] * 0.1f;

        ExpressionFrame frame;
        frame.frame = frame_count_;
        frame.morph_targets = targets;
        frame.material_params = mat;
        frame.lyapunov_exponent = attractor.lyapunov_exponent();
        frame.cognitive_mode = mode;
        return frame;
    }

    // Simplified update (no endocrine system)
    ExpressionFrame update_simple(float valence, float arousal, float load) {
        float conc[HORMONE_COUNT] = {};
        return update(conc, MODE_EXPLORE, load, valence, arousal);
    }

    uint64_t frame_count() const { return frame_count_; }
    bool is_healthy() const { return attractor.is_healthy(); }

private:
    uint64_t frame_count_;
    std::unordered_map<std::string, float> last_targets_;
};

}} // namespace cog::prime

#endif // COG_PRIME_METAHUMAN_DNA_HPP

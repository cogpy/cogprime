// cog/prime/prime.hpp — Integrated AGI Architecture (SiliconSage / CogPrime)
// AtomSpace, Pattern Matching, Cognitive Cycle, Ontogenesis, Memory Systems
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_PRIME_HPP
#define COG_PRIME_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <algorithm>
#include <random>
#include <sstream>
#include <cassert>

namespace cog { namespace prime {

// ─────────────────────────────────────────────────────────────────────────────
// Thought — Unit of cognitive processing
// ─────────────────────────────────────────────────────────────────────────────
struct Thought {
    uint32_t id;
    std::string content;
    float salience;      // How important/relevant (0..1)
    float valence;       // Positive/negative affect (-1..1)
    float arousal;       // Activation level (0..1)
    uint64_t timestamp;

    Thought() : id(0), salience(0), valence(0), arousal(0), timestamp(0) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// Action — Output of action selection
// ─────────────────────────────────────────────────────────────────────────────
struct Action {
    uint32_t id;
    std::string name;
    float urgency;
    float expected_reward;
    std::vector<Handle> parameters; // AtomSpace handles

    Action() : id(0), urgency(0), expected_reward(0) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// CognitiveState — Global workspace state
// ─────────────────────────────────────────────────────────────────────────────
struct CognitiveState {
    std::vector<Thought> attentional_focus;
    std::vector<Action> action_queue;
    float global_activation;
    float curiosity;
    float confidence;
    uint64_t cycle_count;

    CognitiveState() : global_activation(0.5f), curiosity(0.5f),
                       confidence(0.5f), cycle_count(0) {}

    void add_thought(const Thought& t) {
        attentional_focus.push_back(t);
        // Sort by salience, keep top N
        std::sort(attentional_focus.begin(), attentional_focus.end(),
                  [](const Thought& a, const Thought& b) {
                      return a.salience > b.salience;
                  });
        if (attentional_focus.size() > 32) {
            attentional_focus.resize(32);
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// PatternMatcher — Graph pattern matching on AtomSpace
// ─────────────────────────────────────────────────────────────────────────────
class PatternMatcher {
public:
    using Bindings = std::unordered_map<Handle, Handle>; // Variable → Atom

    explicit PatternMatcher(const AtomSpace& as) : as_(as) {}

    // Match a pattern against the AtomSpace
    // Pattern is a link with VariableNodes as wildcards
    std::vector<Bindings> match(Handle pattern_handle) const {
        std::vector<Bindings> results;
        const Atom* pattern = as_.get_atom(pattern_handle);
        if (!pattern || !pattern->is_link()) return results;

        // Collect variables in the pattern
        std::vector<Handle> variables;
        collect_variables(pattern_handle, variables);

        // Try all possible bindings
        if (variables.empty()) {
            // No variables: check if pattern exists literally
            auto matches = as_.pattern_match(pattern->type, pattern->outgoing);
            for (auto h : matches) {
                results.push_back(Bindings());
            }
        } else {
            // Generate candidate bindings
            Bindings bindings;
            match_recursive(pattern_handle, bindings, results);
        }

        return results;
    }

    // Simple query: find all links of type with specific node in position
    std::vector<Handle> query(AtomType link_type, Handle node_handle, size_t position) const {
        std::vector<Handle> results;
        auto links = as_.get_by_type(link_type);
        for (auto h : links) {
            const Atom* a = as_.get_atom(h);
            if (a && position < a->outgoing.size() && a->outgoing[position] == node_handle) {
                results.push_back(h);
            }
        }
        return results;
    }

    // Inheritance query: find all X such that (InheritanceLink X target)
    std::vector<Handle> get_supertypes(Handle node) const {
        std::vector<Handle> result;
        auto links = as_.get_incoming(node);
        for (auto lh : links) {
            const Atom* link = as_.get_atom(lh);
            if (link && link->type == AtomType::INHERITANCE_LINK &&
                link->outgoing.size() == 2 && link->outgoing[0] == node) {
                result.push_back(link->outgoing[1]);
            }
        }
        return result;
    }

    std::vector<Handle> get_subtypes(Handle node) const {
        std::vector<Handle> result;
        auto links = as_.get_incoming(node);
        for (auto lh : links) {
            const Atom* link = as_.get_atom(lh);
            if (link && link->type == AtomType::INHERITANCE_LINK &&
                link->outgoing.size() == 2 && link->outgoing[1] == node) {
                result.push_back(link->outgoing[0]);
            }
        }
        return result;
    }

private:
    const AtomSpace& as_;

    void collect_variables(Handle h, std::vector<Handle>& vars) const {
        const Atom* a = as_.get_atom(h);
        if (!a) return;
        if (a->type == AtomType::VARIABLE_NODE) {
            if (std::find(vars.begin(), vars.end(), h) == vars.end()) {
                vars.push_back(h);
            }
        }
        for (auto oh : a->outgoing) {
            collect_variables(oh, vars);
        }
    }

    void match_recursive(Handle pattern, Bindings& bindings,
                         std::vector<Bindings>& results) const {
        const Atom* pat = as_.get_atom(pattern);
        if (!pat) return;

        if (pat->type == AtomType::VARIABLE_NODE) {
            // Try binding to every atom
            as_.foreach_atom([&](const Atom& candidate) {
                auto it = bindings.find(pattern);
                if (it != bindings.end()) {
                    if (it->second == candidate.handle) {
                        // Already bound to this: consistent
                    }
                    return; // Already bound to something else
                }
                bindings[pattern] = candidate.handle;
                results.push_back(bindings);
                bindings.erase(pattern);
            });
            return;
        }

        if (pat->is_link()) {
            // Find all links of same type
            auto candidates = as_.get_by_type(pat->type);
            for (auto ch : candidates) {
                const Atom* cand = as_.get_atom(ch);
                if (!cand || cand->outgoing.size() != pat->outgoing.size()) continue;

                Bindings trial = bindings;
                bool ok = true;
                for (size_t i = 0; i < pat->outgoing.size() && ok; ++i) {
                    const Atom* sub_pat = as_.get_atom(pat->outgoing[i]);
                    if (!sub_pat) { ok = false; continue; }
                    if (sub_pat->type == AtomType::VARIABLE_NODE) {
                        auto it = trial.find(pat->outgoing[i]);
                        if (it != trial.end()) {
                            if (it->second != cand->outgoing[i]) ok = false;
                        } else {
                            trial[pat->outgoing[i]] = cand->outgoing[i];
                        }
                    } else {
                        if (pat->outgoing[i] != cand->outgoing[i]) ok = false;
                    }
                }
                if (ok) results.push_back(trial);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Memory Systems
// ─────────────────────────────────────────────────────────────────────────────

// Sensory Memory: short-lived, high-capacity buffer
class SensoryMemory {
public:
    explicit SensoryMemory(size_t capacity = 256) : capacity_(capacity) {}

    void store(const Thought& t) {
        buffer_.push_back(t);
        if (buffer_.size() > capacity_) {
            buffer_.erase(buffer_.begin());
        }
    }

    // Decay: reduce salience of all items
    void decay(float rate = 0.1f) {
        for (auto& t : buffer_) {
            t.salience *= (1.0f - rate);
        }
        // Remove items below threshold
        buffer_.erase(std::remove_if(buffer_.begin(), buffer_.end(),
            [](const Thought& t) { return t.salience < 0.01f; }), buffer_.end());
    }

    const std::vector<Thought>& buffer() const { return buffer_; }
    size_t size() const { return buffer_.size(); }

private:
    size_t capacity_;
    std::vector<Thought> buffer_;
};

// Semantic Memory: long-term factual knowledge (backed by AtomSpace)
class SemanticMemory {
public:
    explicit SemanticMemory(AtomSpace& as) : as_(as) {}

    Handle store_fact(const std::string& subject, const std::string& predicate,
                      const std::string& object, float confidence = 0.9f) {
        Handle s = as_.add_node(AtomType::CONCEPT_NODE, subject);
        Handle p = as_.add_node(AtomType::PREDICATE_NODE, predicate);
        Handle o = as_.add_node(AtomType::CONCEPT_NODE, object);
        Handle list = as_.add_link(AtomType::LIST_LINK, {s, o});
        return as_.add_link(AtomType::EVALUATION_LINK, {p, list},
                           TruthValue(1.0f, confidence));
    }

    Handle store_inheritance(const std::string& child, const std::string& parent,
                             float strength = 1.0f, float confidence = 0.9f) {
        Handle c = as_.add_node(AtomType::CONCEPT_NODE, child);
        Handle p = as_.add_node(AtomType::CONCEPT_NODE, parent);
        return as_.add_link(AtomType::INHERITANCE_LINK, {c, p},
                           TruthValue(strength, confidence));
    }

    AtomSpace& atomspace() { return as_; }

private:
    AtomSpace& as_;
};

// Episodic Memory: autobiographical events with temporal context
class EpisodicMemory {
public:
    struct Episode {
        uint32_t id;
        uint64_t timestamp;
        std::vector<Thought> thoughts;
        std::vector<Action> actions;
        float emotional_valence;
        float importance;

        Episode() : id(0), timestamp(0), emotional_valence(0), importance(0) {}
    };

    void store(const Episode& ep) {
        episodes_.push_back(ep);
    }

    // Retrieve episodes by recency
    std::vector<const Episode*> recent(size_t n) const {
        std::vector<const Episode*> result;
        size_t start = (episodes_.size() > n) ? episodes_.size() - n : 0;
        for (size_t i = start; i < episodes_.size(); ++i) {
            result.push_back(&episodes_[i]);
        }
        return result;
    }

    // Retrieve episodes by emotional similarity
    std::vector<const Episode*> by_valence(float target_valence, float threshold = 0.3f) const {
        std::vector<const Episode*> result;
        for (auto& ep : episodes_) {
            if (std::fabs(ep.emotional_valence - target_valence) < threshold) {
                result.push_back(&ep);
            }
        }
        return result;
    }

    size_t size() const { return episodes_.size(); }

private:
    std::vector<Episode> episodes_;
};

// Procedural Memory: learned action sequences
class ProceduralMemory {
public:
    struct Procedure {
        uint32_t id;
        std::string name;
        std::vector<Action> steps;
        float success_rate;
        uint32_t execution_count;

        Procedure() : id(0), success_rate(0), execution_count(0) {}
    };

    uint32_t store(const std::string& name, const std::vector<Action>& steps) {
        Procedure p;
        p.id = static_cast<uint32_t>(procedures_.size() + 1);
        p.name = name;
        p.steps = steps;
        p.success_rate = 0.5f;
        p.execution_count = 0;
        procedures_.push_back(p);
        return p.id;
    }

    void update_success(uint32_t id, bool success) {
        for (auto& p : procedures_) {
            if (p.id == id) {
                p.execution_count++;
                float alpha = 1.0f / static_cast<float>(p.execution_count);
                p.success_rate = (1.0f - alpha) * p.success_rate + alpha * (success ? 1.0f : 0.0f);
                break;
            }
        }
    }

    const Procedure* find(const std::string& name) const {
        for (auto& p : procedures_) {
            if (p.name == name) return &p;
        }
        return nullptr;
    }

    // Get best procedure for a goal
    const Procedure* best_for_goal(const std::string& goal_prefix) const {
        const Procedure* best = nullptr;
        float best_rate = -1;
        for (auto& p : procedures_) {
            if (p.name.find(goal_prefix) != std::string::npos &&
                p.success_rate > best_rate) {
                best = &p;
                best_rate = p.success_rate;
            }
        }
        return best;
    }

private:
    std::vector<Procedure> procedures_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Cognitive Modules — Perceive, Reason, Act
// ─────────────────────────────────────────────────────────────────────────────

class PerceptionModule {
public:
    virtual ~PerceptionModule() {}
    virtual std::vector<Thought> perceive(const CognitiveState& state) = 0;
};

class ReasoningModule {
public:
    virtual ~ReasoningModule() {}
    virtual std::vector<Thought> reason(const CognitiveState& state,
                                         const AtomSpace& knowledge) = 0;
};

class ActionSelectionModule {
public:
    virtual ~ActionSelectionModule() {}
    virtual std::vector<Action> select(const CognitiveState& state) = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Default Reasoning: PLN-based forward chainer
// ─────────────────────────────────────────────────────────────────────────────
class PLNReasoner : public ReasoningModule {
public:
    std::vector<Thought> reason(const CognitiveState& state,
                                 const AtomSpace& knowledge) override {
        std::vector<Thought> conclusions;
        // Forward chain: for each InheritanceLink A→B and B→C, derive A→C
        auto inh_links = knowledge.get_by_type(AtomType::INHERITANCE_LINK);
        for (size_t i = 0; i < inh_links.size(); ++i) {
            const Atom* ab = knowledge.get_atom(inh_links[i]);
            if (!ab || ab->outgoing.size() != 2) continue;
            Handle b = ab->outgoing[1];
            for (size_t j = 0; j < inh_links.size(); ++j) {
                if (i == j) continue;
                const Atom* bc = knowledge.get_atom(inh_links[j]);
                if (!bc || bc->outgoing.size() != 2) continue;
                if (bc->outgoing[0] != b) continue;
                // Found A→B→C chain
                TruthValue derived = TruthValue::deduction(ab->tv, bc->tv);
                if (derived.confidence > 0.1f) {
                    const Atom* a_atom = knowledge.get_atom(ab->outgoing[0]);
                    const Atom* c_atom = knowledge.get_atom(bc->outgoing[1]);
                    if (a_atom && c_atom) {
                        Thought t;
                        t.content = a_atom->name + " → " + c_atom->name;
                        t.salience = derived.confidence;
                        t.valence = derived.strength - 0.5f;
                        conclusions.push_back(t);
                    }
                }
            }
        }
        return conclusions;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Ontogenesis — Self-generating kernel evolution
// ─────────────────────────────────────────────────────────────────────────────
struct GripMetrics {
    float relevance;
    float coherence;
    float participation;
    float transparency;

    GripMetrics() : relevance(0), coherence(0), participation(0), transparency(0) {}

    float score() const {
        return (relevance + coherence + participation + transparency) / 4.0f;
    }
};

struct GeneratedKernel {
    uint32_t id;
    uint32_t generation;
    std::vector<float> genome;  // Encoded parameters
    GripMetrics fitness;

    GeneratedKernel() : id(0), generation(0) {}
};

class OntogenesisEngine {
public:
    struct Config {
        size_t population_size;
        size_t genome_length;
        float mutation_rate;
        float crossover_rate;
        uint32_t seed;

        Config() : population_size(32), genome_length(64),
                   mutation_rate(0.05f), crossover_rate(0.7f), seed(42) {}
    };

    explicit OntogenesisEngine(const Config& cfg = Config())
        : cfg_(cfg), rng_(cfg.seed), generation_(0)
    {
        // Initialize population
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < cfg.population_size; ++i) {
            GeneratedKernel k;
            k.id = static_cast<uint32_t>(i);
            k.generation = 0;
            k.genome.resize(cfg.genome_length);
            for (auto& g : k.genome) g = dist(rng_);
            population_.push_back(k);
        }
    }

    // Evaluate fitness of all kernels using a provided function
    void evaluate(std::function<GripMetrics(const GeneratedKernel&)> fitness_fn) {
        for (auto& k : population_) {
            k.fitness = fitness_fn(k);
        }
    }

    // Evolve one generation
    void evolve() {
        ++generation_;

        // Sort by fitness
        std::sort(population_.begin(), population_.end(),
                  [](const GeneratedKernel& a, const GeneratedKernel& b) {
                      return a.fitness.score() > b.fitness.score();
                  });

        std::vector<GeneratedKernel> next_gen;
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        std::uniform_int_distribution<size_t> parent_dist(0, cfg_.population_size / 2);
        std::normal_distribution<float> mutation(0.0f, 0.1f);

        // Elitism: keep top 10%
        size_t elite = std::max<size_t>(1, cfg_.population_size / 10);
        for (size_t i = 0; i < elite; ++i) {
            auto k = population_[i];
            k.generation = generation_;
            next_gen.push_back(k);
        }

        // Fill rest with crossover + mutation
        while (next_gen.size() < cfg_.population_size) {
            size_t p1 = parent_dist(rng_);
            size_t p2 = parent_dist(rng_);
            GeneratedKernel child;
            child.id = static_cast<uint32_t>(next_gen.size());
            child.generation = generation_;
            child.genome.resize(cfg_.genome_length);

            // Crossover
            if (prob(rng_) < cfg_.crossover_rate) {
                std::uniform_int_distribution<size_t> xp(0, cfg_.genome_length - 1);
                size_t crosspoint = xp(rng_);
                for (size_t i = 0; i < cfg_.genome_length; ++i) {
                    child.genome[i] = (i < crosspoint) ?
                        population_[p1].genome[i] : population_[p2].genome[i];
                }
            } else {
                child.genome = population_[p1].genome;
            }

            // Mutation
            for (auto& g : child.genome) {
                if (prob(rng_) < cfg_.mutation_rate) {
                    g += mutation(rng_);
                    g = std::max(-1.0f, std::min(1.0f, g));
                }
            }

            next_gen.push_back(child);
        }

        population_ = next_gen;
    }

    const GeneratedKernel& best() const { return population_.front(); }
    uint32_t generation() const { return generation_; }
    const std::vector<GeneratedKernel>& population() const { return population_; }

private:
    Config cfg_;
    std::mt19937 rng_;
    uint32_t generation_;
    std::vector<GeneratedKernel> population_;
};

// ─────────────────────────────────────────────────────────────────────────────
// CognitiveCycle — Main perceive → reason → act loop
// ─────────────────────────────────────────────────────────────────────────────
class CognitiveCycle {
public:
    CognitiveCycle(AtomSpace& as)
        : as_(as), semantic_(as), cycle_(0) {}

    void set_perception(std::shared_ptr<PerceptionModule> m) { perception_ = m; }
    void set_reasoning(std::shared_ptr<ReasoningModule> m) { reasoning_ = m; }
    void set_action(std::shared_ptr<ActionSelectionModule> m) { action_ = m; }

    // Run one cognitive cycle
    CognitiveState step() {
        ++cycle_;
        state_.cycle_count = cycle_;

        // 1. Perceive
        if (perception_) {
            auto percepts = perception_->perceive(state_);
            for (auto& t : percepts) {
                t.timestamp = cycle_;
                sensory_.store(t);
                state_.add_thought(t);
            }
        }

        // 2. Reason
        if (reasoning_) {
            auto conclusions = reasoning_->reason(state_, as_);
            for (auto& t : conclusions) {
                t.timestamp = cycle_;
                state_.add_thought(t);
            }
        }

        // 3. Act
        if (action_) {
            auto actions = action_->select(state_);
            state_.action_queue = actions;
        }

        // 4. Memory maintenance
        sensory_.decay(0.05f);

        // 5. ECAN: decay attention values
        as_.foreach_atom([](const Atom& a) {
            // Note: const iteration; actual mutation would need mutable access
        });

        return state_;
    }

    CognitiveState& state() { return state_; }
    SemanticMemory& semantic() { return semantic_; }
    EpisodicMemory& episodic() { return episodic_; }
    ProceduralMemory& procedural() { return procedural_; }
    uint64_t cycle_count() const { return cycle_; }

private:
    AtomSpace& as_;
    CognitiveState state_;
    SensoryMemory sensory_;
    SemanticMemory semantic_;
    EpisodicMemory episodic_;
    ProceduralMemory procedural_;
    std::shared_ptr<PerceptionModule> perception_;
    std::shared_ptr<ReasoningModule> reasoning_;
    std::shared_ptr<ActionSelectionModule> action_;
    uint64_t cycle_;
};

}} // namespace cog::prime

#endif // COG_PRIME_HPP

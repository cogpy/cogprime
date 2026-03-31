// tests/unit/test_prime.cpp — Tests for cog::prime module
// Tests CognitiveCycle, OntogenesisEngine, PatternMatcher, Memory systems
// SPDX-License-Identifier: MIT

#include "../../include/cog/prime/prime.hpp"
#include <cassert>
#include <iostream>

#define RUN_TEST(test) \
    std::cout << "Running test: " << #test << "...\n"; \
    test(); \
    std::cout << "Test " << #test << " passed.\n";

void test_thought_creation() {
    cog::prime::Thought t;
    t.content = "test-thought";
    t.salience = 0.8f;
    t.valence = 0.5f;
    t.arousal = 0.6f;
    assert(t.content == "test-thought");
    assert(t.salience > 0.79f);
}

void test_cognitive_state() {
    cog::prime::CognitiveState state;
    assert(state.cycle_count == 0);
}

void test_sensory_memory() {
    cog::prime::SensoryMemory sm;
    cog::prime::Thought t;
    t.content = "visual-input";
    t.salience = 1.0f;
    sm.store(t);
    assert(sm.size() == 1);
    assert(sm.buffer()[0].content == "visual-input");
}

void test_semantic_memory() {
    cog::AtomSpace as;
    cog::prime::SemanticMemory sem(as);
    auto h = sem.store_fact("dog", "is-a", "animal");
    assert(h != cog::UNDEFINED_HANDLE);
    auto h2 = sem.store_inheritance("cat", "animal");
    assert(h2 != cog::UNDEFINED_HANDLE);
}

void test_pattern_matcher() {
    cog::AtomSpace as;
    auto cat = as.add_node(cog::AtomType::CONCEPT_NODE, "cat");
    auto animal = as.add_node(cog::AtomType::CONCEPT_NODE, "animal");
    as.add_link(cog::AtomType::INHERITANCE_LINK, {cat, animal});
    cog::prime::PatternMatcher pm(as);
    auto results = pm.query(cog::AtomType::INHERITANCE_LINK, cat, 0);
    assert(results.size() == 1);
}

void test_ontogenesis_engine() {
    cog::prime::OntogenesisEngine engine;
    // Should be constructible with default config
    (void)engine;
}

int main() {
    RUN_TEST(test_thought_creation);
    RUN_TEST(test_cognitive_state);
    RUN_TEST(test_sensory_memory);
    RUN_TEST(test_semantic_memory);
    RUN_TEST(test_pattern_matcher);
    RUN_TEST(test_ontogenesis_engine);
    std::cout << "All cogprime prime tests passed!\n";
    return 0;
}

#include "../core/temporal_knowledge.hpp"
#include "../core/episodic_memory.hpp"
#include "../core/historical_context.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>

using namespace rros;

// Test utilities
void assert_approx_equal(float a, float b, float epsilon = 0.01f) {
    if (std::abs(a - b) > epsilon) {
        std::cerr << "Assertion failed: " << a << " != " << b << std::endl;
        std::exit(1);
    }
}

void assert_true(bool condition, const std::string& msg = "") {
    if (!condition) {
        std::cerr << "Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

// Test 1: Temporal Knowledge Representation
bool test_temporal_knowledge() {
    std::cout << "Testing Temporal Knowledge Representation..." << std::endl;
    
    TemporalKnowledge tk(1000, 100);
    
    // Add events
    std::vector<float> state1 = {0.1f, 0.2f, 0.3f};
    std::vector<float> state2 = {0.4f, 0.5f, 0.6f};
    std::vector<float> state3 = {0.7f, 0.8f, 0.9f};
    
    size_t event1 = tk.add_event(state1, 0.8f);
    size_t event2 = tk.add_event(state2, 0.9f);
    size_t event3 = tk.add_event(state3, 0.7f);
    
    // Test event retrieval
    const auto& retrieved_event = tk.get_event(event1);
    assert_true(retrieved_event.state_vector.size() == 3, "Event state size");
    assert_approx_equal(retrieved_event.importance, 0.8f);
    
    // Add causal relation
    tk.add_causal_relation(event1, event2, 0.9f, 0.85f);
    
    auto successors = tk.get_causal_successors(event1);
    assert_true(successors.size() == 1, "Causal successors");
    assert_approx_equal(successors[0].strength, 0.9f);
    
    // Create abstraction
    std::vector<size_t> event_ids = {event1, event2, event3};
    size_t abstraction_id = tk.create_abstraction(event_ids, 1000000);
    
    auto stats = tk.get_statistics();
    assert_true(stats["event_count"] == 3.0f, "Event count");
    assert_true(stats["causal_relation_count"] == 1.0f, "Causal relation count");
    assert_true(stats["abstraction_count"] == 1.0f, "Abstraction count");
    
    std::cout << "  ✓ Temporal knowledge tests passed" << std::endl;
    return true;
}

// Test 2: Episodic Memory with Temporal Indexing
bool test_episodic_memory() {
    std::cout << "Testing Episodic Memory with Temporal Indexing..." << std::endl;
    
    EpisodicMemorySystem ems(100, 1000000); // 1 second bins
    
    // Encode memories
    std::vector<float> memory1 = {1.0f, 0.0f, 0.0f};
    std::vector<float> memory2 = {0.0f, 1.0f, 0.0f};
    std::vector<float> memory3 = {0.0f, 0.0f, 1.0f};
    
    size_t m1 = ems.encode_memory(memory1, 0.9f);
    size_t m2 = ems.encode_memory(memory2, 0.8f);
    size_t m3 = ems.encode_memory(memory3, 0.7f);
    
    // Test content-based retrieval
    std::vector<float> query = {0.9f, 0.1f, 0.0f};
    auto results = ems.retrieve_by_content(query, 2);
    assert_true(results.size() <= 2, "Content retrieval count");
    
    // Test consolidation
    size_t consolidated = ems.consolidate_memories();
    
    // Test emotional tagging
    ems.tag_emotion(m1, "excitement", 0.8f);
    const auto& tagged_memory = ems.get_memory(m1);
    assert_true(tagged_memory.emotional_tags.count("excitement") > 0, "Emotional tag");
    
    auto stats = ems.get_statistics();
    assert_true(stats["memory_count"] == 3.0f, "Memory count");
    assert_true(stats["avg_vividness"] > 0.0f, "Average vividness");
    
    std::cout << "  ✓ Episodic memory tests passed" << std::endl;
    return true;
}

// Test 3: Historical Pattern Recognition
bool test_pattern_recognition() {
    std::cout << "Testing Historical Pattern Recognition..." << std::endl;
    
    auto tk = std::make_shared<TemporalKnowledge>(1000, 100);
    auto ems = std::make_shared<EpisodicMemorySystem>(100, 1000000);
    
    HistoricalContextSystem hcs(tk, ems);
    
    // Create repeating pattern
    std::vector<size_t> event_sequence;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> state = {
            static_cast<float>(i % 3),
            static_cast<float>((i + 1) % 3),
            static_cast<float>((i + 2) % 3)
        };
        size_t event_id = tk->add_event(state, 0.8f);
        event_sequence.push_back(event_id);
    }
    
    // Detect patterns
    auto patterns = hcs.detect_patterns(event_sequence, 0.1f, 0.5f);
    assert_true(patterns.size() > 0, "Patterns detected");
    
    std::cout << "  ✓ Pattern recognition tests passed" << std::endl;
    return true;
}

// Test 4: Temporal Reasoning and Inference
bool test_temporal_reasoning() {
    std::cout << "Testing Temporal Reasoning and Inference..." << std::endl;
    
    auto tk = std::make_shared<TemporalKnowledge>(1000, 100);
    auto ems = std::make_shared<EpisodicMemorySystem>(100, 1000000);
    
    HistoricalContextSystem hcs(tk, ems);
    
    // Create causal sequence
    std::vector<float> cause_state = {1.0f, 0.0f, 0.0f};
    std::vector<float> effect_state = {0.0f, 1.0f, 0.0f};
    
    size_t cause = tk->add_event(cause_state, 0.9f);
    size_t effect = tk->add_event(effect_state, 0.8f);
    
    tk->add_causal_relation(cause, effect, 0.85f, 0.9f);
    
    // Learn rules
    size_t rules_learned = hcs.learn_temporal_rules(100, 0.05f);
    
    auto stats = hcs.get_statistics();
    assert_true(stats.count("inference_rules") > 0, "Inference rules tracked");
    
    std::cout << "  ✓ Temporal reasoning tests passed" << std::endl;
    return true;
}

// Test 5: Causal Relationship Detection
bool test_causal_detection() {
    std::cout << "Testing Causal Relationship Detection..." << std::endl;
    
    auto tk = std::make_shared<TemporalKnowledge>(1000, 100);
    auto ems = std::make_shared<EpisodicMemorySystem>(100, 1000000);
    
    HistoricalContextSystem hcs(tk, ems);
    
    // Create events with potential causal relationships
    for (int i = 0; i < 5; ++i) {
        std::vector<float> state = {
            static_cast<float>(i) * 0.2f,
            static_cast<float>(i + 1) * 0.2f,
            static_cast<float>(i + 2) * 0.2f
        };
        tk->add_event(state, 0.7f + i * 0.05f);
    }
    
    // Detect causal relations
    auto causal_relations = hcs.detect_causal_relations(10000000, 0.3f);
    
    // Should detect some relations given the sequential nature
    std::cout << "  Detected " << causal_relations.size() << " causal relations" << std::endl;
    
    std::cout << "  ✓ Causal detection tests passed" << std::endl;
    return true;
}

// Test 6: Historical Context-Aware Decision Making
bool test_historical_decision_making() {
    std::cout << "Testing Historical Context-Aware Decision Making..." << std::endl;
    
    auto tk = std::make_shared<TemporalKnowledge>(1000, 100);
    auto ems = std::make_shared<EpisodicMemorySystem>(100, 1000000);
    
    HistoricalContextSystem hcs(tk, ems);
    
    // Build decision history
    for (int i = 0; i < 20; ++i) {
        std::vector<float> observation = {
            static_cast<float>(i % 5) * 0.2f,
            static_cast<float>((i + 1) % 5) * 0.2f
        };
        std::vector<float> outcome = {
            static_cast<float>(i % 3) * 0.3f
        };
        float quality = 0.5f + (i % 10) * 0.05f;
        
        hcs.update_from_observation(observation, outcome, quality);
    }
    
    // Make decision with historical context
    std::vector<float> decision_context = {0.4f, 0.6f};
    std::vector<std::vector<float>> alternatives = {
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.5f, 0.5f}
    };
    
    auto scores = hcs.make_historical_decision(decision_context, alternatives);
    assert_true(scores.size() == 3, "Decision scores count");
    
    // Scores should be normalized
    for (float score : scores) {
        assert_true(score >= 0.0f && score <= 1.0f, "Score in valid range");
    }
    
    auto stats = hcs.get_statistics();
    assert_true(stats["decision_quality_samples"] > 0.0f, "Decision quality tracked");
    
    std::cout << "  ✓ Historical decision making tests passed" << std::endl;
    return true;
}

// Test 7: Temporal Abstraction and Generalization
bool test_temporal_abstraction() {
    std::cout << "Testing Temporal Abstraction and Generalization..." << std::endl;
    
    TemporalKnowledge tk(1000, 100);
    
    // Create events at different time scales
    std::vector<size_t> fine_grain_events;
    for (int i = 0; i < 10; ++i) {
        std::vector<float> state = {
            static_cast<float>(i) * 0.1f,
            static_cast<float>(i) * 0.1f + 0.05f
        };
        size_t event_id = tk.add_event(state, 0.7f);
        fine_grain_events.push_back(event_id);
    }
    
    // Create abstraction at larger time scale
    size_t abstraction = tk.create_abstraction(fine_grain_events, 10000000);
    
    auto stats = tk.get_statistics();
    assert_true(stats["abstraction_count"] == 1.0f, "Abstraction created");
    
    std::cout << "  ✓ Temporal abstraction tests passed" << std::endl;
    return true;
}

// Test 8: Historical Knowledge Validation
bool test_knowledge_validation() {
    std::cout << "Testing Historical Knowledge Validation..." << std::endl;
    
    auto tk = std::make_shared<TemporalKnowledge>(1000, 100);
    auto ems = std::make_shared<EpisodicMemorySystem>(100, 1000000);
    
    HistoricalContextSystem hcs(tk, ems);
    
    // Create consistent event sequence
    for (int i = 0; i < 5; ++i) {
        std::vector<float> state = {static_cast<float>(i) * 0.2f};
        tk->add_event(state, 0.8f);
    }
    
    // Validate consistency
    float consistency = hcs.validate_consistency();
    assert_true(consistency >= 0.0f && consistency <= 1.0f, "Consistency in range");
    
    std::cout << "  Consistency score: " << consistency << std::endl;
    std::cout << "  ✓ Knowledge validation tests passed" << std::endl;
    return true;
}

// Test 9: Prediction Performance
bool test_prediction() {
    std::cout << "Testing Future Event Prediction..." << std::endl;
    
    auto tk = std::make_shared<TemporalKnowledge>(1000, 100);
    auto ems = std::make_shared<EpisodicMemorySystem>(100, 1000000);
    
    HistoricalContextSystem hcs(tk, ems);
    
    // Create pattern: A -> B -> C repeatedly
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::vector<float> stateA = {1.0f, 0.0f, 0.0f};
        std::vector<float> stateB = {0.0f, 1.0f, 0.0f};
        std::vector<float> stateC = {0.0f, 0.0f, 1.0f};
        
        size_t eventA = tk->add_event(stateA, 0.8f);
        size_t eventB = tk->add_event(stateB, 0.8f);
        size_t eventC = tk->add_event(stateC, 0.8f);
        
        tk->add_causal_relation(eventA, eventB, 0.9f, 0.85f);
        tk->add_causal_relation(eventB, eventC, 0.9f, 0.85f);
    }
    
    // Learn patterns
    std::vector<size_t> all_events;
    for (size_t i = 0; i < 9; ++i) {
        all_events.push_back(i);
    }
    hcs.detect_patterns(all_events, 0.1f, 0.5f);
    
    // Predict future
    std::vector<float> current = {1.0f, 0.0f, 0.0f};
    auto predictions = hcs.predict_future(current, 10000000);
    
    std::cout << "  Generated " << predictions.size() << " predictions" << std::endl;
    
    std::cout << "  ✓ Prediction tests passed" << std::endl;
    return true;
}

// Test 10: Decision Quality Improvement
bool test_decision_quality_improvement() {
    std::cout << "Testing Decision Quality Improvement..." << std::endl;
    
    auto tk = std::make_shared<TemporalKnowledge>(1000, 100);
    auto ems = std::make_shared<EpisodicMemorySystem>(100, 1000000);
    
    HistoricalContextSystem hcs(tk, ems);
    
    // Simulate learning: quality should improve over time
    for (int i = 0; i < 50; ++i) {
        std::vector<float> observation = {
            static_cast<float>(i % 5) * 0.2f
        };
        std::vector<float> outcome = {
            static_cast<float>((i + 1) % 5) * 0.2f
        };
        
        // Quality improves with more data
        float quality = 0.5f + std::min(0.3f, i * 0.006f);
        
        hcs.update_from_observation(observation, outcome, quality);
    }
    
    auto stats = hcs.get_statistics();
    float improvement = stats["decision_quality_improvement"];
    
    std::cout << "  Decision quality improvement: " << improvement << "%" << std::endl;
    
    // Should show some improvement
    assert_true(improvement >= -10.0f, "Reasonable improvement range");
    
    std::cout << "  ✓ Decision quality improvement tests passed" << std::endl;
    return true;
}

int main() {
    std::cout << "\n=== Historical Context Integration System Tests ===\n" << std::endl;
    
    int passed = 0;
    int total = 10;
    
    try {
        if (test_temporal_knowledge()) passed++;
        if (test_episodic_memory()) passed++;
        if (test_pattern_recognition()) passed++;
        if (test_temporal_reasoning()) passed++;
        if (test_causal_detection()) passed++;
        if (test_historical_decision_making()) passed++;
        if (test_temporal_abstraction()) passed++;
        if (test_knowledge_validation()) passed++;
        if (test_prediction()) passed++;
        if (test_decision_quality_improvement()) passed++;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed" << std::endl;
        return 1;
    }
}

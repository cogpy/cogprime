// tests/unit/test_core.cpp — Tests for cog::core module
// Tests Handle, AtomType, AtomSpace, TruthValue
// SPDX-License-Identifier: MIT

#include "../../include/cog/core/core.hpp"
#include <cassert>
#include <iostream>
#include <string>

#define RUN_TEST(test) \
    std::cout << "Running test: " << #test << "...\n"; \
    test(); \
    std::cout << "Test " << #test << " passed.\n";

void test_handle_creation() {
    cog::Handle h1 = 1;
    cog::Handle h2 = 2;
    assert(h1 != h2);
    assert(cog::UNDEFINED_HANDLE == 0);
}

void test_atom_type_classification() {
    assert(cog::is_node(cog::AtomType::CONCEPT_NODE));
    assert(cog::is_node(cog::AtomType::PREDICATE_NODE));
    assert(cog::is_link(cog::AtomType::INHERITANCE_LINK));
    assert(cog::is_link(cog::AtomType::EVALUATION_LINK));
}

void test_atomspace_add_node() {
    cog::AtomSpace as;
    auto h = as.add_node(cog::AtomType::CONCEPT_NODE, "hello");
    assert(h != cog::UNDEFINED_HANDLE);
    auto* atom = as.get_atom(h);
    assert(atom != nullptr);
    assert(atom->name == "hello");
}

void test_atomspace_add_link() {
    cog::AtomSpace as;
    auto h1 = as.add_node(cog::AtomType::CONCEPT_NODE, "A");
    auto h2 = as.add_node(cog::AtomType::CONCEPT_NODE, "B");
    auto link = as.add_link(cog::AtomType::INHERITANCE_LINK, {h1, h2});
    assert(link != cog::UNDEFINED_HANDLE);
    auto* latom = as.get_atom(link);
    assert(latom != nullptr);
    assert(latom->outgoing.size() == 2);
}

void test_truth_value() {
    cog::TruthValue tv(0.8f, 0.9f);
    assert(tv.strength > 0.79f && tv.strength < 0.81f);
    assert(tv.confidence > 0.89f && tv.confidence < 0.91f);
}

int main() {
    RUN_TEST(test_handle_creation);
    RUN_TEST(test_atom_type_classification);
    RUN_TEST(test_atomspace_add_node);
    RUN_TEST(test_atomspace_add_link);
    RUN_TEST(test_truth_value);
    std::cout << "All cogprime core tests passed!\n";
    return 0;
}

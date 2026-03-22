#include "../../include/cog/prime/prime.hpp"
#include <cassert>
#include <iostream>

// Minimal testing macro
#define RUN_TEST(test) \
    std::cout << "Running test: " << #test << "...\n"; \
    test(); \
    std::cout << "Test " << #test << " passed.\n";

void test_is_prime() {
    assert(is_prime(2) == true);
    assert(is_prime(3) == true);
    assert(is_prime(4) == false);
    assert(is_prime(5) == true);
    assert(is_prime(17) == true);
    assert(is_prime(18) == false);
    assert(is_prime(100) == false);
}

void test_next_prime() {
    assert(next_prime(2) == 3);
    assert(next_prime(3) == 5);
    assert(next_prime(5) == 7);
    assert(next_prime(17) == 19);
}

int main() {
    RUN_TEST(test_is_prime);
    RUN_TEST(test_next_prime);
    std::cout << "All cogprime prime tests passed!\n";
    return 0;
}


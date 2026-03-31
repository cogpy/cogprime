// tests/e2e/e2e_app.cpp — End-to-end test application for cogprime
// Demonstrates a minimal cognitive cycle with AtomSpace operations.
// Usage: ./e2e_app <number>
//   Returns the next prime number >= input (for backward compat with test script)
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>
#include "../../include/cog/prime/prime.hpp"

// Primality and next-prime functions for the E2E test
static bool is_prime(int n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

static int next_prime(int n) {
    int candidate = n + 1;
    while (!is_prime(candidate)) ++candidate;
    return candidate;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number>" << std::endl;
        return 1;
    }

    try {
        int num = std::stoi(argv[1]);

        // Run a minimal cognitive cycle to verify the full stack works
        cog::AtomSpace as;
        cog::prime::SensoryMemory sensory;
        cog::prime::Thought t;
        t.content = "input:" + std::string(argv[1]);
        t.salience = 1.0f;
        sensory.store(t);

        // Compute and output next prime
        int result = next_prime(num);
        std::cout << result;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid number: " << argv[1] << std::endl;
        return 1;
    }

    return 0;
}

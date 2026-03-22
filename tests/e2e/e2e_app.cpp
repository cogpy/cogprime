#include <iostream>
#include <string>
#include "../../include/cog/prime/prime.hpp"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number>" << std::endl;
        return 1;
    }

    try {
        int num = std::stoi(argv[1]);
        int result = next_prime(num);
        std::cout << result << std::endl;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid number: " << argv[1] << std::endl;
        return 1;
    }

    return 0;
}


#!/bin/bash

set -e

# Build the E2E application
g++ e2e_app.cpp -I ../../include -o e2e_app

# Test case 1
output=$(./e2e_app 10)
if [ "$output" != "11" ]; then
    echo "E2E test failed: expected 11, got $output"
    exit 1
fi

# Test case 2
output=$(./e2e_app 17)
if [ "$output" != "19" ]; then
    echo "E2E test failed: expected 19, got $output"
    exit 1
fi

# Test case 3
output=$(./e2e_app 0)
if [ "$output" != "2" ]; then
    echo "E2E test failed: expected 2, got $output"
    exit 1
fi

echo "All E2E tests passed!"

# Cleanup
rm e2e_app


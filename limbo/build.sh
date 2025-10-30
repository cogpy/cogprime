#!/bin/bash

#
# Build script for OpenCog Limbo implementation
# This script provides a simpler interface than the Makefile
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Limbo compiler is available
check_limbo() {
    if ! command -v limbo &> /dev/null; then
        print_warn "Limbo compiler not found in PATH"
        print_info "This implementation requires the Limbo compiler from Inferno OS"
        print_info "See: http://www.vitanuova.com/inferno/"
        print_info ""
        print_info "Skipping compilation - source files are available for review"
        return 1
    fi
    return 0
}

# Build modules
build_modules() {
    print_info "Building OpenCog Limbo modules..."
    
    modules=(
        "opencog/atomspace"
        "opencog/perception"
        "opencog/reasoning"
        "opencog/action"
        "opencog/learning"
        "opencog/cogcore"
    )
    
    for module in "${modules[@]}"; do
        if [ -f "${module}.b" ]; then
            print_info "Building ${module}..."
            limbo "${module}.b" 2>&1 || {
                print_error "Failed to build ${module}"
                return 1
            }
        else
            print_error "Module file ${module}.b not found"
            return 1
        fi
    done
    
    print_info "All modules built successfully"
    return 0
}

# Build examples
build_examples() {
    print_info "Building examples..."
    
    if [ -d "examples" ]; then
        cd examples
        for example in *.b; do
            if [ -f "$example" ]; then
                print_info "Building example: $example"
                limbo "$example" 2>&1 || {
                    print_warn "Failed to build $example"
                }
            fi
        done
        cd ..
    else
        print_warn "Examples directory not found"
    fi
}

# Run tests
run_tests() {
    print_info "Running tests..."
    print_warn "Test framework not yet implemented"
    # TODO: Implement test framework
}

# Clean build artifacts
clean() {
    print_info "Cleaning build artifacts..."
    rm -f opencog/*.dis
    rm -f examples/*.dis
    print_info "Clean complete"
}

# Show usage
usage() {
    cat << EOF
OpenCog Limbo Build Script

Usage: $0 [COMMAND]

Commands:
    build       Build all modules and examples (default)
    modules     Build only core modules
    examples    Build example programs
    test        Run test suite
    clean       Remove build artifacts
    help        Show this help message

Environment Variables:
    LIMBO       Path to Limbo compiler (default: limbo)

Examples:
    $0              # Build everything
    $0 modules      # Build only modules
    $0 clean        # Clean build artifacts
EOF
}

# Main script logic
main() {
    local command="${1:-build}"
    
    print_info "OpenCog Limbo Build System"
    print_info "Working directory: $SCRIPT_DIR"
    echo
    
    case "$command" in
        build)
            if check_limbo; then
                build_modules && build_examples
            else
                print_info "Limbo compiler not available - showing file structure instead"
                ls -lh opencog/*.b opencog/*.m 2>/dev/null || true
                [ -d examples ] && ls -lh examples/*.b 2>/dev/null || true
            fi
            ;;
        modules)
            if check_limbo; then
                build_modules
            fi
            ;;
        examples)
            if check_limbo; then
                build_examples
            fi
            ;;
        test)
            run_tests
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            print_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

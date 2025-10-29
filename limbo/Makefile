# Makefile for OpenCog Limbo Implementation

# Limbo compiler
LIMBO = limbo

# Directories
SRC_DIR = opencog
BUILD_DIR = dis/cogprime
EXAMPLE_DIR = examples

# Module files
MODULES = \
	opencog/atomspace.b \
	opencog/perception.b \
	opencog/reasoning.b \
	opencog/action.b \
	opencog/learning.b \
	opencog/cogcore.b

# Example programs
EXAMPLES = \
	examples/simple_agent.b

# Compiled output
DIS_FILES = $(MODULES:.b=.dis)
EXAMPLE_DIS = $(EXAMPLES:.b=.dis)

.PHONY: all clean modules examples test install

all: modules examples

# Build all modules
modules: $(DIS_FILES)

# Build examples
examples: modules $(EXAMPLE_DIS)

# Pattern rule for compiling .b to .dis
%.dis: %.b %.m
	@echo "Compiling $<..."
	@$(LIMBO) $<

# Specific module dependencies
opencog/atomspace.dis: opencog/atomspace.b opencog/atomspace.m
	@echo "Compiling AtomSpace module..."
	@$(LIMBO) opencog/atomspace.b

opencog/perception.dis: opencog/perception.b opencog/perception.m opencog/atomspace.dis
	@echo "Compiling Perception module..."
	@$(LIMBO) opencog/perception.b

opencog/reasoning.dis: opencog/reasoning.b opencog/reasoning.m opencog/atomspace.dis
	@echo "Compiling Reasoning module..."
	@$(LIMBO) opencog/reasoning.b

opencog/action.dis: opencog/action.b opencog/action.m opencog/atomspace.dis
	@echo "Compiling Action module..."
	@$(LIMBO) opencog/action.b

opencog/learning.dis: opencog/learning.b opencog/learning.m opencog/atomspace.dis
	@echo "Compiling Learning module..."
	@$(LIMBO) opencog/learning.b

opencog/cogcore.dis: opencog/cogcore.b opencog/cogcore.m opencog/atomspace.dis opencog/perception.dis opencog/reasoning.dis opencog/action.dis opencog/learning.dis
	@echo "Compiling CogCore module..."
	@$(LIMBO) opencog/cogcore.b

# Examples
examples/simple_agent.dis: examples/simple_agent.b modules
	@echo "Compiling simple_agent example..."
	@$(LIMBO) examples/simple_agent.b

# Test target (requires test files to be created)
test: modules
	@echo "Running tests..."
	@echo "Test framework not yet implemented"

# Install compiled modules
install: modules
	@echo "Installing modules to $(BUILD_DIR)..."
	@mkdir -p $(BUILD_DIR)
	@cp -v opencog/*.dis $(BUILD_DIR)/ 2>/dev/null || echo "No .dis files to install"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f opencog/*.dis
	@rm -f examples/*.dis
	@rm -f $(BUILD_DIR)/*.dis

# Help target
help:
	@echo "OpenCog Limbo Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build all modules and examples (default)"
	@echo "  modules   - Build only the core modules"
	@echo "  examples  - Build example programs"
	@echo "  test      - Run test suite"
	@echo "  install   - Install compiled modules"
	@echo "  clean     - Remove build artifacts"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Modules:"
	@echo "  atomspace - Core hypergraph database"
	@echo "  perception - Sensory processing"
	@echo "  reasoning  - Pattern matching and inference"
	@echo "  action     - Action selection"
	@echo "  learning   - Experience-based learning"
	@echo "  cogcore    - Integrated cognitive core"

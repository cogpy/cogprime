# Makefile for OpenCog Limbo Implementation

# Limbo compiler
LIMBO = limbo

# Directories
SRC_DIR = .
BUILD_DIR = dis/cogprime
EXAMPLE_DIR = examples

# Module files
MODULES = \
	atomspace.b \
	perception.b \
	reasoning.b \
	action.b \
	learning.b \
	cogcore.b

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
atomspace.dis: atomspace.b atomspace.m
	@echo "Compiling AtomSpace module..."
	@$(LIMBO) atomspace.b

perception.dis: perception.b perception.m atomspace.dis
	@echo "Compiling Perception module..."
	@$(LIMBO) perception.b

reasoning.dis: reasoning.b reasoning.m atomspace.dis
	@echo "Compiling Reasoning module..."
	@$(LIMBO) reasoning.b

action.dis: action.b action.m atomspace.dis
	@echo "Compiling Action module..."
	@$(LIMBO) action.b

learning.dis: learning.b learning.m atomspace.dis
	@echo "Compiling Learning module..."
	@$(LIMBO) learning.b

cogcore.dis: cogcore.b cogcore.m atomspace.dis perception.dis reasoning.dis action.dis learning.dis
	@echo "Compiling CogCore module..."
	@$(LIMBO) cogcore.b

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
	@cp -v *.dis $(BUILD_DIR)/ 2>/dev/null || echo "No .dis files to install"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f *.dis
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

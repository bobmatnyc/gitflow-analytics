# ============================================================================
# testing.mk - Test Execution Targets for GitFlow Analytics
# ============================================================================
# Provides: test execution, coverage, parallel/serial modes
# Include in main Makefile with: -include .makefiles/testing.mk
#
# Adapted from python-project-template for gitflow-analytics
# Dependencies: common.mk (for PYTHON, ENV system, PYTEST_ARGS)
# Last updated: 2024-12-08
# ============================================================================

# ============================================================================
# Test Target Declarations
# ============================================================================
.PHONY: test test-serial test-parallel test-fast test-coverage
.PHONY: test-unit test-integration test-e2e test-cov test-tui test-cli
.PHONY: test-ml test-reports test-config

# ============================================================================
# Primary Test Targets
# ============================================================================

test: test-parallel ## Run tests with parallel execution (default, 3-4x faster)

test-parallel: ## Run tests in parallel using all available CPUs
	@echo "$(YELLOW)🧪 Running tests in parallel (using all CPUs)...$(NC)"
	@if command -v pytest >/dev/null 2>&1; then \
		if command -v pytest-xdist >/dev/null 2>&1 || $(PYTHON) -c "import pytest_xdist" 2>/dev/null; then \
			$(PYTHON) -m pytest $(TEST_DIR)/ $(PYTEST_ARGS); \
		else \
			echo "$(YELLOW)⚠ pytest-xdist not found, running serially$(NC)"; \
			$(PYTHON) -m pytest $(TEST_DIR)/ -v; \
		fi; \
	else \
		echo "$(RED)✗ pytest not found. Install with: pip install pytest$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Parallel tests completed$(NC)"

test-serial: ## Run tests serially for debugging (disables parallelization)
	@echo "$(YELLOW)🧪 Running tests serially (debugging mode)...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n 0 -v
	@echo "$(GREEN)✓ Serial tests completed$(NC)"

# ============================================================================
# Fast Testing (Unit Tests Only)
# ============================================================================

test-fast: ## Run unit tests only in parallel (fastest)
	@echo "$(YELLOW)⚡ Running unit tests in parallel...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto -m unit -v
	@echo "$(GREEN)✓ Unit tests completed$(NC)"

# ============================================================================
# Coverage Reporting
# ============================================================================

test-coverage: ## Run tests with coverage report (parallel)
	@echo "$(YELLOW)📊 Running tests with coverage...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto \
		--cov=$(SRC_DIR) \
		--cov-report=html \
		--cov-report=term \
		--cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"
	@echo "$(BLUE)View with: open htmlcov/index.html$(NC)"

test-cov: test-coverage ## Alias for test-coverage (maintains compatibility)

# ============================================================================
# Test Category Targets (GitFlow Analytics specific)
# ============================================================================

test-unit: ## Run unit tests only
	@echo "$(YELLOW)🧪 Running unit tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto -m unit -v

test-integration: ## Run integration tests only
	@echo "$(YELLOW)🧪 Running integration tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/integration/ -n auto -v

test-e2e: ## Run end-to-end tests only
	@echo "$(YELLOW)🧪 Running e2e tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto -m "not unit and not integration" -v

test-tui: ## Run Terminal User Interface tests
	@echo "$(YELLOW)🧪 Running TUI tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto -m tui -v

test-cli: ## Run Command Line Interface tests
	@echo "$(YELLOW)🧪 Running CLI tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto -m cli -v

test-ml: ## Run Machine Learning and qualitative analysis tests
	@echo "$(YELLOW)🧪 Running ML tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto -m ml -v

test-reports: ## Run report generation tests
	@echo "$(YELLOW)🧪 Running report tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto -m reports -v

test-config: ## Run configuration loading and validation tests
	@echo "$(YELLOW)🧪 Running config tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR)/ -n auto -m config -v

# ============================================================================
# ENV-Specific Test Configurations
# ============================================================================
# The PYTEST_ARGS variable is configured in common.mk based on ENV:
#
# development (default):
#   PYTEST_ARGS := -n auto -v --tb=long
#   - Parallel execution with all CPUs
#   - Verbose output
#   - Long traceback for debugging
#
# staging:
#   PYTEST_ARGS := -n auto -v --tb=line
#   - Parallel execution
#   - Shorter traceback for CI logs
#
# production:
#   PYTEST_ARGS := -n auto -v --tb=short --strict-markers
#   - Parallel execution
#   - Minimal traceback
#   - Strict marker enforcement
#
# Override with: make ENV=production test
# ============================================================================

# ============================================================================
# Test Utilities
# ============================================================================

test-install-deps: ## Install test dependencies (pytest-xdist for parallel execution)
	@echo "$(YELLOW)📦 Installing test dependencies...$(NC)"
	@$(PIP) install pytest-xdist pytest-cov
	@echo "$(GREEN)✓ Test dependencies installed$(NC)"

test-markers: ## Show available test markers
	@echo "$(BLUE)Available test markers:$(NC)"
	@echo "  unit         - Unit tests for individual components"
	@echo "  integration  - Integration tests across multiple components"
	@echo "  tui          - Terminal User Interface tests"
	@echo "  cli          - Command Line Interface tests"
	@echo "  config       - Configuration loading and validation tests"
	@echo "  ml           - Machine Learning and qualitative analysis tests"
	@echo "  reports      - Report generation tests"
	@echo "  slow         - Tests that take a long time to run"
	@echo "  network      - Tests that require network access"
	@echo "  external     - Tests that require external services"

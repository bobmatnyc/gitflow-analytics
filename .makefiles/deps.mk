# ============================================================================
# deps.mk - Dependency Management for GitFlow Analytics
# ============================================================================
# Provides: pip/setuptools dependency management, requirements generation
# Include in main Makefile with: -include .makefiles/deps.mk
#
# Adapted from python-project-template for gitflow-analytics (pip/setuptools)
# Dependencies: common.mk (for colors, PYTHON)
# Last updated: 2024-12-08
# ============================================================================
#
# Workflow for updating dependencies:
#   1. make deps-check         - Verify current dependency state
#   2. make deps-update        - Update to latest compatible versions
#   3. make test               - Test with updated deps
#   4. make deps-freeze        - Generate requirements.txt
#   5. git add requirements*   - Commit if tests pass
#
# For reproducible installs:
#   make install               - Install exact versions from requirements
#
# For CI/CD integration:
#   make deps-check            - Verify dependencies are current
#   make deps-freeze           - Generate requirements.txt for Docker
# ============================================================================

# ============================================================================
# Dependency Management Target Declarations
# ============================================================================
.PHONY: deps-check deps-update deps-freeze deps-info deps-outdated
.PHONY: install install-prod install-dev install-editable venv
.PHONY: deps-compile deps-sync

# ============================================================================
# Installation Targets
# ============================================================================

venv: ## Create virtual environment if it doesn't exist
	@if [ ! -d "venv" ]; then \
		echo "$(YELLOW)🐍 Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv venv; \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
		echo "$(YELLOW)💡 Activate with: source venv/bin/activate$(NC)"; \
	else \
		echo "$(GREEN)✓ Virtual environment already exists$(NC)"; \
	fi

install: venv ## Install project in development mode with all dependencies
	@echo "$(YELLOW)📦 Installing gitflow-analytics in development mode...$(NC)"
	@if [ -d "venv" ] && [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(YELLOW)💡 Using virtual environment...$(NC)"; \
		. venv/bin/activate && $(PIP) install -e .; \
	else \
		$(PIP) install -e .; \
	fi
	@echo "$(GREEN)✓ Project installed with core dependencies$(NC)"

install-dev: ## Install project with development dependencies
	@echo "$(YELLOW)📦 Installing gitflow-analytics with dev dependencies...$(NC)"
	@$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Project installed with dev dependencies$(NC)"

install-prod: ## Install production dependencies only (no dev deps)
	@echo "$(YELLOW)📦 Installing production dependencies...$(NC)"
	@$(PIP) install -e .
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

install-editable: install-dev ## Alias for development installation

# ============================================================================
# Requirements Management
# ============================================================================

deps-freeze: ## Generate requirements.txt from current environment
	@echo "$(YELLOW)📤 Generating requirements files...$(NC)"
	@$(PIP) freeze > requirements.txt
	@echo "$(GREEN)✓ Generated requirements.txt$(NC)"
	@echo "$(BLUE)💡 Consider using pip-tools for better dependency management$(NC)"

deps-compile: ## Generate requirements.txt using pip-tools (if available)
	@echo "$(YELLOW)📤 Compiling requirements with pip-tools...$(NC)"
	@if command -v pip-compile >/dev/null 2>&1; then \
		pip-compile pyproject.toml --output-file requirements.txt; \
		pip-compile pyproject.toml --extra dev --output-file requirements-dev.txt; \
		echo "$(GREEN)✓ Requirements compiled with pip-tools$(NC)"; \
	else \
		echo "$(YELLOW)⚠ pip-tools not found. Install with: pip install pip-tools$(NC)"; \
		echo "$(YELLOW)Falling back to pip freeze...$(NC)"; \
		$(MAKE) deps-freeze; \
	fi

deps-sync: ## Sync environment with requirements.txt (if using pip-tools)
	@echo "$(YELLOW)🔄 Syncing environment with requirements...$(NC)"
	@if command -v pip-sync >/dev/null 2>&1; then \
		pip-sync requirements-dev.txt; \
		echo "$(GREEN)✓ Environment synced with requirements$(NC)"; \
	else \
		echo "$(YELLOW)⚠ pip-sync not found. Install with: pip install pip-tools$(NC)"; \
		echo "$(YELLOW)Falling back to regular install...$(NC)"; \
		$(MAKE) install-dev; \
	fi

# ============================================================================
# Dependency Checking and Updates
# ============================================================================

deps-check: ## Check if dependencies are up to date
	@echo "$(YELLOW)🔍 Checking dependency status...$(NC)"
	@if [ -f requirements.txt ]; then \
		echo "$(GREEN)✓ requirements.txt exists$(NC)"; \
		echo "Requirements file modified: $$(stat -f %Sm -t '%Y-%m-%d %H:%M:%S' requirements.txt 2>/dev/null || stat -c %y requirements.txt 2>/dev/null || echo 'unknown')"; \
	else \
		echo "$(YELLOW)⚠ requirements.txt not found$(NC)"; \
		echo "$(BLUE)  Run: make deps-freeze$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Checking for outdated packages...$(NC)"
	@$(MAKE) deps-outdated

deps-outdated: ## Show outdated packages
	@echo "$(YELLOW)📋 Checking for outdated packages...$(NC)"
	@$(PIP) list --outdated --format=columns || echo "$(GREEN)✓ All packages are up to date$(NC)"

deps-update: ## Update all dependencies to latest compatible versions
	@echo "$(YELLOW)⬆️  Updating dependencies...$(NC)"
	@echo "$(BLUE)Note: This will update all packages to latest versions$(NC)"
	@read -p "Continue? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(PIP) install --upgrade pip setuptools wheel; \
		$(PIP) install --upgrade -e ".[dev]"; \
		echo "$(GREEN)✓ Dependencies updated$(NC)"; \
		echo "$(YELLOW)📋 Generating new requirements.txt...$(NC)"; \
		$(MAKE) deps-freeze; \
		echo "$(YELLOW)📋 Review changes with: git diff requirements.txt$(NC)"; \
	else \
		echo "$(RED)Update cancelled$(NC)"; \
	fi

deps-info: ## Display dependency information
	@echo "$(BLUE)════════════════════════════════════════$(NC)"
	@echo "$(BLUE)GitFlow Analytics - Dependency Information$(NC)"
	@echo "$(BLUE)════════════════════════════════════════$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo ""
	@if [ -f requirements.txt ]; then \
		echo "$(GREEN)✓ requirements.txt exists$(NC)"; \
		echo "Requirements count: $$(wc -l < requirements.txt) packages"; \
		echo "Requirements modified: $$(stat -f %Sm -t '%Y-%m-%d %H:%M:%S' requirements.txt 2>/dev/null || stat -c %y requirements.txt 2>/dev/null || echo 'unknown')"; \
	else \
		echo "$(YELLOW)⚠ requirements.txt not found$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Core dependencies from pyproject.toml:$(NC)"
	@grep -A 20 "dependencies = \[" pyproject.toml | grep -E "^\s*\"" | head -10 || echo "  (could not parse)"
	@echo ""
	@echo "$(YELLOW)Development dependencies:$(NC)"
	@grep -A 20 "dev = \[" pyproject.toml | grep -E "^\s*\"" | head -10 || echo "  (could not parse)"

# ============================================================================
# Dependency Management Tools Installation
# ============================================================================

deps-install-tools: ## Install dependency management tools (pip-tools)
	@echo "$(YELLOW)📦 Installing dependency management tools...$(NC)"
	@$(PIP) install pip-tools
	@echo "$(GREEN)✓ pip-tools installed$(NC)"
	@echo "$(BLUE)💡 Now you can use:$(NC)"
	@echo "  make deps-compile  - Compile requirements with pip-tools"
	@echo "  make deps-sync     - Sync environment with requirements"

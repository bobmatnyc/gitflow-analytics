# ============================================================================
# release.mk - Version & Publishing Management for GitFlow Analytics
# ============================================================================
# Provides: version bumping, build, publish to PyPI, GitHub releases
# Include in main Makefile with: -include .makefiles/release.mk
#
# Adapted from python-project-template for gitflow-analytics
# Dependencies: common.mk (for colors, VERSION, PYTHON, ENV)
#               quality.mk (for pre-publish checks)
# Last updated: 2024-12-08
# ============================================================================

# ============================================================================
# Release Target Declarations
# ============================================================================
.PHONY: release-check release-patch release-minor release-major
.PHONY: release-build release-publish release-verify release-full
.PHONY: release-dry-run release-test-pypi
.PHONY: patch minor major version
.PHONY: build clean clean-all publish publish-test

# ============================================================================
# Version Management (GitFlow Analytics Integration)
# ============================================================================

version: ## Show current version
	@echo "$(BLUE)Current version:$(NC)"
	@$(PYTHON) $(VERSION_SCRIPT) get

patch: ## Bump patch version (X.Y.Z → X.Y.Z+1)
	@echo "$(YELLOW)🔧 Bumping patch version...$(NC)"
	@CURRENT=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	$(PYTHON) $(VERSION_SCRIPT) bump --type patch; \
	NEW=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "$(GREEN)✓ Version bumped: $$CURRENT → $$NEW$(NC)"

minor: ## Bump minor version (X.Y.Z → X.Y+1.0)
	@echo "$(YELLOW)✨ Bumping minor version...$(NC)"
	@CURRENT=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	$(PYTHON) $(VERSION_SCRIPT) bump --type minor; \
	NEW=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "$(GREEN)✓ Version bumped: $$CURRENT → $$NEW$(NC)"

major: ## Bump major version (X.Y.Z → X+1.0.0)
	@echo "$(YELLOW)💥 Bumping major version...$(NC)"
	@CURRENT=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	$(PYTHON) $(VERSION_SCRIPT) bump --type major; \
	NEW=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "$(GREEN)✓ Version bumped: $$CURRENT → $$NEW$(NC)"

# Legacy compatibility targets
version-patch: patch ## Alias for patch (maintains compatibility)
version-minor: minor ## Alias for minor (maintains compatibility)
version-major: major ## Alias for major (maintains compatibility)

# ============================================================================
# Release Prerequisites Check
# ============================================================================

release-check: ## Check if environment is ready for release
	@echo "$(YELLOW)🔍 Checking release prerequisites...$(NC)"
	@echo "Checking required tools..."
	@command -v git >/dev/null 2>&1 || (echo "$(RED)✗ git not found$(NC)" && exit 1)
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "$(RED)✗ python not found$(NC)" && exit 1)
	@echo "$(GREEN)✓ All required tools found$(NC)"
	@echo "Checking working directory..."
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "$(RED)✗ Working directory is not clean$(NC)"; \
		git status --short; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Working directory is clean$(NC)"
	@echo "Checking current branch..."
	@BRANCH=$$(git branch --show-current); \
	if [ "$$BRANCH" != "main" ]; then \
		echo "$(YELLOW)⚠ Currently on branch '$$BRANCH', not 'main'$(NC)"; \
		read -p "Continue anyway? [y/N]: " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "$(RED)Aborted$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo "$(GREEN)✓ On main branch$(NC)"; \
	fi
	@echo "$(GREEN)✓ Release prerequisites check passed$(NC)"

# ============================================================================
# Build Management
# ============================================================================

clean: ## Remove build artifacts
	@echo "$(YELLOW)🧹 Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)/ $(DIST_DIR)/ *.egg-info
	@echo "$(GREEN)✓ Build artifacts cleaned$(NC)"

clean-all: clean ## Remove all generated files
	@echo "$(YELLOW)🧹 Cleaning all generated files...$(NC)"
	@rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ All generated files cleaned$(NC)"

build: clean ## Build Python package for release
	@echo "$(YELLOW)📦 Building package...$(NC)"
	@$(MAKE) build-metadata
	@$(PIP) install --upgrade build
	@$(PYTHON) -m build $(BUILD_FLAGS)
	@if command -v twine >/dev/null 2>&1; then \
		twine check $(DIST_DIR)/*; \
		echo "$(GREEN)✓ Package validation passed$(NC)"; \
	else \
		echo "$(YELLOW)⚠ twine not found, skipping package validation$(NC)"; \
	fi
	@echo "$(GREEN)✓ Package built successfully$(NC)"
	@ls -la $(DIST_DIR)/

release-build: pre-publish build ## Build package with full quality checks

# ============================================================================
# Release Workflow Shortcuts
# ============================================================================

release-patch: release-check patch release-build ## Create a patch release (X.Y.Z+1)
	@echo "$(GREEN)✓ Patch release prepared$(NC)"
	@echo "$(BLUE)Next: Run 'make release-publish' to publish$(NC)"

release-minor: release-check minor release-build ## Create a minor release (X.Y+1.0)
	@echo "$(GREEN)✓ Minor release prepared$(NC)"
	@echo "$(BLUE)Next: Run 'make release-publish' to publish$(NC)"

release-major: release-check major release-build ## Create a major release (X+1.0.0)
	@echo "$(GREEN)✓ Major release prepared$(NC)"
	@echo "$(BLUE)Next: Run 'make release-publish' to publish$(NC)"

# ============================================================================
# Publishing to PyPI
# ============================================================================

publish: ## Publish to PyPI (requires PYPI_API_TOKEN)
	@echo "$(YELLOW)🚀 Publishing to PyPI...$(NC)"
	@if [ -z "$$PYPI_API_TOKEN" ]; then \
		echo "$(RED)✗ PYPI_API_TOKEN not set$(NC)"; \
		echo "$(YELLOW)Set it in .env.local: PYPI_API_TOKEN=pypi-your-token$(NC)"; \
		exit 1; \
	fi
	@$(PIP) install --upgrade twine
	@TWINE_USERNAME=__token__ TWINE_PASSWORD=$$PYPI_API_TOKEN \
		$(TWINE) upload $(DIST_DIR)/*
	@echo "$(GREEN)✓ Published to PyPI$(NC)"

publish-test: ## Publish to TestPyPI for testing
	@echo "$(YELLOW)🧪 Publishing to TestPyPI...$(NC)"
	@$(PIP) install --upgrade twine
	@$(TWINE) upload --repository testpypi $(DIST_DIR)/*
	@echo "$(GREEN)✓ Published to TestPyPI$(NC)"
	@echo "$(BLUE)Test install: pip install --index-url https://test.pypi.org/simple/ gitflow-analytics$(NC)"

release-publish: ## Publish release and create git tag
	@echo "$(YELLOW)🚀 Publishing release...$(NC)"
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "Publishing version: $$VERSION"; \
	read -p "Continue with publishing? [y/N]: " confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
		echo "$(RED)Publishing aborted$(NC)"; \
		exit 1; \
	fi
	@$(MAKE) publish
	@echo "$(YELLOW)📤 Creating git tag...$(NC)"
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	git add $(VERSION_FILE); \
	git commit -m "chore: bump version to $$VERSION" || true; \
	git tag "v$$VERSION"; \
	git push origin main; \
	git push origin "v$$VERSION"
	@echo "$(GREEN)✓ Git tag created and pushed$(NC)"
	@$(MAKE) release-verify

release-full: release-patch release-publish ## Complete patch release workflow

# ============================================================================
# Release Verification
# ============================================================================

release-verify: ## Verify release across all channels
	@echo "$(YELLOW)🔍 Verifying release...$(NC)"
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "Verifying version: $$VERSION"; \
	echo ""; \
	echo "$(BLUE)📦 PyPI:$(NC) https://pypi.org/project/gitflow-analytics/$$VERSION/"; \
	echo "$(BLUE)🏷️  GitHub:$(NC) https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v$$VERSION"; \
	echo ""; \
	echo "$(GREEN)✓ Release verification links generated$(NC)"
	@echo "$(BLUE)💡 Test installation with:$(NC)"
	@echo "  pip install gitflow-analytics==$$($(PYTHON) $(VERSION_SCRIPT) get)"

# ============================================================================
# Dry Run and Information
# ============================================================================

release-dry-run: ## Show what a patch release would do (dry run)
	@echo "$(YELLOW)🔍 DRY RUN: Patch release preview$(NC)"
	@echo "This would:"
	@echo "  1. Check prerequisites and working directory"
	@echo "  2. Bump patch version"
	@echo "  3. Run pre-publish quality checks"
	@echo "  4. Build Python package"
	@echo "  5. Wait for confirmation to publish"
	@echo "  6. Publish to PyPI and create git tag"
	@echo "  7. Show verification links"
	@echo ""
	@echo "$(BLUE)Current version:$(NC) $$($(PYTHON) $(VERSION_SCRIPT) get 2>/dev/null || echo 'unknown')"
	@if [ -f "$(VERSION_SCRIPT)" ]; then \
		CURRENT=$$($(PYTHON) $(VERSION_SCRIPT) get 2>/dev/null || echo "unknown"); \
		if [ "$$CURRENT" != "unknown" ]; then \
			echo "$(BLUE)Next patch version would be:$(NC) (run 'python $(VERSION_SCRIPT) patch --dry-run' to see)"; \
		fi; \
	fi

# ============================================================================
# Legacy Compatibility Targets
# ============================================================================

release: release-full ## Legacy compatibility target for full release workflow

# ============================================================================
# Environment and Status Information
# ============================================================================

show-env: ## Show environment and configuration
	@echo "$(BLUE)════════════════════════════════════════$(NC)"
	@echo "$(BLUE)GitFlow Analytics - Environment$(NC)"
	@echo "$(BLUE)════════════════════════════════════════$(NC)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo "Version: $$($(PYTHON) $(VERSION_SCRIPT) get 2>/dev/null || echo 'unknown')"
	@echo "Environment: $(ENV)"
	@echo ""
	@echo "$(YELLOW)PyPI Configuration:$(NC)"
	@if [ -n "$$PYPI_API_TOKEN" ]; then \
		echo "  PYPI_API_TOKEN: ✓ configured"; \
	else \
		echo "  PYPI_API_TOKEN: ✗ not set"; \
		echo "  $(BLUE)Set in .env.local: PYPI_API_TOKEN=pypi-your-token$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Git Status:$(NC)"
	@git status --short || echo "  (no changes)"

dev-status: show-env ## Show development status (alias for show-env)

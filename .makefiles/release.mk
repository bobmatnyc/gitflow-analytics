# ============================================================================
# release.mk - Version & Publishing Management for GitFlow Analytics
# ============================================================================
# Provides: version bumping, build, publish to PyPI, GitHub releases,
#           Homebrew tap updates, and one-command full-publish workflows.
# Include in main Makefile with: -include .makefiles/release.mk
#
# Adapted from python-project-template for gitflow-analytics
# Dependencies: common.mk (for colors, VERSION, PYTHON, ENV, UV, GH,
#                          HOMEBREW_TAP_DIR, HOMEBREW_FORMULA)
#               quality.mk (for pre-publish checks)
# Last updated: 2025-02
# ============================================================================

# ============================================================================
# Release Target Declarations
# ============================================================================
.PHONY: release-check release-patch release-minor release-major
.PHONY: release-build release-verify release-full release-dry-run
.PHONY: release-test-pypi release-all
.PHONY: patch minor major version
.PHONY: build clean clean-all
.PHONY: publish publish-pypi publish-test publish-gh publish-brew publish-all
.PHONY: gh-release brew-update commit-version

# ============================================================================
# CI-friendly confirmation gate
# ============================================================================
# Pass CONFIRM=yes on the command line to skip interactive prompts.
# Example: make publish-all CONFIRM=yes
CONFIRM ?=

# Macro: prompt user unless CONFIRM=yes
# Usage: $(call confirm-prompt, "Proceeding with publish?")
define confirm-prompt
	@if [ "$(CONFIRM)" != "yes" ]; then \
		read -p "$(1) [y/N]: " _confirm; \
		if [ "$$_confirm" != "y" ] && [ "$$_confirm" != "Y" ]; then \
			echo "$(RED)Aborted$(NC)"; \
			exit 1; \
		fi; \
	fi
endef

# ============================================================================
# Version Management (GitFlow Analytics Integration)
# ============================================================================

version: ## Show current version
	@echo "$(BLUE)Current version:$(NC)"
	@$(PYTHON) $(VERSION_SCRIPT) get

patch: ## Bump patch version (X.Y.Z -> X.Y.Z+1)
	@echo "$(YELLOW)Bumping patch version...$(NC)"
	@CURRENT=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	$(PYTHON) $(VERSION_SCRIPT) bump --type patch; \
	NEW=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "$(GREEN)Version bumped: $$CURRENT -> $$NEW$(NC)"

minor: ## Bump minor version (X.Y.Z -> X.Y+1.0)
	@echo "$(YELLOW)Bumping minor version...$(NC)"
	@CURRENT=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	$(PYTHON) $(VERSION_SCRIPT) bump --type minor; \
	NEW=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "$(GREEN)Version bumped: $$CURRENT -> $$NEW$(NC)"

major: ## Bump major version (X.Y.Z -> X+1.0.0)
	@echo "$(YELLOW)Bumping major version...$(NC)"
	@CURRENT=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	$(PYTHON) $(VERSION_SCRIPT) bump --type major; \
	NEW=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "$(GREEN)Version bumped: $$CURRENT -> $$NEW$(NC)"

# Legacy compatibility targets
version-patch: patch ## Alias for patch (maintains compatibility)
version-minor: minor ## Alias for minor (maintains compatibility)
version-major: major ## Alias for major (maintains compatibility)

# ============================================================================
# Release Prerequisites Check
# ============================================================================

release-check: ## Check if environment is ready for release
	@echo "$(YELLOW)Checking release prerequisites...$(NC)"
	@echo "Checking required tools..."
	@command -v git >/dev/null 2>&1 || (echo "$(RED)git not found$(NC)" && exit 1)
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "$(RED)python not found$(NC)" && exit 1)
	@if [ -z "$(UV)" ]; then \
		echo "$(YELLOW)uv not found — install with: curl -LsSf https://astral.sh/uv/install.sh | sh$(NC)"; \
	else \
		echo "$(GREEN)uv: $(UV)$(NC)"; \
	fi
	@if [ -z "$(GH)" ]; then \
		echo "$(YELLOW)gh not found — GitHub releases will not be available$(NC)"; \
	else \
		echo "$(GREEN)gh: $(GH)$(NC)"; \
	fi
	@echo "$(GREEN)All required tools found$(NC)"
	@echo "Checking working directory..."
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "$(RED)Working directory is not clean$(NC)"; \
		git status --short; \
		exit 1; \
	fi
	@echo "$(GREEN)Working directory is clean$(NC)"
	@echo "Checking current branch..."
	@BRANCH=$$(git branch --show-current); \
	if [ "$$BRANCH" != "main" ]; then \
		echo "$(YELLOW)Currently on branch '$$BRANCH', not 'main'$(NC)"; \
		if [ "$(CONFIRM)" != "yes" ]; then \
			read -p "Continue anyway? [y/N]: " confirm; \
			if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
				echo "$(RED)Aborted$(NC)"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "$(GREEN)On main branch$(NC)"; \
	fi
	@echo "$(GREEN)Release prerequisites check passed$(NC)"

# ============================================================================
# Build Management
# ============================================================================

clean: ## Remove build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)/ $(DIST_DIR)/ *.egg-info
	@echo "$(GREEN)Build artifacts cleaned$(NC)"

clean-all: clean ## Remove all generated files
	@echo "$(YELLOW)Cleaning all generated files...$(NC)"
	@rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)All generated files cleaned$(NC)"

build: clean ## Build Python package for release using uv
	@echo "$(YELLOW)Building package...$(NC)"
	@$(MAKE) build-metadata
	@if [ -n "$(UV)" ]; then \
		echo "Using uv build..."; \
		$(UV) build; \
	else \
		echo "$(YELLOW)uv not found, falling back to python -m build$(NC)"; \
		$(PIP) install --upgrade build; \
		$(PYTHON) -m build $(BUILD_FLAGS); \
	fi
	@echo "$(GREEN)Package built successfully$(NC)"
	@ls -la $(DIST_DIR)/

release-build: pre-publish build ## Build package with full quality checks

# ============================================================================
# Version Commit
# ============================================================================

commit-version: ## Commit version bump files and push to main
	@echo "$(YELLOW)Committing version bump...$(NC)"
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "Committing version $$VERSION"; \
	git add pyproject.toml $(VERSION_FILE); \
	git commit --no-verify -m "chore: bump version to $$VERSION"; \
	git push origin main
	@echo "$(GREEN)Version bump committed and pushed$(NC)"

# ============================================================================
# Publishing to PyPI
# ============================================================================

publish-pypi: ## Publish to PyPI using uv (reads credentials from ~/.pypirc)
	@echo "$(YELLOW)Publishing to PyPI...$(NC)"
	@if [ -z "$(UV)" ]; then \
		echo "$(RED)uv not found — cannot publish$(NC)"; \
		echo "$(YELLOW)Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d "$(DIST_DIR)" ] || [ -z "$$(ls -A $(DIST_DIR) 2>/dev/null)" ]; then \
		echo "$(RED)No dist files found. Run 'make build' first.$(NC)"; \
		exit 1; \
	fi
	$(call confirm-prompt,Publish to PyPI?)
	@$(UV) publish
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "$(GREEN)Published version $$VERSION to PyPI$(NC)"; \
	echo "$(BLUE)View at: https://pypi.org/project/gitflow-analytics/$$VERSION/$(NC)"

# publish is kept as alias for backward compatibility
publish: publish-pypi ## Publish to PyPI (alias for publish-pypi)

publish-test: ## Publish to TestPyPI for testing
	@echo "$(YELLOW)Publishing to TestPyPI...$(NC)"
	@if [ -n "$(UV)" ]; then \
		$(UV) publish --publish-url https://test.pypi.org/legacy/; \
	else \
		$(PIP) install --upgrade twine; \
		$(TWINE) upload --repository testpypi $(DIST_DIR)/*; \
	fi
	@echo "$(GREEN)Published to TestPyPI$(NC)"
	@echo "$(BLUE)Test install: pip install --index-url https://test.pypi.org/simple/ gitflow-analytics$(NC)"

# ============================================================================
# GitHub Release
# ============================================================================

gh-release: ## Create a GitHub release with auto-generated notes from git log
	@echo "$(YELLOW)Creating GitHub release...$(NC)"
	@if [ -z "$(GH)" ]; then \
		echo "$(RED)gh CLI not found$(NC)"; \
		echo "$(YELLOW)Install: https://cli.github.com$(NC)"; \
		exit 1; \
	fi
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	TAG="v$$VERSION"; \
	echo "Creating GitHub release $$TAG"; \
	PREV_TAG=$$(git describe --tags --abbrev=0 "$$TAG^" 2>/dev/null || \
	            git describe --tags --abbrev=0 HEAD~1 2>/dev/null || \
	            git rev-list --max-parents=0 HEAD); \
	echo "Generating release notes since $$PREV_TAG..."; \
	{ \
	  echo "## Changes in $$TAG"; \
	  echo ""; \
	  git log --pretty=format:"- %s (%h)" "$$PREV_TAG..$$TAG" 2>/dev/null || \
	  git log --pretty=format:"- %s (%h)" "$$PREV_TAG..HEAD"; \
	  echo ""; \
	  echo ""; \
	  echo "**Full changelog:** https://github.com/bobmatnyc/gitflow-analytics/compare/$$PREV_TAG...$$TAG"; \
	} > /tmp/release-notes.txt; \
	echo "Release notes preview:"; \
	cat /tmp/release-notes.txt; \
	echo ""
	$(call confirm-prompt,Create GitHub release?)
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	TAG="v$$VERSION"; \
	DIST_ARGS=""; \
	if [ -d "$(DIST_DIR)" ] && [ -n "$$(ls -A $(DIST_DIR) 2>/dev/null)" ]; then \
		for f in $(DIST_DIR)/*; do DIST_ARGS="$$DIST_ARGS $$f"; done; \
	fi; \
	$(GH) release create "$$TAG" \
		--title "$$TAG" \
		--notes-file /tmp/release-notes.txt \
		$$DIST_ARGS \
		|| (echo "$(YELLOW)Note: tag $$TAG may not exist yet — creating it now$(NC)" && \
		    git tag "$$TAG" && git push origin "$$TAG" && \
		    $(GH) release create "$$TAG" \
		        --title "$$TAG" \
		        --notes-file /tmp/release-notes.txt \
		        $$DIST_ARGS); \
	echo "$(GREEN)GitHub release $$TAG created$(NC)"; \
	echo "$(BLUE)View at: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/$$TAG$(NC)"

# publish-gh is a friendly alias for gh-release
publish-gh: gh-release ## Create GitHub release (alias for gh-release)

# ============================================================================
# Homebrew Tap Update
# ============================================================================

brew-update: ## Update Homebrew tap formula with new version and SHA256
	@echo "$(YELLOW)Updating Homebrew tap...$(NC)"
	@if [ ! -d "$(HOMEBREW_TAP_DIR)" ]; then \
		echo "$(RED)Homebrew tap directory not found: $(HOMEBREW_TAP_DIR)$(NC)"; \
		echo "$(YELLOW)Clone your tap first: git clone <tap-repo> $(HOMEBREW_TAP_DIR)$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(HOMEBREW_FORMULA)" ]; then \
		echo "$(RED)Formula not found: $(HOMEBREW_FORMULA)$(NC)"; \
		exit 1; \
	fi
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	SDIST="$(DIST_DIR)/gitflow_analytics-$$VERSION.tar.gz"; \
	if [ ! -f "$$SDIST" ]; then \
		SDIST="$(DIST_DIR)/gitflow-analytics-$$VERSION.tar.gz"; \
	fi; \
	if [ ! -f "$$SDIST" ]; then \
		echo "$(YELLOW)Local sdist not found, downloading from PyPI...$(NC)"; \
		PYPI_URL="https://files.pythonhosted.org/packages/source/g/gitflow-analytics/gitflow_analytics-$$VERSION.tar.gz"; \
		SDIST="/tmp/gitflow_analytics-$$VERSION.tar.gz"; \
		curl -fsSL "$$PYPI_URL" -o "$$SDIST" \
			|| (PYPI_URL2="https://files.pythonhosted.org/packages/source/g/gitflow-analytics/gitflow-analytics-$$VERSION.tar.gz"; \
			    curl -fsSL "$$PYPI_URL2" -o "$$SDIST"); \
	fi; \
	echo "Computing SHA256 for $$SDIST"; \
	SHA256=$$(shasum -a 256 "$$SDIST" | awk '{print $$1}'); \
	echo "SHA256: $$SHA256"; \
	PYPI_URL="https://files.pythonhosted.org/packages/source/g/gitflow-analytics/gitflow_analytics-$$VERSION.tar.gz"; \
	echo "Updating $(HOMEBREW_FORMULA)..."; \
	sed -i '' \
		-e "s|url \".*\"|url \"$$PYPI_URL\"|" \
		-e "s|sha256 \".*\"|sha256 \"$$SHA256\"|" \
		-e "s|version \".*\"|version \"$$VERSION\"|" \
		"$(HOMEBREW_FORMULA)"; \
	echo "$(GREEN)Formula updated$(NC)"; \
	echo "Formula diff:"; \
	git -C "$(HOMEBREW_TAP_DIR)" diff
	$(call confirm-prompt,Commit and push Homebrew tap?)
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	cd "$(HOMEBREW_TAP_DIR)" && \
	git add Formula/gitflow-analytics.rb && \
	git commit -m "chore: update gitflow-analytics to v$$VERSION" && \
	git push origin HEAD
	@echo "$(GREEN)Homebrew tap updated and pushed$(NC)"

# publish-brew is a friendly alias for brew-update
publish-brew: brew-update ## Update Homebrew tap (alias for brew-update)

# ============================================================================
# Full Publish Workflow
# ============================================================================

publish-all: build publish-pypi gh-release brew-update ## Full publish: build -> PyPI -> GitHub release -> Homebrew tap
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo ""; \
	echo "$(GREEN)=====================================================$(NC)"; \
	echo "$(GREEN) Published v$$VERSION successfully!$(NC)"; \
	echo "$(GREEN)=====================================================$(NC)"; \
	echo "$(BLUE)PyPI:    https://pypi.org/project/gitflow-analytics/$$VERSION/$(NC)"; \
	echo "$(BLUE)GitHub:  https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v$$VERSION$(NC)"

release-all: patch commit-version publish-all ## Version bump + full publish workflow (patch -> commit -> publish-all)

# ============================================================================
# Release Workflow Shortcuts (with quality gate)
# ============================================================================

release-patch: release-check patch release-build ## Create a patch release (X.Y.Z+1)
	@echo "$(GREEN)Patch release prepared$(NC)"
	@echo "$(BLUE)Next: Run 'make publish-all' to publish$(NC)"

release-minor: release-check minor release-build ## Create a minor release (X.Y+1.0)
	@echo "$(GREEN)Minor release prepared$(NC)"
	@echo "$(BLUE)Next: Run 'make publish-all' to publish$(NC)"

release-major: release-check major release-build ## Create a major release (X+1.0.0)
	@echo "$(GREEN)Major release prepared$(NC)"
	@echo "$(BLUE)Next: Run 'make publish-all' to publish$(NC)"

# ============================================================================
# Release Verification
# ============================================================================

release-verify: ## Verify release across all channels
	@echo "$(YELLOW)Verifying release...$(NC)"
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "Verifying version: $$VERSION"; \
	echo ""; \
	echo "$(BLUE)PyPI:   https://pypi.org/project/gitflow-analytics/$$VERSION/$(NC)"; \
	echo "$(BLUE)GitHub: https://github.com/bobmatnyc/gitflow-analytics/releases/tag/v$$VERSION$(NC)"; \
	echo ""; \
	echo "$(GREEN)Release verification links generated$(NC)"
	@echo "$(BLUE)Test installation with:$(NC)"
	@echo "  pip install gitflow-analytics==$$($(PYTHON) $(VERSION_SCRIPT) get)"

# ============================================================================
# Dry Run and Information
# ============================================================================

release-dry-run: ## Show what a patch release would do (dry run)
	@echo "$(YELLOW)DRY RUN: Patch release preview$(NC)"
	@echo "This would:"
	@echo "  1. Check prerequisites and working directory"
	@echo "  2. Bump patch version"
	@echo "  3. Run pre-publish quality checks"
	@echo "  4. Build Python package (uv build)"
	@echo "  5. Publish to PyPI (uv publish)"
	@echo "  6. Create GitHub release (gh release create)"
	@echo "  7. Update Homebrew tap formula and push"
	@echo ""
	@echo "$(BLUE)Current version:$(NC) $$($(PYTHON) $(VERSION_SCRIPT) get 2>/dev/null || echo 'unknown')"
	@if [ -n "$(UV)" ]; then \
		echo "$(GREEN)uv: $(UV)$(NC)"; \
	else \
		echo "$(YELLOW)uv: not found (required for publish-pypi)$(NC)"; \
	fi
	@if [ -n "$(GH)" ]; then \
		echo "$(GREEN)gh: $(GH)$(NC)"; \
	else \
		echo "$(YELLOW)gh: not found (required for gh-release)$(NC)"; \
	fi
	@if [ -d "$(HOMEBREW_TAP_DIR)" ]; then \
		echo "$(GREEN)Homebrew tap: $(HOMEBREW_TAP_DIR)$(NC)"; \
	else \
		echo "$(YELLOW)Homebrew tap: $(HOMEBREW_TAP_DIR) (not found)$(NC)"; \
	fi

# ============================================================================
# Legacy Compatibility Targets
# ============================================================================

release: release-full ## Legacy compatibility target for full release workflow

release-full: release-patch publish ## Complete patch release workflow (legacy)

release-publish: ## Publish release and create git tag (legacy interactive flow)
	@echo "$(YELLOW)Publishing release...$(NC)"
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	echo "Publishing version: $$VERSION"
	$(call confirm-prompt,Continue with publishing?)
	@$(MAKE) publish-pypi
	@echo "$(YELLOW)Creating git tag...$(NC)"
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get); \
	git add $(VERSION_FILE); \
	git commit --no-verify -m "chore: bump version to $$VERSION" || true; \
	git tag "v$$VERSION"; \
	git push origin main; \
	git push origin "v$$VERSION"
	@echo "$(GREEN)Git tag created and pushed$(NC)"
	@$(MAKE) release-verify

# ============================================================================
# Environment and Status Information
# ============================================================================

show-env: ## Show environment and configuration
	@echo "$(BLUE)============================================$(NC)"
	@echo "$(BLUE)GitFlow Analytics - Environment$(NC)"
	@echo "$(BLUE)============================================$(NC)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo "Version: $$($(PYTHON) $(VERSION_SCRIPT) get 2>/dev/null || echo 'unknown')"
	@echo "Environment: $(ENV)"
	@echo ""
	@echo "$(YELLOW)Build Tools:$(NC)"
	@if [ -n "$(UV)" ]; then \
		echo "  uv:     $(UV) ($$($(UV) --version 2>/dev/null || echo 'version unknown'))"; \
	else \
		echo "  uv:     not found"; \
	fi
	@if [ -n "$(GH)" ]; then \
		echo "  gh:     $(GH) ($$($(GH) --version 2>/dev/null | head -1 || echo 'version unknown'))"; \
	else \
		echo "  gh:     not found"; \
	fi
	@echo ""
	@echo "$(YELLOW)Homebrew Tap:$(NC)"
	@if [ -d "$(HOMEBREW_TAP_DIR)" ]; then \
		echo "  Tap dir: $(HOMEBREW_TAP_DIR)"; \
		echo "  Formula: $(HOMEBREW_FORMULA)"; \
	else \
		echo "  Tap dir not found: $(HOMEBREW_TAP_DIR)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Git Status:$(NC)"
	@git status --short || echo "  (no changes)"

dev-status: show-env ## Show development status (alias for show-env)

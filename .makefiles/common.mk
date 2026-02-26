# ============================================================================
# common.mk - Core Makefile Infrastructure for GitFlow Analytics
# ============================================================================
# Provides: strict error handling, colors, ENV system, build metadata
# Include in main Makefile with: -include .makefiles/common.mk
#
# Adapted from python-project-template for gitflow-analytics
# Last updated: 2024-12-08
# ============================================================================

# ============================================================================
# Strict Error Handling
# ============================================================================
# Enable bash strict mode for safer shell execution
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# ============================================================================
# Terminal Colors
# ============================================================================
# ANSI color codes for terminal output formatting
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m  # No Color

# ============================================================================
# External Tool Detection
# ============================================================================
# uv: fast Python package manager (replaces pip + twine for build/publish)
UV := $(shell command -v uv 2>/dev/null)
# gh: GitHub CLI for creating releases and interacting with GitHub API
GH := $(shell command -v gh 2>/dev/null)

# ============================================================================
# Homebrew Tap Configuration
# ============================================================================
HOMEBREW_TAP_DIR := $(shell brew --repository bobmatnyc/tools 2>/dev/null || echo "$(HOME)/homebrew-tools")
HOMEBREW_FORMULA := $(HOMEBREW_TAP_DIR)/Formula/gitflow-analytics.rb

# ============================================================================
# Environment Detection & Configuration
# ============================================================================
# Environment-based configuration system
# Supports: development (default), staging, production
# Override with: make ENV=production <target>
ENV ?= development
export ENV

# Detect user's shell for compatibility
DETECTED_SHELL := $(shell echo $$SHELL | grep -o '[^/]*$$')

# Python detection (prefer venv python, then python3, then python)
PYTHON := $(or $(shell [ -f venv/bin/python ] && echo venv/bin/python),$(shell command -v python3 2>/dev/null),$(shell command -v python 2>/dev/null))
PIP := $(PYTHON) -m pip

# Tool commands
BLACK := $(PYTHON) -m black
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
BANDIT := $(PYTHON) -m bandit
TWINE := $(PYTHON) -m twine

# ENV-specific configurations
# Customize these for your project's needs
ifeq ($(ENV),production)
    # Production: strict, fast, minimal output
    PYTEST_ARGS := -n auto -v --tb=short --strict-markers
    BUILD_FLAGS := --no-isolation
    RUFF_ARGS := --quiet
else ifeq ($(ENV),staging)
    # Staging: balanced settings for pre-production testing
    PYTEST_ARGS := -n auto -v --tb=line
    BUILD_FLAGS :=
    RUFF_ARGS :=
else
    # Development (default): verbose, helpful errors
    PYTEST_ARGS := -n auto -v --tb=long
    BUILD_FLAGS :=
    RUFF_ARGS := --verbose
endif

# ============================================================================
# Project-Specific Variables
# ============================================================================
# GitFlow Analytics specific directories and files
SRC_DIR := src/gitflow_analytics
TEST_DIR := tests
VERSION_SCRIPT := scripts/manage_version.py
VERSION_FILE := src/gitflow_analytics/_version.py

# Build directories
BUILD_DIR := build
DIST_DIR := dist

# Load environment variables from .env.local if it exists
-include .env.local
export

# ============================================================================
# Utility Functions
# ============================================================================
# Check if command exists in PATH
command-exists = $(shell command -v $(1) 2>/dev/null)

# Get current version — read directly from _version.py for reliability.
# Falls back to version script if _version.py is unavailable.
ifneq (,$(wildcard $(VERSION_FILE)))
    VERSION := $(shell grep '__version__' $(VERSION_FILE) | sed 's/.*"\(.*\)"/\1/' 2>/dev/null || \
                       $(PYTHON) $(VERSION_SCRIPT) get 2>/dev/null || echo "unknown")
endif

# Get current build number (if BUILD_NUMBER file exists)
BUILD_NUMBER_FILE ?= BUILD_NUMBER
ifneq (,$(wildcard $(BUILD_NUMBER_FILE)))
    BUILD_NUMBER := $(shell cat $(BUILD_NUMBER_FILE))
endif

# ============================================================================
# Build Metadata Tracking
# ============================================================================
.PHONY: build-metadata build-info-json

build-metadata: ## Track build metadata in JSON format
	@echo "$(YELLOW)📋 Tracking build metadata...$(NC)"
	@mkdir -p $(BUILD_DIR)
	@VERSION=$$($(PYTHON) $(VERSION_SCRIPT) get 2>/dev/null || echo "0.0.0"); \
	BUILD_NUM=$$(cat $(BUILD_NUMBER_FILE) 2>/dev/null || echo "0"); \
	COMMIT=$$(git rev-parse HEAD 2>/dev/null || echo "unknown"); \
	SHORT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo "unknown"); \
	BRANCH=$$(git branch --show-current 2>/dev/null || echo "unknown"); \
	TIMESTAMP=$$(date -u +%Y-%m-%dT%H:%M:%SZ); \
	PYTHON_VER=$$($(PYTHON) --version 2>&1); \
	echo "{" > $(BUILD_DIR)/metadata.json; \
	echo '  "version": "'$$VERSION'",' >> $(BUILD_DIR)/metadata.json; \
	echo '  "build_number": '$$BUILD_NUM',' >> $(BUILD_DIR)/metadata.json; \
	echo '  "commit": "'$$COMMIT'",' >> $(BUILD_DIR)/metadata.json; \
	echo '  "commit_short": "'$$SHORT_COMMIT'",' >> $(BUILD_DIR)/metadata.json; \
	echo '  "branch": "'$$BRANCH'",' >> $(BUILD_DIR)/metadata.json; \
	echo '  "timestamp": "'$$TIMESTAMP'",' >> $(BUILD_DIR)/metadata.json; \
	echo '  "python_version": "'$$PYTHON_VER'",' >> $(BUILD_DIR)/metadata.json; \
	echo '  "environment": "'$${ENV:-development}'"' >> $(BUILD_DIR)/metadata.json; \
	echo "}" >> $(BUILD_DIR)/metadata.json
	@echo "$(GREEN)✓ Build metadata saved to $(BUILD_DIR)/metadata.json$(NC)"

build-info-json: build-metadata ## Display build metadata from JSON
	@if [ -f $(BUILD_DIR)/metadata.json ]; then \
		cat $(BUILD_DIR)/metadata.json; \
	else \
		echo "$(YELLOW)No build metadata found. Run 'make build-metadata' first.$(NC)"; \
	fi

# ============================================================================
# Environment Information Target
# ============================================================================
.PHONY: env-info

env-info: ## Display current environment configuration
	@echo "$(BLUE)════════════════════════════════════════$(NC)"
	@echo "$(BLUE)GitFlow Analytics - Environment Configuration$(NC)"
	@echo "$(BLUE)════════════════════════════════════════$(NC)"
	@echo "Environment: $(ENV)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo "Shell: $(DETECTED_SHELL)"
	@echo "Version: $(or $(VERSION),unknown)"
	@echo "Build: $(or $(BUILD_NUMBER),unknown)"
	@echo ""
	@echo "$(YELLOW)Environment-Specific Settings:$(NC)"
	@echo "Pytest Args: $(PYTEST_ARGS)"
	@echo "Build Flags: $(BUILD_FLAGS)"
	@echo "Ruff Args: $(RUFF_ARGS)"
	@echo ""
	@echo "$(GREEN)To change environment:$(NC)"
	@echo "  make ENV=production <target>"
	@echo "  make ENV=staging <target>"

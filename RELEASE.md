# Release Management Guide

This document describes the Makefile-based release workflow for gitflow-analytics.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Release Workflow](#release-workflow)
- [Common Tasks](#common-tasks)
- [PyPI Publishing](#pypi-publishing)
- [Troubleshooting](#troubleshooting)

## Overview

The gitflow-analytics project uses a Makefile-based workflow for:

- **Quality Control**: Automated testing, linting, and formatting
- **Version Management**: Semantic versioning with automated bumping
- **Build Automation**: Clean, reproducible builds
- **Release Process**: One-command releases with safety checks
- **PyPI Publishing**: Automated package publishing

### Why Makefile?

- **Single Source of Truth**: All release commands in one place
- **Reproducible**: Same commands work locally and in CI
- **Discoverable**: `make help` shows all available commands
- **Safe**: Built-in validation and checks
- **Simple**: No complex GitHub Actions to maintain

## Quick Start

### First Time Setup

```bash
# 1. Install development dependencies
make install-dev

# 2. Configure PyPI token (for publishing)
echo 'PYPI_API_TOKEN=pypi-your-token-here' > .env.local

# 3. Verify setup
make show-env
```

### Making a Release

```bash
# Patch release (3.13.1 → 3.13.2)
make release-patch

# Minor release (3.13.1 → 3.14.0)
make release-minor

# Major release (3.13.1 → 4.0.0)
make release-major
```

That's it! The Makefile handles everything automatically.

## Prerequisites

### Required Tools

- **Python 3.9+**: The project runtime
- **Git**: For version control and tagging
- **pip**: For package management

### Required Permissions

- **Git**: Push access to `main` branch and tags
- **PyPI**: API token with upload permissions

### Environment Variables

Create a `.env.local` file in the project root:

```bash
# Required for PyPI publishing
PYPI_API_TOKEN=pypi-your-token-here

# Optional: for TestPyPI
TEST_PYPI_API_TOKEN=pypi-your-test-token-here
```

**Get Your PyPI Token:**

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Scope it to the `gitflow-analytics` project (recommended)
4. Copy the token (starts with `pypi-`)
5. Add to `.env.local` (never commit this file!)

## Release Workflow

### Understanding the Process

A complete release involves these steps:

1. **Quality Gate**: Run tests and linting
2. **Version Bump**: Update version number
3. **Build**: Create distribution packages
4. **Validation**: Verify everything is ready
5. **Git Tag**: Tag the release version
6. **Git Push**: Push code and tags to GitHub
7. **Publish**: Upload to PyPI

The Makefile automates all of these steps.

### Full Release Process

```bash
# One-command release (recommended)
make release-patch   # or release-minor, or release-major

# This will:
# 1. Run all tests
# 2. Run linting checks
# 3. Bump the version
# 4. Commit the version change
# 5. Build distribution packages
# 6. Validate packages
# 7. Check git status is clean
# 8. Create and push git tag
# 9. Publish to PyPI (if token configured)
```

### Step-by-Step Release

If you prefer more control:

```bash
# 1. Ensure quality
make quality-gate

# 2. Bump version
make version-patch   # or version-minor, or version-major

# 3. Build packages
make build

# 4. Check release readiness
make release-check

# 5. Complete the release
make release
```

### Manual Version Management

```bash
# Check current version
make version

# Bump versions manually
make version-patch   # 3.13.1 → 3.13.2
make version-minor   # 3.13.1 → 3.14.0
make version-major   # 3.13.1 → 4.0.0

# Or set specific version
python scripts/manage_version.py set --version 4.0.0
git add src/gitflow_analytics/_version.py
git commit -m "chore: bump version to 4.0.0"
```

## Common Tasks

### Development Workflow

```bash
# Install for development
make install-dev

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Run all quality checks
make quality-gate
```

### Testing

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage report
make test-cov
```

### Code Quality

```bash
# Check code style (no changes)
make lint

# Auto-fix code style issues
make lint-fix

# Format code (alias for lint-fix)
make format

# Type checking (if configured)
make type-check

# Security scanning
make security-check
```

### Building

```bash
# Clean build artifacts
make clean

# Build distribution packages
make build

# Clean everything (including caches)
make clean-all
```

### Version Information

```bash
# Show current version
make version

# Show environment and configuration
make show-env

# Show development status
make dev-status
```

## PyPI Publishing

### Publishing to PyPI

```bash
# Publish latest build to PyPI
make publish

# Prerequisites:
# - PYPI_API_TOKEN must be set in .env.local
# - Packages must be built (make build)
```

### Publishing to TestPyPI

```bash
# Publish to TestPyPI for testing
make publish-test

# Prerequisites:
# - TEST_PYPI_API_TOKEN set in .env.local
# - Or PYPI_API_TOKEN will be used as fallback
```

### Testing Published Packages

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ gitflow-analytics

# Install from PyPI
pip install gitflow-analytics
```

## Troubleshooting

### Git Working Directory Not Clean

**Problem**: `make release` fails with "Git working directory is not clean"

**Solution**:
```bash
# Check what's uncommitted
git status

# Commit or stash changes
git add .
git commit -m "your message"

# Or stash for later
git stash
```

### Version Already Released

**Problem**: "Version X.Y.Z already released"

**Solution**:
```bash
# Bump the version first
make version-patch   # or version-minor, or version-major

# Then release
make release
```

### PyPI Token Not Set

**Problem**: "PYPI_API_TOKEN not set"

**Solution**:
```bash
# Create .env.local file
echo 'PYPI_API_TOKEN=pypi-your-token-here' > .env.local

# Verify it's loaded
make show-env
```

### Test Failures

**Problem**: `make release` fails during tests

**Solution**:
```bash
# Run tests to see failures
make test

# Fix the failing tests
# Then run full quality gate
make quality-gate

# Once passing, retry release
make release-patch
```

### Build Failures

**Problem**: Build step fails

**Solution**:
```bash
# Clean build artifacts
make clean-all

# Reinstall dependencies
make install-dev

# Try building again
make build
```

### Package Already Exists on PyPI

**Problem**: "File already exists" when publishing

**Solution**:
```bash
# You cannot overwrite published packages
# Must bump version and release again

# Bump version
make version-patch

# Create new release
make release
```

### Missing Dependencies

**Problem**: "Command not found: pytest/ruff/black"

**Solution**:
```bash
# Install all development dependencies
make install-dev

# Or install specific tools
pip install pytest ruff black mypy bandit
```

## Advanced Usage

### Custom Release Workflow

```bash
# 1. Make your changes
git add .
git commit -m "feat: add new feature"

# 2. Run quality checks
make quality-gate

# 3. Bump version for your change type
make version-minor   # for features
# or
make version-patch   # for bug fixes

# 4. Build and validate
make build
make release-check

# 5. Complete release
make release
```

### Pre-release Versions

```bash
# Set a pre-release version manually
python scripts/manage_version.py set --version 4.0.0-beta.1

# Commit it
git add src/gitflow_analytics/_version.py
git commit -m "chore: prepare 4.0.0-beta.1"

# Tag and push
git tag v4.0.0-beta.1
git push origin main
git push origin v4.0.0-beta.1

# Build and publish to TestPyPI
make build
make publish-test
```

### Hotfix Releases

```bash
# 1. Create hotfix branch from latest release tag
git checkout -b hotfix/security-fix v3.13.1

# 2. Make your fix
# ... edit files ...
git commit -am "fix: security vulnerability"

# 3. Bump patch version
make version-patch

# 4. Test thoroughly
make quality-gate

# 5. Merge to main
git checkout main
git merge hotfix/security-fix

# 6. Release
make release
```

### Rollback a Release

If you need to rollback a bad release:

```bash
# 1. Delete the git tag locally and remotely
git tag -d v3.14.0
git push origin :refs/tags/v3.14.0

# 2. Reset to previous version
python scripts/manage_version.py set --version 3.13.1
git add src/gitflow_analytics/_version.py
git commit -m "chore: rollback to 3.13.1"

# 3. Note: Cannot remove from PyPI
# Must release a new fixed version instead
make release-patch
```

## Migrating from GitHub Actions

This project previously used GitHub Actions for releases. The old workflows are backed up in `.github/workflows.backup/` for reference.

### Key Differences

| Aspect | GitHub Actions | Makefile |
|--------|----------------|----------|
| **Trigger** | Automatic on push | Manual command |
| **Control** | Limited | Full control |
| **Debugging** | Hard (remote) | Easy (local) |
| **Speed** | Slower (CI queue) | Instant |
| **Visibility** | Hidden in YAML | Clear in Makefile |

### Benefits of Makefile Approach

- **Local Testing**: Test the entire release process locally
- **No CI Queue**: Instant feedback, no waiting
- **Simpler**: One file instead of multiple YAML configs
- **Reproducible**: Same commands everywhere
- **Discoverable**: `make help` shows everything
- **Flexible**: Easy to customize and extend

## Best Practices

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major (X.0.0)**: Breaking changes
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

```bash
make release-patch   # Bug fixes
make release-minor   # New features
make release-major   # Breaking changes
```

### Release Checklist

Before releasing, ensure:

- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Git working directory is clean
- [ ] Version is bumped appropriately
- [ ] CHANGELOG.md is updated (if applicable)
- [ ] PyPI token is configured

### Commit Message Format

Use conventional commits:

```bash
feat: add new feature
fix: correct bug
docs: update documentation
style: format code
refactor: restructure code
test: add tests
chore: update tooling
```

This helps determine version bumps:
- `feat:` → Minor version bump
- `fix:` → Patch version bump
- `BREAKING CHANGE:` → Major version bump

## Help and Support

### Getting Help

```bash
# Show all available commands
make help

# Show environment configuration
make show-env

# Show current status
make dev-status
```

### Additional Resources

- [Makefile](./Makefile) - Complete command reference
- [CLAUDE.md](./CLAUDE.md) - Project instructions for AI assistants
- [README.md](./README.md) - Project overview
- [PyPI Package](https://pypi.org/project/gitflow-analytics/)
- [GitHub Repository](https://github.com/bobmatnyc/gitflow-analytics)

### Common Questions

**Q: Can I still use GitHub Actions?**
A: Yes, but the Makefile approach is recommended for simplicity and control.

**Q: What if I don't want to publish to PyPI?**
A: Just skip the publish step. Release will still create git tags.

**Q: Can I test releases before publishing?**
A: Yes! Use `make publish-test` to publish to TestPyPI first.

**Q: How do I automate releases again?**
A: You can call Makefile targets from GitHub Actions if needed:
```yaml
- name: Release
  run: make release-patch
```

**Q: What about semantic-release?**
A: The Makefile provides similar functionality with more transparency and control.

---

**Last Updated**: 2025-11-10
**Maintainer**: Bob Matyas <bobmatnyc@gmail.com>

# GitFlow Analytics Project Organization Standard

**Version:** 1.0
**Last Updated:** October 2025
**Status:** Official Organization Standard

This document defines the official file and directory organization standards for the GitFlow Analytics project. All contributors should follow these guidelines to maintain a clean, consistent project structure.

## Table of Contents
- [Organization Philosophy](#organization-philosophy)
- [Directory Structure](#directory-structure)
- [File Placement Rules](#file-placement-rules)
- [Naming Conventions](#naming-conventions)
- [Framework-Specific Patterns](#framework-specific-patterns)
- [Temporary Files Policy](#temporary-files-policy)
- [Migration Procedures](#migration-procedures)

---

## Organization Philosophy

GitFlow Analytics follows a **feature-based organization** with clear separation of concerns:

1. **Source code** (`src/`) - Production application code only
2. **Tests** (`tests/`) - Test files mirroring source structure
3. **Documentation** (`docs/`) - All project documentation
4. **Configuration** (root + `configs/`) - Sample configs and templates
5. **Build artifacts** (`.gitignore`d) - Generated files excluded from version control

**Core Principle:** The root directory should contain only essential project metadata and configuration files. Everything else belongs in an organized subdirectory.

---

## Directory Structure

```
gitflow-analytics/
├── .github/                    # GitHub-specific configuration
│   └── workflows/             # CI/CD workflows
├── configs/                    # Sample configuration files
│   ├── config-sample.yaml     # Basic configuration example
│   └── config-sample-ml.yaml  # ML-enabled configuration example
├── docs/                       # All project documentation
│   ├── getting-started/       # User onboarding guides
│   ├── guides/                # Task-oriented user guides
│   ├── examples/              # Real-world usage examples
│   ├── reference/             # Technical reference material
│   ├── developer/             # Developer and contributor docs
│   ├── architecture/          # System design and architecture
│   ├── design/                # Design documents and decisions
│   ├── deployment/            # Operations and deployment
│   └── _archive/              # Archived documentation
│       ├── old-reports/       # Archived test reports
│       ├── old-logs/          # Archived log files
│       ├── temp-files/        # Archived temporary files
│       └── analysis-files/    # Archived analysis documents
├── examples/                   # Example scripts and configurations
├── scripts/                    # Utility scripts for development
│   └── fix_identity_database.py
├── src/                        # Source code (production)
│   └── gitflow_analytics/     # Main package
│       ├── core/              # Core analysis logic
│       ├── extractors/        # Data extraction modules
│       ├── integrations/      # External API integrations
│       ├── metrics/           # Metric calculations
│       ├── models/            # Data models
│       ├── qualitative/       # ML and qualitative analysis
│       │   ├── classifiers/   # ML classifiers
│       │   ├── core/          # Core ML infrastructure
│       │   ├── models/        # Data schemas
│       │   └── utils/         # Utilities
│       └── reports/           # Report generation
├── tests/                      # Test suite
│   ├── core/                  # Core functionality tests
│   ├── qualitative/           # ML system tests
│   └── [other test modules]   # Mirror src/ structure
├── build/                      # Build artifacts (gitignored)
├── dist/                       # Distribution packages (gitignored)
├── .gitflow-cache/            # Application cache (gitignored)
├── reports/                    # Generated reports (gitignored)
├── .claude-mpm/               # Claude MPM data (gitignored)
├── .mypy_cache/               # Type checking cache (gitignored)
├── .pytest_cache/             # Test cache (gitignored)
├── .ruff_cache/               # Linter cache (gitignored)
├── CHANGELOG.md               # Version history (maintained by semantic-release)
├── CLAUDE.md                  # Developer instructions for AI assistants
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # Project license
├── pyproject.toml             # Python project configuration
├── pytest.ini                 # Pytest configuration
└── README.md                  # Project overview and quick start
```

---

## File Placement Rules

### Source Code (`src/gitflow_analytics/`)

**Rule:** Only production-ready application code belongs in `src/`

| File Type | Location | Example |
|-----------|----------|---------|
| Core logic | `src/gitflow_analytics/core/` | `analyzer.py`, `identity.py` |
| Data extractors | `src/gitflow_analytics/extractors/` | `tickets.py`, `ml_tickets.py` |
| External integrations | `src/gitflow_analytics/integrations/` | `github_client.py` |
| Metrics calculations | `src/gitflow_analytics/metrics/` | `dora.py` |
| Data models | `src/gitflow_analytics/models/` | `database.py` |
| ML components | `src/gitflow_analytics/qualitative/` | Organized by subdirectory |
| Report generators | `src/gitflow_analytics/reports/` | `csv_writer.py`, `narrative_writer.py` |

**Never in `src/`:**
- Test files (use `tests/`)
- Debug scripts (use `scripts/` or gitignore)
- Temporary analysis files (archive to `docs/_archive/`)

### Test Files (`tests/`)

**Rule:** Test files mirror the source structure

| Source File | Test File |
|-------------|-----------|
| `src/gitflow_analytics/core/analyzer.py` | `tests/core/test_analyzer.py` |
| `src/gitflow_analytics/config.py` | `tests/test_config.py` |
| `src/gitflow_analytics/qualitative/classifiers/change_type.py` | `tests/qualitative/test_change_type.py` |

**Test Naming Convention:**
- Prefix with `test_` for test modules: `test_config.py`
- Test functions use `test_` prefix: `def test_load_config():`
- Test directories mirror source without `test_` prefix: `tests/core/` matches `src/gitflow_analytics/core/`

**Never in root:**
- Temporary test scripts (prefix with `test_*.py` - auto-gitignored)
- Test reports (auto-generated in `test_reports/` - gitignored)

### Documentation (`docs/`)

**Rule:** All documentation lives in `docs/` with audience-specific organization

| Content Type | Location | Purpose |
|--------------|----------|---------|
| Getting started | `docs/getting-started/` | User onboarding |
| User guides | `docs/guides/` | Task-oriented instructions |
| Usage examples | `docs/examples/` | Real-world scenarios |
| Technical reference | `docs/reference/` | API docs, schemas, CLI reference |
| Developer docs | `docs/developer/` | Contributing, setup, training |
| Architecture | `docs/architecture/` | System design, patterns |
| Design documents | `docs/design/` | Technical decision records |
| Deployment | `docs/deployment/` | Production operations |
| Archives | `docs/_archive/` | Historical/deprecated content |

**Root-Level Documentation Files (Exceptions):**
- `README.md` - Project overview and quick start
- `CLAUDE.md` - AI assistant developer instructions
- `CONTRIBUTING.md` - Contribution guidelines (linked from docs)
- `CHANGELOG.md` - Automatically maintained by semantic-release
- `LICENSE` - Project license

**Documentation Files That Should Move:**
| Current Location | Correct Location | Reason |
|------------------|------------------|--------|
| `./code_analysis_report.md` | `docs/_archive/analysis-files/` | Temporary analysis document |
| `./IMPLEMENTATION_SUMMARY.md` | `docs/_archive/temp-files/` | Implementation notes (archived) |
| `./FEATURES_READY_FOR_TESTING.md` | `docs/_archive/temp-files/` | Testing notes (archived) |
| `./GIT_URL_CLONING_IMPLEMENTATION.md` | `docs/design/` or `docs/_archive/` | Design document or archive |
| `./TIMEZONE_BUG_RESOLUTION.md` | `docs/_archive/temp-files/` | Bug resolution notes (archived) |
| `./USAGE_EXAMPLES.md` | `docs/examples/` or consolidate with existing | Usage documentation |
| `./TEST_REPORT_GIT_CLONING.md` | `docs/_archive/temp-files/` | Test report (archived) |
| `./CHANGELOG_INTERACTIVE_LAUNCHER.md` | `docs/_archive/temp-files/` | Feature-specific changelog (merge into main CHANGELOG) |

### Configuration Files

**Rule:** Sample configs in `configs/`, active configs gitignored in root or custom locations

| File Type | Location | Tracked in Git? |
|-----------|----------|-----------------|
| Sample configurations | `configs/config-sample*.yaml` | Yes |
| Active configurations | Root or custom directory | No (gitignored) |
| Environment templates | `.env.example`, `.env.sample` | Yes |
| Active environment files | `.env` | No (gitignored) |

### Scripts and Utilities

**Rule:** Reusable scripts in `scripts/`, temporary scripts gitignored

| Script Type | Location | Example |
|-------------|----------|---------|
| Utility scripts | `scripts/` | `fix_identity_database.py` |
| Development helpers | `scripts/` | `validate_config.sh` |
| Debug scripts | Gitignored (prefix `debug_*.py`) | `debug_config.py`, `debug_env.py` |
| Test runners | Gitignored (prefix `run_*.py`) | `run_all_tests.py` |
| Ad-hoc analysis | Gitignored or `docs/_archive/` | `run_security_analysis.py` |

**Scripts in Root (Should be cleaned up):**
| Current File | Action | Reason |
|--------------|--------|--------|
| `debug_config.py` | Gitignored (already is) | Debug script |
| `debug_env.py` | Gitignored (already is) | Debug script |
| `run_all_tests.py` | Gitignored (already is) | Test runner |
| `run_security_analysis.py` | Move to `scripts/` or gitignore | Security analysis script |
| `run_security_all_repos.py` | Move to `scripts/` or gitignore | Security analysis script |
| `test_*.py` files | Gitignored (already is) | Temporary test scripts |
| `*.sh` scripts | Gitignored (already is) | Test/validation scripts |

### Build Artifacts and Caches

**Rule:** All generated files are gitignored and never committed

| Directory/File | Purpose | Gitignored? |
|----------------|---------|-------------|
| `build/` | Build artifacts | Yes |
| `dist/` | Distribution packages | Yes |
| `.gitflow-cache/` | Application cache | Yes |
| `.pytest_cache/` | Test cache | Yes |
| `.mypy_cache/` | Type checking cache | Yes |
| `.ruff_cache/` | Linter cache | Yes |
| `.qualitative_cache/` | ML cache | Yes |
| `htmlcov/` | Coverage reports | Yes |
| `reports/` | Generated reports | Yes |
| `*.db` files in root | Database files | Yes |

**Test/Debug Directories (Should not exist in version control):**
- `debug-cache/` - Gitignored
- `EWTN-test/` - Gitignored (project-specific test data)
- `kuzu-memories/` - Gitignored (MCP vector search data)
- `test_reports/`, `test-cache/` - Gitignored
- `test-qa-install/` - Gitignored
- `tickets/` - Gitignored (if not intentional)

---

## Naming Conventions

### Python Files

| Type | Convention | Example |
|------|------------|---------|
| Modules | `snake_case.py` | `identity_resolver.py` |
| Test modules | `test_*.py` | `test_identity.py` |
| Private modules | `_module.py` | `_version.py` |
| Debug scripts | `debug_*.py` (gitignored) | `debug_config.py` |

### Documentation Files

| Type | Convention | Example |
|------|------------|---------|
| Guide documents | `lowercase-with-hyphens.md` | `ml-categorization.md` |
| Reference docs | `lowercase-with-hyphens.md` | `configuration-schema.md` |
| Design docs | `lowercase-with-hyphens.md` | `commit-classification-design.md` |
| Root docs | `UPPERCASE.md` | `README.md`, `CONTRIBUTING.md` |

### Configuration Files

| Type | Convention | Example |
|------|------------|---------|
| Sample configs | `config-sample-*.yaml` | `config-sample.yaml`, `config-sample-ml.yaml` |
| Active configs | `config-*.yaml` (gitignored) | `config-production.yaml` |
| Environment templates | `.env.sample`, `.env.example` | `.env.sample` |
| Active environment | `.env` (gitignored) | `.env` |

### Directories

| Type | Convention | Example |
|------|------------|---------|
| Source directories | `snake_case/` | `qualitative/`, `extractors/` |
| Documentation sections | `lowercase-with-hyphens/` | `getting-started/`, `developer/` |
| Hidden directories | `.directory-name/` (usually gitignored) | `.gitflow-cache/`, `.pytest_cache/` |

---

## Framework-Specific Patterns

### Python Package (setuptools/pyproject.toml)

GitFlow Analytics follows modern Python packaging standards:

**Package Structure:**
```
src/
└── gitflow_analytics/          # Package name (importable)
    ├── __init__.py            # Package initialization
    ├── _version.py            # Version (managed by semantic-release)
    ├── cli.py                 # Click CLI entry point
    └── [modules]/             # Feature modules
```

**Testing Structure (pytest):**
```
tests/
├── __init__.py                # Makes tests a package
├── conftest.py                # Shared fixtures
├── test_*.py                  # Top-level test modules
└── [test_modules]/            # Test subdirectories mirror src/
```

**Configuration Files:**
- `pyproject.toml` - Modern Python project configuration (PEP 518)
- `pytest.ini` - Pytest-specific configuration
- `setup.py` - Legacy (no longer used, pyproject.toml only)

### CI/CD (GitHub Actions)

**Workflow Organization:**
```
.github/
└── workflows/
    ├── semantic-release.yml   # Main release workflow
    ├── tests.yml              # Testing on multiple Python versions
    └── release.yml            # Additional release validation
```

### Version Control (Git)

**Important Patterns:**
- Use `git mv` for moving files to preserve history
- Never commit files matching `.gitignore` patterns
- Keep `.gitignore` patterns organized by category

---

## Temporary Files Policy

### Files That Should NEVER Be Committed

Based on `.gitignore` patterns:

**Debug and Test Files:**
- `debug_*.py`, `fix_*.py` - Debug scripts
- `test_*.sh`, `run_*.sh`, `validate_*.sh` - Shell scripts
- `analyze_*.py`, `mock_*.py` - Analysis/mock scripts
- `*_validation.py` - Validation scripts

**Documentation Snapshots:**
- `*_ANALYSIS*.md`, `*_CLASSIFICATION*.md` - Analysis documents
- `*_IMPLEMENTATION*.md`, `*_REPORT*.md` - Implementation notes
- `*_SUMMARY.md`, `*_VALIDATION*.md` - Summary/validation docs
- `HOTFIX_*.md`, `RELEASE_*.md` - Release notes
- `COMMIT_MESSAGE*.md` - Commit drafts

**Generated Reports:**
- Files with timestamp patterns: `*_20[0-9][0-9][0-9][0-9][0-9][0-9]*.{csv,json,md}`
- `narrative_report_*.md`, `database_qualitative_report_*.md`
- `activity_distribution_*.csv`, `developer_focus_*.csv`

**Test Artifacts:**
- `test_reports/`, `test-cache/`, `.test-cache/`
- Test database files: `*.db` in tests/

### Migration to `docs/_archive/`

If a temporary file has historical value:
1. Move to appropriate `docs/_archive/` subdirectory
2. Add README.md in archive directory explaining contents
3. Remove from root directory
4. Update any references in documentation

**Archive Structure:**
```
docs/_archive/
├── old-reports/         # Historical test reports
├── old-logs/            # Historical log files
├── temp-files/          # Temporary implementation/analysis documents
└── analysis-files/      # One-off analysis documents
```

---

## Migration Procedures

### Moving Files Safely

When reorganizing files:

**1. Verify Git Status**
```bash
git status
# Ensure no uncommitted changes that might be lost
```

**2. Use Git Move (preserves history)**
```bash
git mv source_file.md destination/file.md
```

**3. Update References**
- Search for file references in documentation
- Update import statements if moving Python files
- Update relative links in markdown files

**4. Test After Migration**
```bash
# For Python files
pytest tests/

# For documentation
# Verify all links work
# Check examples still run
```

**5. Commit Changes**
```bash
git add .
git commit -m "docs: reorganize [file type] to follow PROJECT_ORGANIZATION.md"
```

### Batch Reorganization

For multiple files:

**1. Create Reorganization Plan**
```
Proposed Changes:
  docs/_archive/temp-files/
    ← IMPLEMENTATION_SUMMARY.md
    ← FEATURES_READY_FOR_TESTING.md
    ← TIMEZONE_BUG_RESOLUTION.md

  docs/design/
    ← GIT_URL_CLONING_IMPLEMENTATION.md (if current design doc)

  # Or alternatively to archive:
  docs/_archive/analysis-files/
    ← GIT_URL_CLONING_IMPLEMENTATION.md (if historical)
```

**2. Execute with Backup**
```bash
# Create backup
tar -czf backup_$(date +%Y%m%d).tar.gz .

# Run moves
git mv IMPLEMENTATION_SUMMARY.md docs/_archive/temp-files/
git mv FEATURES_READY_FOR_TESTING.md docs/_archive/temp-files/
# ... etc
```

**3. Verify and Commit**
```bash
git status
pytest tests/
git commit -m "docs: reorganize temporary files to _archive/"
```

---

## Maintenance and Updates

### Regular Organization Audits

Quarterly, review:
- [ ] Root directory for misplaced files
- [ ] `.gitignore` effectiveness
- [ ] `docs/_archive/` organization
- [ ] Test coverage for moved/renamed files

### When to Update This Document

Update `PROJECT_ORGANIZATION.md` when:
- Adding new top-level directories
- Changing organizational philosophy
- Discovering new patterns in framework usage
- Community feedback suggests improvements

**Update Process:**
1. Update this document
2. Update `docs/STRUCTURE.md` if documentation changes
3. Update `CLAUDE.md` to reference this document
4. Announce changes in CHANGELOG.md
5. Add version/date to header

---

## References

- [docs/STRUCTURE.md](../STRUCTURE.md) - Documentation organization details
- [CLAUDE.md](/CLAUDE.md) - Developer instructions (links to this document)
- [CONTRIBUTING.md](/CONTRIBUTING.md) - Contribution guidelines
- [.gitignore](/.gitignore) - Files excluded from version control

---

**Document Status:** Official Project Standard
**Enforcement:** All new contributions must follow these guidelines
**Questions?** Open an issue with tag `documentation` or `organization`

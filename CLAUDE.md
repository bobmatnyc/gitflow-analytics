# Claude Developer Instructions for GitFlow Analytics

This document provides specific instructions for Claude (AI assistant) when working on the GitFlow Analytics project. It ensures consistent development practices and helps maintain code quality.

## Project Overview

GitFlow Analytics is a Python package that analyzes Git repositories to generate developer productivity insights without requiring external project management tools. It provides comprehensive metrics including commit patterns, developer focus, ticket tracking, and DORA metrics.

## Documentation Structure

GitFlow Analytics uses a comprehensive documentation system organized for different audiences. All documentation is located in the `docs/` directory:

- **[docs/STRUCTURE.md](docs/STRUCTURE.md)** - Complete documentation organization guide
- **[docs/README.md](docs/README.md)** - Main documentation index and navigation
- **[docs/getting-started/](docs/getting-started/)** - New user onboarding and tutorials  
- **[docs/guides/](docs/guides/)** - Task-oriented configuration and usage guides
- **[docs/examples/](docs/examples/)** - Real-world usage scenarios and templates
- **[docs/reference/](docs/reference/)** - Technical specifications and API documentation
- **[docs/developer/](docs/developer/)** - Contribution guidelines and development setup
- **[docs/architecture/](docs/architecture/)** - System design and architectural decisions
- **[docs/design/](docs/design/)** - Design documents and technical decision records
- **[docs/deployment/](docs/deployment/)** - Production deployment and operations

### Documentation Guidelines for Developers

When working on the project:
- **Update docs with code changes** - Documentation should stay current with implementation
- **Follow the structure** - Place new documentation in the appropriate section
- **Cross-reference related topics** - Use relative links to connect related information
- **Test all examples** - Ensure code samples work and produce expected output
- **Update index files** - Keep section README.md files current when adding new content

### Documentation Audience Focus

- **User Documentation** (`getting-started/`, `guides/`, `examples/`) - External users and administrators
- **Developer Documentation** (`developer/`, `architecture/`, `design/`) - Contributors and maintainers  
- **Reference Documentation** (`reference/`) - Technical specifications for integration

## Key Development Guidelines

### 1. Code Quality Standards

When modifying code:
- **Always run linting and type checking** before committing:
  ```bash
  ruff check src/
  mypy src/
  black src/
  ```
- **Run tests** after making changes:
  ```bash
  pytest tests/
  ```
- **Follow existing code patterns** - check neighboring files for conventions

### 2. Identity Resolution

The project has a sophisticated developer identity resolution system:
- **Automatic Analysis**: Runs LLM-based identity analysis on first run (when no manual mappings exist)
- Handles multiple email addresses per developer
- Supports manual identity mappings in configuration
- Uses fuzzy matching with configurable threshold (default: 0.85)
- **Caching**: Identity analysis results are cached for 7 days to avoid re-running
- **Important**: When debugging identity issues, check `.gitflow-cache/identities.db`

#### Automatic Identity Analysis

The system now automatically analyzes developer identities **by default** when no manual mappings exist:
- Runs automatically during analysis (default command behavior)
- Prompts interactively for approval
- Updates configuration if approved
- Only prompts once every 7 days

```yaml
# Disable automatic analysis (enabled by default)
analysis:
  identity:
    auto_analysis: false
```

To skip identity analysis for a single run:
```bash
# Simplified syntax (default)
gitflow-analytics -c config.yaml --skip-identity-analysis

# Explicit analyze command
gitflow-analytics analyze -c config.yaml --skip-identity-analysis
```

To manually run identity analysis:
```bash
gitflow-analytics identities -c config.yaml
```

#### Display Name Control

Manual identity mappings now support an optional `name` field to control display names in reports:
```yaml
analysis:
  identity:
    manual_mappings:
      # Consolidate John Smith identities  
      - name: "John Smith"  # Controls how name appears in reports
        primary_email: "john.smith@company.com"
        aliases:
          - "150280367+jsmith@users.noreply.github.com"
          - "jsmith-company@users.noreply.github.com"
```

This feature resolves duplicate entries when the same developer appears with different name formats. The structure supports both `primary_email` (preferred) and `canonical_email` (backward compatibility).

### 3. Caching System

The project uses SQLite for caching:
- Commit cache: `.gitflow-cache/gitflow_cache.db`
- Identity cache: `.gitflow-cache/identities.db`
- **Always provide `--clear-cache` option** when testing configuration changes

### 4. Configuration Management

Configuration uses YAML with environment variable support:
- Variables use format: `${VARIABLE_NAME}`
- **Environment files**: Automatically loads `.env` file from same directory as config YAML
- **Organization support**: `github.organization` field enables automatic repository discovery
- **Directory defaults**: Cache and reports now default to config file directory (not current working directory)
- Default ticket platform can be specified
- Branch mapping rules for project inference
- Manual identity mappings for consolidating developer identities
- Full backward compatibility with existing repository-based configurations

### 5. Default Command Behavior

GitFlow Analytics now uses `analyze` as the default command when no subcommand is specified:

- **Simplified syntax**: `gitflow-analytics -c config.yaml --weeks 8`
- **Explicit command**: `gitflow-analytics analyze -c config.yaml --weeks 8` (backward compatible)
- **Improved UX**: Users can omit the `analyze` subcommand for the most common operation
- **Developer impact**: Update examples and documentation to use simplified syntax as primary

#### Using .env Files

The system automatically looks for a `.env` file in the same directory as your configuration YAML:
```bash
# Example .env file
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
JIRA_ACCESS_USER=your.email@company.com
JIRA_ACCESS_TOKEN=xxxxxxxxxxxxxxxxxxxx
```

This approach is recommended for:
- Keeping credentials out of configuration files
- Easy credential management across environments
- Preventing accidental credential commits

### 6. Report Generation

The system generates multiple report types:
- **CSV Reports**: Weekly metrics, developer stats, activity distribution, **untracked commits**
- **Markdown Reports**: Comprehensive narrative summaries with **restored multi-section format** and **enhanced untracked work analysis**
- **JSON Export**: Complete data export for API integration

#### Restored Narrative Report Sections

The narrative report system has been enhanced to include comprehensive sections:

**Core Sections (Always Present):**
- **Executive Summary**: High-level metrics, active projects list, top contributor
- **Team Composition**: Developer profiles with ALL project percentages, work styles, activity patterns
- **Project Activity**: Activity breakdown by project with contributor percentages within each project
- **Development Patterns**: Key insights from productivity and collaboration analysis
- **Issue Tracking**: Simplified platform usage display and coverage analysis
- **Enhanced Untracked Work Analysis**: Comprehensive categorization with dual percentage metrics
- **Recommendations**: Actionable insights based on analysis patterns

**Conditional Sections:**
- **Qualitative Analysis**: LLM-generated insights (when ChatGPT integration is available)
- **Pull Request Analysis**: PR metrics (when PR data is available)
- **PM Platform Integration**: Story point tracking and correlation insights (when PM data is available)

#### Developer Project Percentage Calculation

The narrative report shows ALL projects each developer works on with precise percentages:

```python
# In _write_team_composition(), the system:
# 1. Looks for both _dev_pct and _pct patterns in focus data
# 2. Shows ALL projects with percentages, not just the primary project
# 3. Sorts by percentage descending for clear priority indication

# Example output:
# Projects: FRONTEND (85.0%), SERVICE_TS (15.0%)
```

This enhancement provides complete visibility into developer work distribution across projects.

#### Simplified Issue Tracking Display

The issue tracking section has been streamlined:
- **Platform Usage**: Clean display without complex nested metrics
- **Coverage Analysis**: Simple commit-to-ticket ratio
- **Enhanced Untracked Analysis**: Detailed but focused on actionable insights

This approach matches the original report format while maintaining enhanced analytical depth.

#### Enhanced Untracked Commit Analysis

The project now includes sophisticated untracked commit analysis with several key enhancements:

**Automatic Commit Categorization**: Uses regex pattern matching to classify commits into categories:
- `bug_fix`: Error corrections and fixes
- `feature`: New functionality development
- `refactor`: Code restructuring and optimization
- `documentation`: Documentation updates
- `maintenance`: Routine upkeep and dependencies
- `test`: Testing-related changes
- `style`: Formatting and linting
- `build`: Build system and CI/CD changes
- `other`: Uncategorized commits

**Configurable File Threshold**: 
- Default threshold reduced from 3 to 1 file changed
- Configurable via `TicketExtractor` constructor parameter
- Filters out merge commits automatically
- Captures more granular untracked work patterns

**Enhanced Metadata Collection**:
- Full and abbreviated commit hashes
- Canonical developer identity resolution
- Project key for multi-repository analysis
- Detailed change metrics (files, lines added/removed)
- Commit categorization for pattern analysis
- Timestamp preservation for chronological analysis

**Dual Percentage Metrics**: 
- Percentage of total untracked work (developer's share of all untracked commits)
- Percentage of developer's individual work (proportion of their commits that are untracked)
- Provides context for process improvement recommendations

**Process Recommendations**: 
- Category-based recommendations (e.g., track features/bugs, accept maintenance)
- Developer-specific guidance identification
- Positive recognition for appropriate untracked work patterns

### 7. Testing Workflow

When testing changes:
1. Use the recess-recreo repositories as test data
2. Run with `--weeks 8` for consistent test periods
3. Check all report outputs for correctness
4. Verify identity resolution is working properly

### 8. Common Tasks

#### Adding a New Report Type

1. Create report generator in `src/gitflow_analytics/reports/`
2. Add to report generation pipeline in `cli.py`
3. Update configuration to support format selection
4. Document the report format in README

#### Working with ML-Enhanced Commit Categorization

1. **Implementation Architecture**: The ML categorization system is built on top of the existing `TicketExtractor`:
   - `MLTicketExtractor` extends `TicketExtractor` with ML capabilities
   - Uses the existing qualitative analysis infrastructure (`ChangeTypeClassifier`)
   - Maintains full backward compatibility with existing reports and configurations

2. **Hybrid Classification Approach**:
   ```python
   # The system tries ML first, falls back to rules
   if ml_confidence >= hybrid_threshold:
       return ml_category
   else:
       return rule_based_category
   ```

3. **Key Components**:
   - **MLTicketExtractor**: Main entry point, extends TicketExtractor
   - **ChangeTypeClassifier**: Core ML logic using spaCy and semantic analysis
   - **MLPredictionCache**: SQLite-based caching for performance
   - **Semantic patterns**: Extensible keyword patterns for each category

4. **Performance Optimization**:
   - **Caching**: ML predictions cached in SQLite for repeat analysis
   - **Batch processing**: Commits processed in configurable batches
   - **Lazy loading**: spaCy models loaded only when needed
   - **Graceful degradation**: Falls back to rule-based if ML fails

5. **Testing ML Categorization**:
   ```python
   from gitflow_analytics.extractors.ml_tickets import MLTicketExtractor
   
   # Create extractor with ML enabled
   extractor = MLTicketExtractor(enable_ml=True)
   
   # Test categorization with confidence
   result = extractor.categorize_commit_with_confidence(
       "fix: resolve memory leak in cache cleanup",
       files_changed=["src/cache.py"]
   )
   
   print(f"Category: {result['category']}")
   print(f"Confidence: {result['confidence']}")
   print(f"Method: {result['method']}")  # 'ml', 'rules', or 'cached'
   ```

6. **Extending Categories**: Add new semantic patterns to `ChangeTypeClassifier.change_patterns`
7. **Configuration**: ML behavior controlled via `ml_categorization` config section
8. **Debugging**: Use `get_ml_statistics()` method for performance insights

#### Working with Untracked Commit Analysis

1. **Extending Categorization**: Add new category patterns to `TicketExtractor.category_patterns`
2. **Customizing Thresholds**: Modify `untracked_file_threshold` in `TicketExtractor` constructor
3. **Adding Recommendations**: Extend `_generate_untracked_recommendations()` method
4. **Testing Categories**: Use `categorize_commit()` method to test pattern matching
5. **Report Customization**: Modify `generate_untracked_commits_report()` for additional fields

#### Adding a New Ticket Platform

1. Update regex patterns in `TicketExtractor`
2. Add platform to ticket counting logic
3. Test with sample commit messages
4. Update documentation

#### Debugging Identity Issues

1. Check identity database:
   ```bash
   sqlite3 .gitflow-cache/identities.db "SELECT * FROM developer_identities"
   ```
2. Review manual mappings in config
3. Clear cache and re-run analysis
4. Check for typos in email addresses

#### Debugging Narrative Report Generation

1. **Check Report Sections**: Verify all expected sections are generated:
   ```python
   # Debug narrative report generation
   from gitflow_analytics.reports.narrative_writer import NarrativeReportGenerator
   generator = NarrativeReportGenerator()
   
   # Check if PR data is available for PR analysis section
   if pr_metrics and pr_metrics.get('total_prs', 0) > 0:
       print("PR Analysis section will be included")
   
   # Check if PM data is available for PM integration section  
   if pm_data and 'metrics' in pm_data:
       print("PM Platform Integration section will be included")
   ```

2. **Debug Developer Project Percentages**: Ensure focus data is correctly formatted:
   ```python
   # Check focus data structure for developer projects
   for dev in focus_data:
       print(f"Developer: {dev['developer']}")
       # Look for _dev_pct or _pct patterns
       project_keys = [k for k in dev.keys() if k.endswith('_dev_pct') or k.endswith('_pct')]
       print(f"Project percentage keys: {project_keys}")
   ```

3. **Validate Untracked Analysis Data**: Check untracked commits structure:
   ```python
   # Debug untracked commits analysis
   for commit in untracked_commits[:5]:  # Check first 5
       required_fields = ['hash', 'author', 'message', 'category', 'project_key']
       missing = [f for f in required_fields if f not in commit]
       if missing:
           print(f"Missing fields in commit {commit.get('hash', 'unknown')}: {missing}")
   ```

4. **Test Report Template Rendering**: Verify template string formatting:
   ```python
   # Test narrative templates
   generator = NarrativeReportGenerator()
   test_template = generator.templates['high_performer']
   result = test_template.format(name="Test Dev", commits=10, pct=25.5)
   print(f"Template result: {result}")
   ```

#### Debugging Untracked Commit Analysis

1. **Review Categorization Patterns**: Check if commit messages match expected patterns
   ```python
   # Example: Debug category matching
   from gitflow_analytics.extractors.tickets import TicketExtractor
   extractor = TicketExtractor()
   category = extractor.categorize_commit("fix: resolve login bug")
   # Should return "bug_fix"
   ```

2. **Inspect Untracked Commits CSV**: Review the generated `untracked_commits_YYYYMMDD.csv` for:
   - Unexpected categorizations
   - Missing commits (check file threshold)
   - Merge commits incorrectly included

3. **Validate File Threshold**: Adjust `untracked_file_threshold` parameter:
   ```python
   # Include all commits (threshold = 1)
   extractor = TicketExtractor(untracked_file_threshold=1)
   
   # Only significant commits (threshold = 3)
   extractor = TicketExtractor(untracked_file_threshold=3)
   ```

4. **Check Ticket Reference Detection**: Verify ticket patterns are matching correctly:
   ```bash
   # Search for commits that should have tickets but don't
   grep -E "(PROJ-|#[0-9]|CU-)" commit_messages.txt
   ```

5. **Review Developer Mapping**: Ensure canonical IDs are resolving correctly for dual percentage calculations

#### Working with Organization Support

1. **Organization Discovery**: When `github.organization` is specified and no repositories are manually configured:
   - All non-archived repositories are automatically discovered from the GitHub organization
   - Repositories are cloned to local directories if they don't exist
   - Uses the organization name as the project key prefix if not specified

2. **Testing Organization Configs**:
   ```bash
   # Test with organization discovery (simplified syntax)
   gitflow-analytics -c config-org.yaml --weeks 4 --validate-only
   
   # Run with discovered repositories (simplified syntax)
   gitflow-analytics -c config-org.yaml --weeks 4
   
   # Explicit analyze command (backward compatibility)
   gitflow-analytics analyze -c config-org.yaml --weeks 4
   ```

3. **Directory Structure**: With organization support, the recommended directory structure is:
   ```
   /project/
   ├── config-org.yaml       # Organization config
   ├── repos/                # Auto-cloned repositories
   │   ├── repo1/
   │   ├── repo2/
   │   └── repo3/
   ├── .gitflow-cache/       # Cache (relative to config)
   └── reports/              # Reports (default output location)
   ```

4. **Debugging Organization Discovery**:
   - Check GitHub token has organization read permissions
   - Verify organization name is correct (case-sensitive)
   - Use `--validate-only` to test configuration without full analysis
   - Check for API rate limiting issues

### 9. Performance Considerations

- **Batch processing**: Commits are processed in batches (default: 1000)
- **Progress bars**: Use tqdm for long operations
- **Caching**: Aggressive caching to avoid re-processing
- **Memory usage**: Be mindful with large repositories

### 10. Error Handling

- **GitHub API errors**: Handle rate limiting and authentication failures gracefully
- **File system errors**: Check permissions and paths
- **Database locks**: Use proper session management with SQLAlchemy
- **Configuration errors**: Provide helpful error messages

### 11. Documentation Updates

When adding features:
1. Update README.md with user-facing changes
2. Update this file (CLAUDE.md) with developer notes
3. Add docstrings to all new functions/classes
4. Update configuration examples if needed

## Project Structure

```
gitflow-analytics/
├── src/gitflow_analytics/
│   ├── __init__.py          # Package initialization
│   ├── _version.py          # Version information
│   ├── cli.py               # CLI entry point
│   ├── config.py            # Configuration handling
│   ├── core/                # Core analysis logic
│   │   ├── analyzer.py      # Git analysis
│   │   ├── branch_mapper.py # Branch to project mapping
│   │   ├── cache.py         # Caching system
│   │   └── identity.py      # Developer identity resolution
│   ├── extractors/          # Data extraction
│   │   ├── story_points.py  # Story point extraction
│   │   ├── tickets.py       # Ticket reference extraction (rule-based)
│   │   └── ml_tickets.py    # ML-enhanced ticket extraction
│   ├── integrations/        # External integrations
│   │   └── github_client.py # GitHub API client
│   ├── metrics/             # Metric calculations
│   │   └── dora.py          # DORA metrics
│   ├── models/              # Data models
│   │   └── database.py      # SQLAlchemy models
│   ├── qualitative/         # ML and qualitative analysis
│   │   ├── classifiers/     # ML classifiers
│   │   │   ├── change_type.py    # Commit categorization ML
│   │   │   ├── domain_classifier.py
│   │   │   ├── intent_analyzer.py
│   │   │   └── risk_analyzer.py
│   │   ├── core/            # Core ML infrastructure
│   │   │   ├── nlp_engine.py     # spaCy integration
│   │   │   ├── processor.py      # Batch processing
│   │   │   ├── pattern_cache.py  # Pattern learning cache
│   │   │   └── llm_fallback.py   # LLM fallback support
│   │   ├── models/          # Data schemas
│   │   │   └── schemas.py        # Configuration schemas
│   │   └── utils/           # Utilities
│   │       ├── text_processing.py
│   │       ├── batch_processor.py
│   │       ├── cost_tracker.py
│   │       └── metrics.py
│   └── reports/             # Report generation
│       ├── analytics_writer.py
│       ├── csv_writer.py
│       └── narrative_writer.py
├── tests/                   # Test suite
│   └── qualitative/        # ML system tests
├── docs/                    # Documentation
│   ├── design/              # Design documents
│   └── DEPLOY.md            # Deployment guide
├── config-sample.yaml       # Sample configuration
├── config-sample-ml.yaml    # ML configuration sample
├── pyproject.toml           # Project metadata
└── README.md                # User documentation
```

## Atomic Versioning System

The project uses **python-semantic-release** for automated, atomic version management. This ensures consistency between the source code version, git tags, PyPI releases, and GitHub releases.

### Version Source of Truth

The single source of truth for versioning is:
- **File**: `src/gitflow_analytics/_version.py`
- **Format**: `__version__ = "X.Y.Z"`
- **Management**: Automatically updated by semantic-release

### Semantic Versioning Rules

Version bumps are determined by conventional commit messages:
- **MAJOR** (X.0.0): Breaking changes (rare, manual intervention required)
- **MINOR** (0.X.0): New features (`feat:` commits)
- **PATCH** (0.0.X): Bug fixes and maintenance (`fix:`, `docs:`, `chore:`, etc.)

### Conventional Commit Format

Use these prefixes for automatic version detection:
```bash
# Minor version bump (new features)
feat: add new story point extraction pattern
feat(cli): add --validate-only flag for configuration testing

# Patch version bump (fixes, improvements, maintenance)
fix: resolve identity resolution bug with similar names
fix(cache): handle database lock errors gracefully
docs: update installation instructions
chore: update dependency versions
style: fix code formatting issues
refactor: improve error handling in GitHub client
perf: optimize commit batch processing
test: add integration tests for organization discovery
ci: update GitHub Actions workflow
build: update pyproject.toml configuration
```

### CLI Version Display Fix

The project includes a fix for proper CLI version display:
- Version is dynamically imported from `_version.py`
- CLI command `gitflow-analytics --version` correctly shows current version
- Prevents version mismatch between package and CLI

## Automated Release Process

The project uses GitHub Actions for fully automated releases:

### Workflow Triggers
1. **Push to main branch**: Triggers semantic analysis
2. **Manual dispatch**: Can be triggered manually via GitHub UI
3. **Conventional commits**: Drive version bump decisions

### Release Steps (Automated)
1. **Semantic Analysis**: Analyze commits since last release
2. **Version Calculation**: Determine next version based on commit types
3. **Version Update**: Update `_version.py` with new version
4. **Git Operations**: Create git tag and GitHub release
5. **Quality Checks**: Run full test suite, linting, and type checking
6. **Package Build**: Build wheel and source distributions
7. **PyPI Publishing**: Automatically publish to PyPI
8. **Asset Upload**: Attach build artifacts to GitHub release
9. **Changelog**: Auto-generate and update CHANGELOG.md

### Manual Override (Emergency Only)

For emergency releases or version fixes:
```bash
# Only if semantic-release is not working
git checkout main
git pull origin main
# Edit src/gitflow_analytics/_version.py manually
git commit -m "chore(release): manual version bump to X.Y.Z"
git tag -a vX.Y.Z -m "Emergency release X.Y.Z"
git push origin main --tags
```

### GitHub Actions Configuration

The project includes comprehensive CI/CD:
- **`.github/workflows/semantic-release.yml`**: Main release workflow
- **`.github/workflows/tests.yml`**: Testing on multiple Python versions
- **`.github/workflows/release.yml`**: Additional release validation

### PyPI Publishing

- **Trusted Publishing**: Uses GitHub's OIDC for secure PyPI publishing
- **No API Keys**: No need to manage PyPI tokens in secrets
- **Automatic**: Publishing happens on every version tag creation

### 12. YAML Configuration Error Handling

The project provides friendly YAML configuration error messages to help users quickly fix common issues.

#### Implementation Details

The `ConfigLoader` class in `config.py` includes comprehensive YAML error handling:

```python
def _handle_yaml_error(self, error: yaml.YAMLError, config_path: Path) -> str:
    """Generate user-friendly error messages for YAML parsing errors."""
```

The error handler detects and provides specific guidance for:
- **Tab characters** - Most common YAML error
- **Missing colons** - After key names
- **Unclosed quotes** - String delimiter issues
- **Invalid indentation** - Inconsistent spacing
- **Invalid escape sequences** - In string values

#### Extending Error Handling

To add new error patterns:

1. Add detection logic in `_handle_yaml_error()`:
```python
if "new_error_pattern" in str(error).lower():
    return self._format_error_message(
        config_path, error,
        "🚫 New error description!",
        "💡 Fix: Specific fix instructions"
    )
```

2. Add test cases in `tests/test_config.py`:
```python
def test_new_error_type(self):
    yaml_content = '''your test case'''
    # Test implementation
```

#### Error Message Format

All YAML error messages follow this structure:
- ❌ Clear error indicator
- 🚫 Specific problem identification
- 💡 Actionable fix suggestion
- 📍 Context from YAML parser
- 📁 File location
- 🔗 Help resources

## Common Gotchas

1. **Timezone issues**: GitHub API returns timezone-aware timestamps, while database-stored datetimes may be timezone-naive. The system now ensures all datetime comparisons use UTC-aware timestamps
2. **Branch detection**: Simplified branch detection may not work for all workflows
3. **Memory usage**: Large repositories can consume significant memory
4. **Identity resolution**: Manual mappings must be applied after initial analysis
5. **Cache invalidation**: Some changes require clearing the cache
6. **Directory defaults**: Cache and reports now default to config file directory, not current working directory
7. **Organization permissions**: GitHub token must have organization read access for automatic repository discovery
8. **Version conflicts**: Never manually edit version in `_version.py` unless bypassing semantic-release
9. **Commit message format**: Incorrect commit message format will not trigger version bumps
10. **Release permissions**: Only repository owners can trigger releases (configured in workflow)
11. **Untracked commit threshold**: Default file threshold changed from 3 to 1 - may increase untracked commit counts
12. **Category matching**: Commit categorization is case-insensitive and uses regex - test patterns carefully
13. **Dual percentage metrics**: Require proper developer identity resolution to calculate individual work percentages
14. **Merge commit filtering**: Merge commits are automatically excluded from untracked analysis regardless of file changes
15. **ML model dependencies**: spaCy models must be downloaded separately (`python -m spacy download en_core_web_sm`)
16. **ML graceful degradation**: System falls back to rule-based categorization if ML components fail to load
17. **ML cache location**: ML predictions cached in `.gitflow-cache/ml_predictions.db` - clear cache to reset ML results
18. **ML confidence thresholds**: Low confidence thresholds may result in poor categorizations, high thresholds fall back to rules more often
19. **spaCy model size**: Larger models (md, lg) provide better accuracy but use more memory and are slower to load
20. **Hybrid threshold tuning**: The `hybrid_threshold` balances ML vs rule-based usage - tune based on your commit message patterns
21. **YAML tab characters**: YAML files cannot contain tab characters - use spaces for indentation (2 or 4 spaces)
22. **YAML error messages**: Enhanced error handling provides specific fix suggestions for common YAML syntax errors
23. **Configuration validation**: Beyond YAML syntax, the system validates required fields and configuration structure
24. **Narrative report sections**: Some sections are conditional on data availability - missing PR data means no PR analysis section
25. **Developer project percentages**: Require proper focus data calculation - check for _dev_pct or _pct patterns in focus data
26. **Issue tracking display**: Platform usage section only appears when ticket_analysis contains ticket_summary data
27. **PM platform integration section**: Only included when pm_data is provided and contains 'metrics' key
28. **Untracked work recommendations**: Generated dynamically based on category analysis - categories dict must be properly populated

## Quick Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# IMPORTANT: After making code changes, reinstall locally to test
pipx uninstall gitflow-analytics && pipx install /Users/masa/Projects/managed/gitflow-analytics

# Run analysis on test repos (simplified syntax - default behavior)
gitflow-analytics --config config-recess.yaml --weeks 8

# Clear cache and re-run (simplified syntax)
gitflow-analytics --config config-recess.yaml --weeks 8 --clear-cache

# Explicit analyze command (backward compatibility)
gitflow-analytics analyze --config config-recess.yaml --weeks 8

# Run tests with coverage
pytest --cov=gitflow_analytics --cov-report=html

# Format code
black src/ tests/

# Check code quality
ruff check src/
mypy src/

# Version and release commands (automated via CI/CD)
semantic-release version --dry-run  # Preview next version
semantic-release version           # Create release (CI only)
gitflow-analytics --version       # Check current version

# ML categorization commands
python -m spacy download en_core_web_sm  # Install spaCy model for ML
python -m spacy list                     # List installed spaCy models

# Test ML categorization
python -c "
from gitflow_analytics.extractors.ml_tickets import MLTicketExtractor
extractor = MLTicketExtractor()
result = extractor.categorize_commit_with_confidence('fix: resolve memory leak')
print(f'Category: {result[\"category\"]}, Confidence: {result[\"confidence\"]:.2f}')
"
```

## Contact

For questions about development practices or architecture decisions, refer to:
- Design documents in `docs/design/`
- GitHub issues for bug reports
- Pull request discussions for feature proposals
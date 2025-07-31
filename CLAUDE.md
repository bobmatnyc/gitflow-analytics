# Claude Developer Instructions for GitFlow Analytics

This document provides specific instructions for Claude (AI assistant) when working on the GitFlow Analytics project. It ensures consistent development practices and helps maintain code quality.

## Project Overview

GitFlow Analytics is a Python package that analyzes Git repositories to generate developer productivity insights without requiring external project management tools. It provides comprehensive metrics including commit patterns, developer focus, ticket tracking, and DORA metrics.

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
- Handles multiple email addresses per developer
- Supports manual identity mappings in configuration
- Uses fuzzy matching with configurable threshold (default: 0.85)
- **Important**: When debugging identity issues, check `.gitflow-cache/identities.db`

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

### 5. Report Generation

The system generates multiple report types:
- **CSV Reports**: Weekly metrics, developer stats, activity distribution
- **Markdown Reports**: Narrative summaries with insights
- **JSON Export**: Complete data export for API integration

### 6. Testing Workflow

When testing changes:
1. Use the recess-recreo repositories as test data
2. Run with `--weeks 8` for consistent test periods
3. Check all report outputs for correctness
4. Verify identity resolution is working properly

### 7. Common Tasks

#### Adding a New Report Type

1. Create report generator in `src/gitflow_analytics/reports/`
2. Add to report generation pipeline in `cli.py`
3. Update configuration to support format selection
4. Document the report format in README

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

#### Working with Organization Support

1. **Organization Discovery**: When `github.organization` is specified and no repositories are manually configured:
   - All non-archived repositories are automatically discovered from the GitHub organization
   - Repositories are cloned to local directories if they don't exist
   - Uses the organization name as the project key prefix if not specified

2. **Testing Organization Configs**:
   ```bash
   # Test with organization discovery
   gitflow-analytics analyze -c config-org.yaml --weeks 4 --validate-only
   
   # Run with discovered repositories
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

### 8. Performance Considerations

- **Batch processing**: Commits are processed in batches (default: 1000)
- **Progress bars**: Use tqdm for long operations
- **Caching**: Aggressive caching to avoid re-processing
- **Memory usage**: Be mindful with large repositories

### 9. Error Handling

- **GitHub API errors**: Handle rate limiting and authentication failures gracefully
- **File system errors**: Check permissions and paths
- **Database locks**: Use proper session management with SQLAlchemy
- **Configuration errors**: Provide helpful error messages

### 10. Documentation Updates

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
│   │   └── tickets.py       # Ticket reference extraction
│   ├── integrations/        # External integrations
│   │   └── github_client.py # GitHub API client
│   ├── metrics/             # Metric calculations
│   │   └── dora.py          # DORA metrics
│   ├── models/              # Data models
│   │   └── database.py      # SQLAlchemy models
│   └── reports/             # Report generation
│       ├── analytics_writer.py
│       ├── csv_writer.py
│       └── narrative_writer.py
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── design/              # Design documents
│   └── DEPLOY.md            # Deployment guide
├── config-sample.yaml       # Sample configuration
├── pyproject.toml           # Project metadata
└── README.md                # User documentation
```

## Version Management

- Version is stored in `src/gitflow_analytics/_version.py`
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version before releases
- Tag releases with `v` prefix (e.g., `v1.0.0`)

## Release Process

1. Update version in `_version.py`
2. Run full test suite
3. Update CHANGELOG.md
4. Commit with message: `chore: bump version to X.Y.Z`
5. Tag the release: `git tag -a vX.Y.Z -m "Release version X.Y.Z"`
6. Push tags: `git push origin main --tags`
7. Build and publish to PyPI (see DEPLOY.md)

## Common Gotchas

1. **Timezone issues**: GitHub API returns timezone-aware timestamps
2. **Branch detection**: Simplified branch detection may not work for all workflows
3. **Memory usage**: Large repositories can consume significant memory
4. **Identity resolution**: Manual mappings must be applied after initial analysis
5. **Cache invalidation**: Some changes require clearing the cache
6. **Directory defaults**: Cache and reports now default to config file directory, not current working directory
7. **Organization permissions**: GitHub token must have organization read access for automatic repository discovery

## Quick Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run analysis on test repos
gitflow-analytics analyze --config config-recess.yaml --weeks 8

# Clear cache and re-run
gitflow-analytics analyze --config config-recess.yaml --weeks 8 --clear-cache

# Run tests with coverage
pytest --cov=gitflow_analytics --cov-report=html

# Format code
black src/ tests/

# Check code quality
ruff check src/
mypy src/
```

## Contact

For questions about development practices or architecture decisions, refer to:
- Design documents in `docs/design/`
- GitHub issues for bug reports
- Pull request discussions for feature proposals
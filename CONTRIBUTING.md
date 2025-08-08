# Contributing to GitFlow Analytics

Thank you for your interest in contributing to GitFlow Analytics! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/gitflow-analytics.git
   cd gitflow-analytics
   ```

2. **Set up development environment:**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install spaCy model for ML features
   python -m spacy download en_core_web_sm
   ```

3. **Verify installation:**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code quality
   ruff check src/
   mypy src/
   black --check src/
   ```

### Development Commands

```bash
# Code formatting
black src/ tests/

# Linting
ruff check src/ tests/
ruff check --fix src/ tests/  # Auto-fix issues

# Type checking
mypy src/

# Run tests with coverage
pytest --cov=gitflow_analytics --cov-report=html

# Test specific module
pytest tests/test_analyzer.py -v

# Install local development version
pip install -e ".[dev]"
```

## 📝 Code Style and Standards

### Code Quality Requirements

All contributions must pass the following checks:

1. **Formatting**: Code must be formatted with `black`
2. **Linting**: Code must pass `ruff` checks
3. **Type Hints**: New code should include type hints and pass `mypy`
4. **Tests**: New features must include tests with >80% coverage
5. **Documentation**: Public APIs must have docstrings

### Pre-commit Setup (Recommended)

Install pre-commit hooks to automatically check code quality:

```bash
# Install pre-commit
pip install pre-commit

# Set up pre-commit hooks
pre-commit install

# Run manually (optional)
pre-commit run --all-files
```

### Code Style Guidelines

- **Line Length**: Maximum 88 characters (black default)
- **Import Organization**: Use `isort` compatible imports
- **Naming Conventions**: 
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Private members: `_leading_underscore`
- **Docstrings**: Use Google-style docstrings for all public functions

#### Example Function Style

```python
def analyze_repository(
    repo_path: Path,
    weeks: int = 12,
    enable_ml: bool = True
) -> AnalysisResult:
    """Analyze a Git repository for productivity insights.
    
    Args:
        repo_path: Path to the Git repository to analyze
        weeks: Number of weeks to analyze (default: 12)
        enable_ml: Enable ML-enhanced commit categorization
        
    Returns:
        AnalysisResult containing comprehensive metrics and insights
        
    Raises:
        RepositoryError: If the repository is invalid or inaccessible
        ConfigurationError: If analysis configuration is invalid
    """
    # Implementation here
    pass
```

## 🧪 Testing Guidelines

### Test Structure

Tests are organized in the `tests/` directory mirroring the `src/` structure:

```
tests/
├── test_analyzer.py           # Core analysis tests
├── test_config.py             # Configuration tests  
├── test_identity.py           # Identity resolution tests
├── qualitative/               # ML system tests
│   ├── test_classifiers.py
│   └── test_nlp_engine.py
└── fixtures/                  # Test data and fixtures
```

### Writing Tests

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-end Tests**: Test complete workflows

Example test structure:

```python
import pytest
from pathlib import Path
from gitflow_analytics.core.analyzer import GitAnalyzer

class TestGitAnalyzer:
    """Test suite for GitAnalyzer class."""
    
    @pytest.fixture
    def sample_repo(self) -> Path:
        """Provide a sample repository for testing."""
        return Path("tests/fixtures/sample-repo")
    
    def test_analyze_commits(self, sample_repo):
        """Test basic commit analysis functionality."""
        analyzer = GitAnalyzer(sample_repo)
        result = analyzer.analyze_commits(weeks=4)
        
        assert result.total_commits > 0
        assert len(result.developers) > 0
        assert result.date_range.weeks == 4
    
    def test_invalid_repository(self):
        """Test handling of invalid repository paths."""
        with pytest.raises(RepositoryError):
            GitAnalyzer(Path("nonexistent/path"))
```

### Test Data and Fixtures

- Use the `tests/fixtures/` directory for sample repositories and data
- Create minimal, focused test repositories
- Use `pytest.fixtures` for reusable test setup
- Mock external APIs (GitHub, JIRA) in tests

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gitflow_analytics --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py

# Run with verbose output
pytest -v

# Run only failed tests from last run
pytest --lf

# Run tests matching pattern
pytest -k "test_identity"
```

## 🏗️ Architecture and Design

### Project Structure

```
src/gitflow_analytics/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface
├── config.py                # Configuration handling
├── core/                    # Core analysis logic
│   ├── analyzer.py          # Main Git analysis engine
│   ├── cache.py             # Caching system
│   ├── identity.py          # Developer identity resolution
│   └── branch_mapper.py     # Branch to project mapping
├── extractors/              # Data extraction components
│   ├── tickets.py           # Ticket reference extraction (rule-based)
│   ├── ml_tickets.py        # ML-enhanced ticket extraction
│   └── story_points.py      # Story point extraction
├── integrations/            # External service integrations
│   └── github_client.py     # GitHub API integration
├── qualitative/             # ML and qualitative analysis
│   ├── classifiers/         # ML classification models
│   ├── core/                # Core ML infrastructure
│   └── utils/               # ML utilities
├── reports/                 # Report generation
│   ├── csv_writer.py        # CSV report generation
│   └── narrative_writer.py  # Markdown narrative reports
└── models/                  # Data models and schemas
    └── database.py          # SQLAlchemy models
```

### Design Principles

1. **Modular Architecture**: Each component has a single responsibility
2. **Extensibility**: Easy to add new ticket platforms, report formats, ML models
3. **Performance**: Intelligent caching and batch processing for large repositories
4. **Graceful Degradation**: ML features fall back to rule-based approaches
5. **Configuration-driven**: Behavior controlled through YAML configuration

### Adding New Features

#### Adding a New Ticket Platform

1. **Update regex patterns** in `extractors/tickets.py`:
   ```python
   # Add new platform pattern
   PLATFORM_PATTERNS = {
       'existing_platform': r'PROJ-\d+',
       'new_platform': r'NEW-\d+',  # Add here
   }
   ```

2. **Add platform detection logic**:
   ```python
   def detect_platform(ticket_ref: str) -> str:
       if re.match(r'NEW-\d+', ticket_ref):
           return 'new_platform'
       # ... existing logic
   ```

3. **Update configuration schema** if needed
4. **Add tests** for the new platform
5. **Update documentation**

#### Adding a New Report Format

1. **Create report writer** in `reports/`:
   ```python
   # reports/json_writer.py
   class JSONReportWriter:
       def generate_report(self, data: AnalysisResult) -> None:
           # Implementation
   ```

2. **Register in CLI** (`cli.py`):
   ```python
   # Add to format options
   if 'json' in output_formats:
       json_writer = JSONReportWriter()
       json_writer.generate_report(analysis_result)
   ```

3. **Update configuration schema**
4. **Add tests**
5. **Update documentation**

#### Extending ML Categorization

1. **Add new categories** in `qualitative/classifiers/change_type.py`:
   ```python
   CHANGE_PATTERNS = {
       'existing_category': ['pattern1', 'pattern2'],
       'new_category': ['new_pattern1', 'new_pattern2'],  # Add here
   }
   ```

2. **Update semantic analysis** if needed
3. **Train/validate** on sample data
4. **Add tests** with expected categorizations
5. **Update documentation**

## 🔄 Contribution Workflow

### Step 1: Planning

1. **Check existing issues** for similar features/bugs
2. **Create an issue** to discuss the change (for significant features)
3. **Get feedback** from maintainers before starting work

### Step 2: Development

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/awesome-new-feature
   # or
   git checkout -b fix/important-bug-fix
   ```

2. **Make atomic commits** with clear messages:
   ```bash
   git commit -m "feat: add support for Linear ticket platform"
   git commit -m "fix: resolve identity resolution bug with similar names"
   git commit -m "docs: update installation instructions for ML features"
   ```

3. **Follow conventional commits** for automatic versioning:
   - `feat:` - New features (minor version bump)
   - `fix:` - Bug fixes (patch version bump)
   - `docs:` - Documentation changes
   - `test:` - Adding or updating tests
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `chore:` - Maintenance tasks

### Step 3: Quality Checks

Run all quality checks before submitting:

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/

# Run tests
pytest --cov=gitflow_analytics

# Test installation
pip install -e ".[dev]"
```

### Step 4: Pull Request

1. **Create a pull request** with:
   - Clear title following conventional commit format
   - Detailed description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

2. **PR Template**:
   ```markdown
   ## Summary
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (fix/feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Added tests for new functionality
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No new warnings introduced
   ```

3. **Respond to feedback** promptly and make requested changes

## 🐛 Reporting Bugs

### Bug Report Template

When reporting bugs, please include:

1. **Environment**:
   - Python version
   - GitFlow Analytics version
   - Operating system
   - Git version

2. **Configuration** (redacted):
   - YAML configuration (remove sensitive tokens)
   - Command line arguments used

3. **Expected vs Actual Behavior**:
   - What you expected to happen
   - What actually happened
   - Error messages or logs

4. **Reproduction Steps**:
   - Minimal steps to reproduce the issue
   - Sample repository if possible (or description)

5. **Additional Context**:
   - Repository size/characteristics
   - Any workarounds found

### Example Bug Report

```markdown
**Bug**: ML categorization fails with spaCy model error

**Environment**:
- Python 3.11.5
- GitFlow Analytics v1.0.9  
- macOS 14.0
- Git 2.42.0

**Configuration**:
```yaml
analysis:
  ml_categorization:
    enabled: true
    min_confidence: 0.7
```

**Expected**: Commits categorized using ML model
**Actual**: Error: "OSError: [E050] Can't find model 'en_core_web_sm'"

**Steps**:
1. Install gitflow-analytics
2. Run without installing spaCy model
3. Error occurs during ML categorization

**Solution**: Better error message suggesting spaCy model installation
```

## 🌟 Feature Requests

### Feature Request Template

1. **Use Case**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Impact**: Who benefits from this feature?
5. **Implementation**: Any implementation suggestions

## 📚 Documentation Contributions

### Types of Documentation

1. **User Documentation**: README, usage guides, configuration examples
2. **Developer Documentation**: CLAUDE.md, API documentation, architecture guides
3. **API Documentation**: Docstrings, type hints
4. **Examples**: Sample configurations, use cases

### Documentation Standards

- Use clear, concise language
- Include practical examples
- Keep configuration examples generic (no company-specific references)
- Test all code examples
- Use proper markdown formatting
- Include screenshots/diagrams where helpful

### Documentation Workflow

1. **Identify documentation needs** (outdated guides, missing examples)
2. **Create or update content** following project standards
3. **Test examples** to ensure they work
4. **Submit pull request** with documentation changes

## 🏷️ Release Process

GitFlow Analytics uses automated semantic versioning:

1. **Conventional Commits** determine version bumps
2. **GitHub Actions** handles automated releases
3. **PyPI Publishing** happens automatically on version tags

### For Maintainers

The release process is fully automated:
1. Merge pull requests to main
2. GitHub Actions analyzes commits
3. Version is bumped automatically
4. Releases are published to PyPI
5. GitHub releases are created with changelogs

## ❓ Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions, ideas, and community support
- **Documentation**: Check README and CLAUDE.md first
- **Code Examples**: Review sample configurations and tests

## 🎉 Recognition

Contributors are recognized through:
- **GitHub Contributors** section
- **Release Notes** mention significant contributions
- **Community Appreciation** in discussions and issues

Thank you for contributing to GitFlow Analytics! 🚀
# Development Setup

This guide will help you set up a local development environment for GitFlow Analytics.

## 📋 Prerequisites

### System Requirements
- **Python 3.9+** (3.11+ recommended)
- **Git 2.20+**
- **4GB+ RAM** (8GB+ recommended for large repositories)
- **2GB+ disk space** for dependencies and cache

### Optional Dependencies
- **Docker** (for containerized development)
- **VS Code** or **PyCharm** (recommended IDEs)
- **GitHub CLI** (for easier PR management)

## 🚀 Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/bobmatnyc/gitflow-analytics.git
cd gitflow-analytics
```

### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n gitflow-analytics python=3.11
conda activate gitflow-analytics
```

### 3. Install Dependencies
```bash
# Install in development mode with all extras
pip install -e ".[dev,github,tui,all]"

# Or install specific extras only
pip install -e ".[dev]"  # Minimum for development
```

### 4. Verify Installation
```bash
# Check CLI works
gitflow-analytics --version

# Run basic tests
pytest tests/ -v

# Check linting
ruff check src/
black --check src/
```

## 🛠️ Development Tools

### Code Quality Tools
All tools are configured in `pyproject.toml`:

```bash
# Linting and formatting
ruff check src/                    # Fast linter
ruff check src/ --fix             # Auto-fix issues
black src/                        # Code formatter
isort src/                        # Import sorting
mypy src/                         # Type checking
bandit -r src/                    # Security scanning

# Testing
pytest                            # Run all tests
pytest --cov=gitflow_analytics    # With coverage
pytest -k "test_name"             # Run specific tests
pytest --lf                      # Run last failed tests
```

### Pre-commit Hooks (Recommended)
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🏗️ Project Structure

```
gitflow-analytics/
├── src/gitflow_analytics/        # Main package
│   ├── cli.py                    # Command-line interface
│   ├── core/                     # Core analysis engine
│   ├── models/                   # Data models
│   ├── extractors/               # Data extraction
│   ├── qualitative/              # ML/NLP analysis
│   ├── reports/                  # Report generation
│   └── tui/                      # Terminal UI
├── tests/                        # Test suite
├── docs/                         # Documentation
├── examples/                     # Example configurations
├── configs/                      # Sample configurations
└── pyproject.toml               # Project configuration
```

## 🧪 Testing Setup

### Test Categories
- **Unit tests**: `tests/test_*.py`
- **Integration tests**: `tests/integrations/`
- **End-to-end tests**: `tests/e2e/`

### Running Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/test_cli.py

# With coverage report
pytest --cov=gitflow_analytics --cov-report=html

# Fast tests only (skip slow integration tests)
pytest -m "not slow"

# Parallel execution
pytest -n auto  # Requires pytest-xdist
```

### Test Data
- Use `tests/fixtures/` for test data
- Mock external services in tests
- Use `pytest.fixture` for reusable test setup

## 🔧 IDE Configuration

### VS Code
Recommended extensions:
- Python
- Pylance
- Black Formatter
- Ruff
- GitLens

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

### PyCharm
- Set interpreter to virtual environment
- Enable pytest as test runner
- Configure Black as external tool
- Install Ruff plugin

## 🐳 Docker Development

### Using Docker Compose
```bash
# Build development container
docker-compose -f docker-compose.dev.yml build

# Run development environment
docker-compose -f docker-compose.dev.yml up

# Run tests in container
docker-compose -f docker-compose.dev.yml run --rm app pytest
```

### Dockerfile.dev
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY . .
RUN pip install -e ".[dev]"

CMD ["bash"]
```

## 🔍 Debugging

### CLI Debugging
```bash
# Enable debug logging
export GITFLOW_DEBUG=1
gitflow-analytics analyze --verbose

# Use Python debugger
python -m pdb -m gitflow_analytics.cli analyze
```

### IDE Debugging
- Set breakpoints in IDE
- Configure launch configurations
- Use interactive debugging for complex issues

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Permission Errors
```bash
# On macOS/Linux, ensure proper permissions
chmod +x scripts/*.sh

# For cache directory issues
export GITFLOW_CACHE_DIR=/tmp/gitflow-cache
```

#### Memory Issues
```bash
# Reduce batch sizes for large repositories
export GITFLOW_BATCH_SIZE=100

# Use streaming for large datasets
export GITFLOW_STREAMING=1
```

## 📊 Performance Profiling

### CPU Profiling
```bash
# Using cProfile
python -m cProfile -o profile.stats -m gitflow_analytics.cli analyze

# Analyze results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

### Memory Profiling
```bash
# Using memory_profiler
pip install memory_profiler
python -m memory_profiler gitflow_analytics/cli.py
```

## 🚀 Next Steps

1. **Read the [Contributing Guide](contributing.md)**
2. **Check [Coding Standards](coding-standards.md)**
3. **Review [Testing Guide](testing-guide.md)**
4. **Explore [Architecture Documentation](../architecture/)**
5. **Join the community discussions**

## 🆘 Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search GitHub issues
- **Discussions**: GitHub Discussions
- **Code Review**: Submit draft PRs for feedback

---

Happy coding! 🎉

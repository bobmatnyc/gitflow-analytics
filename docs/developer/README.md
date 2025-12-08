# Developer Documentation

Documentation for contributors, maintainers, and developers working on GitFlow Analytics.

## 🚀 Getting Started as a Contributor

### [Contributing Guide](contributing.md)
Complete guide to contributing to GitFlow Analytics:
- Code of conduct and community guidelines
- Issue reporting and feature request process
- Pull request workflow and review process  
- Release and versioning procedures

### [Development Setup](development-setup.md)
Set up your local development environment:
- Repository setup and dependency installation
- Development tools configuration (linting, testing)
- IDE setup and debugging configuration
- Local testing with sample repositories

### [Testing Guide](testing-guide.md)
Comprehensive testing procedures and standards:
- Unit testing with pytest framework
- Integration testing with real repositories
- Performance testing and benchmarking
- Test data management and fixtures

## 📐 Code Standards & Architecture

### [Refactoring Guide](refactoring-guide.md)
Code quality improvement tracking and procedures:
- Ongoing refactoring phases and progress
- Code quality metrics and targets
- Refactoring principles and best practices
- Testing strategies for safe refactoring

### [Coding Standards](coding-standards.md)
Code quality guidelines and conventions:
- Python style guide (Black, Ruff, mypy)
- Documentation standards and docstring format
- Error handling and logging patterns
- Performance and security best practices

### [Project Organization](project-organization.md)
Official project structure and organization standards:
- Directory structure and file placement rules
- Naming conventions and patterns
- Framework-specific organization
- Temporary files and cleanup policies

### [Release Process](release-process.md)
Automated release and deployment procedures:
- Semantic versioning with conventional commits
- GitHub Actions CI/CD pipeline
- PyPI publishing and distribution
- Documentation updates and changelog management

## 🏗️ System Understanding

### [Training Guide](training-guide.md)
Understanding the ML and analysis components:
- Machine learning pipeline architecture
- Training data generation and management
- Model evaluation and performance tuning
- Feature engineering and categorization logic

## 🎯 Developer Quick Reference

### Essential Development Commands
```bash
# Setup development environment
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# Code quality checks
ruff check src/ tests/
black src/ tests/ --check  
mypy src/

# Testing
pytest tests/ -v
pytest --cov=gitflow_analytics --cov-report=html

# Local installation for testing
pip install -e .
gitflow-analytics --version
```

### Key Development Areas

**Core Analysis Engine** (`src/gitflow_analytics/core/`)
- Git repository processing and commit analysis
- Developer identity resolution and consolidation
- Caching system and performance optimization

**Data Extraction** (`src/gitflow_analytics/extractors/`)
- Commit categorization (rule-based and ML)
- Ticket reference extraction and parsing
- Story point and project management integration

**ML Pipeline** (`src/gitflow_analytics/qualitative/`)
- spaCy-based natural language processing
- Commit classification with confidence scoring
- Pattern learning and model improvement

**Report Generation** (`src/gitflow_analytics/reports/`)
- CSV, JSON, and Markdown report formats
- Template-based narrative generation
- Data visualization and insight generation

**Integrations** (`src/gitflow_analytics/integrations/`)
- GitHub API client and organization discovery
- JIRA and project management platform adapters
- External tool integration patterns

## 🧪 Testing Strategies

### Unit Testing
- Test individual functions and methods in isolation
- Mock external dependencies (GitHub API, file system)
- Focus on edge cases and error conditions
- Maintain >80% code coverage

### Integration Testing  
- Test complete workflows with sample repositories
- Validate report generation end-to-end
- Test configuration parsing and validation
- Performance testing with large datasets

### Quality Assurance
- Automated linting and type checking in CI
- Manual testing with diverse repository types
- Documentation accuracy verification
- Security scanning and dependency updates

## 📊 Architecture Overview

GitFlow Analytics follows a modular architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │────│  Core Analyzer   │────│ Report Writers  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Config Loader   │    │   Data Models    │    │ Export Formats  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Integrations   │────│  ML Pipeline     │────│ Cache System    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Design Principles
- **Modularity**: Clear separation of concerns
- **Testability**: Dependency injection and mocking support  
- **Performance**: Caching and batch processing
- **Extensibility**: Plugin architecture for new integrations
- **Reliability**: Comprehensive error handling and validation

## 🔧 Development Workflows

### Feature Development
1. **Issue Creation**: Discuss feature requirements and design
2. **Branch Creation**: Create feature branch from main
3. **Implementation**: Write code following standards
4. **Testing**: Add comprehensive tests for new functionality
5. **Documentation**: Update relevant documentation
6. **Review**: Submit pull request for code review
7. **Integration**: Merge after approval and CI success

### Bug Fixing
1. **Reproduction**: Create minimal test case demonstrating issue
2. **Root Cause**: Identify underlying cause through debugging
3. **Fix Implementation**: Minimal change to resolve issue
4. **Test Coverage**: Add test preventing regression
5. **Validation**: Verify fix resolves original issue

### Performance Optimization  
1. **Measurement**: Profile code to identify bottlenecks
2. **Analysis**: Understand performance characteristics
3. **Optimization**: Implement targeted improvements
4. **Benchmarking**: Validate performance gains
5. **Documentation**: Update performance guidelines

## 📚 Learning Resources

### Essential Reading
- **[System Overview](../architecture/system-overview.md)** - High-level architecture
- **[Data Flow](../architecture/data-flow.md)** - Processing pipeline design
- **[Design Documents](../design/)** - Technical decision records
- **[Configuration Guide](../guides/configuration.md)** - Complete feature overview

### External Dependencies
- **GitPython**: Git repository interaction
- **PyGitHub**: GitHub API integration  
- **spaCy**: Natural language processing
- **pandas**: Data analysis and manipulation
- **SQLAlchemy**: Database ORM and caching

## 🤝 Community & Support

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Design questions and community help
- **Pull Requests**: Code review and collaboration
- **Documentation**: Shared knowledge and best practices

### Maintainer Responsibilities
- **Code Review**: Ensure quality and consistency
- **Release Management**: Version planning and deployment
- **Community Support**: Help users and contributors
- **Architecture Decisions**: Guide technical direction

## 🔄 Related Documentation

- **[Architecture](../architecture/)** - System design details
- **[Design Documents](../design/)** - Technical decision records  
- **[Reference](../reference/)** - API and configuration specifications
- **[Deployment](../deployment/)** - Operations and production setup

Ready to contribute? Start with the [Contributing Guide](contributing.md) and [Development Setup](development-setup.md)!
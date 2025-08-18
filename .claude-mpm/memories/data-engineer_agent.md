# Data-Engineer Agent Memory - gitflow-analytics

<!-- MEMORY LIMITS: 8KB max | 10 sections max | 15 items per section -->
<!-- Last Updated: 2025-08-11 10:41:47 | Auto-updated by: data-engineer -->

## Project Context
gitflow-analytics: python (with javascript) standard application
- Main modules: gitflow_analytics, gitflow_analytics/classification, gitflow_analytics/metrics, gitflow_analytics/identity_llm
- Testing: Tests in /tests/ directory
- Key patterns: Unit Testing

## Project Architecture
- Standard Application with python implementation
- Main directories: src, tests, docs
- Core modules: gitflow_analytics, gitflow_analytics/classification, gitflow_analytics/metrics, gitflow_analytics/identity_llm

## Coding Patterns Learned
- Python project: use type hints, follow PEP 8 conventions
- Project uses: Unit Testing

## Implementation Guidelines
- Use pip for dependency management
- Follow tests in /tests/ directory
- Follow pytest fixtures
- Key config files: pyproject.toml

## Domain-Specific Knowledge
<!-- Agent-specific knowledge for gitflow-analytics domain -->
- Key project terms: classification, training, gitflow, identity
- Focus on implementation patterns, coding standards, and best practices

## Effective Strategies
<!-- Successful approaches discovered through experience -->

## Common Mistakes to Avoid
- Avoid circular imports - use late imports when needed
- Don't ignore virtual environment - always activate before work
- Don't ignore database transactions in multi-step operations
- Avoid N+1 queries - use proper joins or prefetching

## Integration Points
- Sqlite database integration

## Performance Considerations
- Use list comprehensions over loops where appropriate
- Consider caching for expensive operations
- Index frequently queried columns
- Use connection pooling for database connections

## Current Technical Context
- Tech stack: python
- Data storage: sqlite
- Key dependencies: click>=8.1, gitpython>=3.1, pygithub>=2.0, tqdm>=4.65
- Documentation: README.md, CONTRIBUTING.md, CHANGELOG.md

## Recent Learnings
<!-- Most recent discoveries and insights -->

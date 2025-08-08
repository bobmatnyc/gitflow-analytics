# Reference Documentation

Technical specifications, schemas, and API documentation for GitFlow Analytics.

## 📖 Command Reference

### [CLI Commands](cli-commands.md)
Complete command-line interface reference:
- All available commands and subcommands
- Command-line arguments and options
- Usage examples and output formats
- Exit codes and error handling

## 📋 Configuration Reference

### [Configuration Schema](configuration-schema.md)  
Complete YAML configuration specification:
- All configuration sections and options
- Data types and validation rules
- Default values and acceptable ranges
- Environment variable substitution
- Configuration inheritance and overrides

### [JSON Export Schema](json-export-schema.md)
Detailed specification of JSON data export format:
- Data structure and field definitions
- Nested object relationships
- Data types and value constraints
- Version compatibility information
- Integration examples

## 🚀 System Reference

### [Cache System](cache-system.md)
Internal caching mechanism documentation:
- SQLite database schema and structure
- Cache invalidation strategies
- Performance characteristics and limits
- Manual cache management commands
- Troubleshooting cache issues

### [API Reference](api-reference.md)
Python API documentation for programmatic usage:
- Core classes and methods
- Data models and schemas
- Configuration objects
- Integration patterns
- Error handling

## 🎯 Quick Reference

### Common CLI Patterns
```bash
# Basic analysis
gitflow-analytics -c config.yaml --weeks 8

# Validate configuration without running
gitflow-analytics -c config.yaml --validate-only

# Clear cache and re-analyze  
gitflow-analytics -c config.yaml --clear-cache --weeks 4

# Export JSON data only
gitflow-analytics -c config.yaml --format json --weeks 12
```

### Essential Configuration Sections
```yaml
github:
  token: "${GITHUB_TOKEN}"
  
repositories:
  - owner: "myorg"
    name: "myrepo"
    
analysis:
  weeks: 8
  
reports:
  output_directory: "./reports"
```

### Key JSON Export Fields
```json
{
  "metadata": {...},
  "developers": [...],
  "repositories": [...],
  "metrics": {...}
}
```

## 📊 Data Models

### Core Entities
- **Developer**: Individual contributor with consolidated identity
- **Repository**: Git repository with associated metadata
- **Commit**: Individual code change with analysis metadata
- **Metric**: Calculated performance and activity measurements

### Relationships
- Developers contribute to Repositories through Commits
- Commits generate Metrics for analysis
- Identity resolution consolidates Developer records

## 🔍 Search & Navigation

### By Use Case

**Setting up authentication**
→ [Configuration Schema](configuration-schema.md#github-authentication)

**Understanding command options**  
→ [CLI Commands](cli-commands.md#command-options)

**Exporting data for external tools**
→ [JSON Export Schema](json-export-schema.md#export-structure)

**Troubleshooting performance**
→ [Cache System](cache-system.md#performance-optimization)

**Using Python API**
→ [API Reference](api-reference.md#getting-started)

### By Component

**Configuration**: Schema reference, validation rules, examples
**Commands**: CLI usage, options, examples  
**Data**: Export formats, schemas, integration
**System**: Caching, performance, internals

## 📚 Reference Completeness

### Coverage Levels
- 🟢 **Complete**: CLI Commands, Configuration Schema, Cache System  
- 🟡 **In Progress**: JSON Export Schema, API Reference
- 🔴 **Planned**: Advanced API patterns, Plugin system

### Accuracy Guarantee  
All reference documentation is:
- ✅ Generated from source code or validated against implementation
- ✅ Tested with working examples
- ✅ Updated with each release
- ✅ Cross-referenced for consistency

## 🔄 Related Documentation

- **[Getting Started](../getting-started/)** - New user onboarding
- **[Guides](../guides/)** - Task-oriented tutorials
- **[Examples](../examples/)** - Real-world usage patterns
- **[Architecture](../architecture/)** - System design documentation

## 📝 Contributing to Reference Docs

Reference documentation should be:
- **Accurate**: Match implementation exactly
- **Complete**: Cover all options and parameters
- **Concise**: Focus on facts, not explanations
- **Testable**: Include verifiable examples
- **Current**: Update with code changes

See [Developer Documentation](../developer/) for contribution guidelines.
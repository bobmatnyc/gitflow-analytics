# Configuration Schema Reference

Complete YAML configuration specification for GitFlow Analytics.

## 📋 Schema Overview

GitFlow Analytics uses YAML configuration files with environment variable support. The configuration follows a hierarchical structure with sensible defaults for most options.

## 🏗️ Top-Level Structure

```yaml
# Complete configuration schema
github:                     # GitHub integration settings
  token: string             # GitHub personal access token
  owner: string             # Default repository owner
  organization: string      # Organization for auto-discovery
  
repositories: []            # Repository list (if not using organization)
  
analysis:                   # Analysis behavior configuration
  weeks: integer            # Time period to analyze
  identity: {}              # Identity resolution settings
  
reports:                    # Report generation settings
  output_directory: string  # Where to save reports
  formats: []               # Output formats to generate
  
ml_categorization: {}       # Machine learning settings

jira: {}                    # JIRA integration (optional)

logging: {}                 # Logging configuration
```

## 🔑 GitHub Configuration

### Required Fields
```yaml
github:
  token: "${GITHUB_TOKEN}"          # Personal access token (required)
```

### Repository-Based Configuration
```yaml
github:
  token: "${GITHUB_TOKEN}"
  owner: "default-owner"            # Default owner for repositories
  
repositories:
  - owner: "myorg"                  # Repository owner
    name: "myrepo"                  # Repository name  
    local_path: "./repos/myrepo"    # Local clone path
    project_key: "PROJ"             # Optional project identifier
```

### Organization-Based Configuration
```yaml
github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"             # Auto-discover all repositories
  
# repositories: []                  # Leave empty for auto-discovery
```

### Advanced GitHub Settings
```yaml
github:
  token: "${GITHUB_TOKEN}"
  api_url: "https://api.github.com" # GitHub API URL (for GitHub Enterprise)
  timeout: 30                       # API request timeout (seconds)
  retry_attempts: 3                 # Number of retry attempts
  rate_limit_buffer: 100            # Buffer for rate limiting
```

## 📊 Analysis Configuration

### Basic Analysis Settings
```yaml
analysis:
  weeks: 8                          # Number of weeks to analyze (default: 4)
  include_merge_commits: false      # Include merge commits (default: false)
  min_commits_per_author: 1         # Minimum commits to include author
  cache_ttl: 3600                   # Cache time-to-live in seconds
```

### File Filtering
```yaml
analysis:
  include_file_patterns:            # Include only these file patterns
    - "*.py"
    - "*.js"
    - "*.ts"
    - "*.md"
  
  exclude_file_patterns:            # Exclude these file patterns
    - "*.min.js"
    - "*.lock"
    - "__pycache__/*"
    - "node_modules/*"
    - ".git/*"
```

### Identity Resolution
```yaml
analysis:
  identity:
    auto_analysis: true             # Enable automatic identity resolution
    fuzzy_threshold: 0.85           # Similarity threshold for matching
    cache_ttl_days: 7               # Cache identity analysis results
    
    manual_mappings:                # Manual identity consolidation
      - name: "John Smith"          # Display name
        primary_email: "john.smith@company.com"
        aliases:
          - "jsmith@company.com"
          - "150280367+jsmith@users.noreply.github.com"
      
      - name: "Jane Doe"
        canonical_email: "jane.doe@company.com"  # Legacy format
        emails:
          - "jane@company.com"
          - "jane.doe.work@gmail.com"
```

## 📝 Reports Configuration

### Output Settings
```yaml
reports:
  output_directory: "./reports"     # Report output directory
  formats: ["csv", "markdown"]      # Output formats
  
  filename_template: "analysis_{date}"  # Custom filename template
  date_format: "%Y%m%d"             # Date format for filenames
  
  include_untracked_analysis: true  # Include untracked commits analysis
  include_detailed_metrics: false   # Include detailed metric breakdowns
```

### Available Formats
```yaml
reports:
  formats:
    - "csv"                         # CSV data files
    - "json"                        # JSON data export
    - "markdown"                    # Narrative markdown report
    - "html"                        # HTML dashboard (future)
```

### Custom Report Templates
```yaml
reports:
  templates:
    narrative:
      executive_summary: true       # Include executive summary
      team_composition: true        # Include team analysis
      project_breakdown: true       # Include project details
      recommendations: true         # Include recommendations
      
    csv:
      weekly_metrics: true          # Generate weekly metrics CSV
      developer_profiles: true      # Generate developer CSV
      untracked_commits: true       # Generate untracked commits CSV
```

## 🧠 ML Categorization Configuration

### Basic ML Settings
```yaml
ml_categorization:
  enabled: true                     # Enable ML categorization
  model_name: "en_core_web_sm"      # spaCy model name
  confidence_threshold: 0.7         # Minimum confidence for ML predictions
  hybrid_threshold: 0.8             # Threshold for ML vs rule-based
```

### Advanced ML Settings  
```yaml
ml_categorization:
  enabled: true
  model_name: "en_core_web_md"      # Larger model for better accuracy
  confidence_threshold: 0.75
  hybrid_threshold: 0.8
  
  batch_size: 100                   # Commit processing batch size
  cache_predictions: true           # Cache ML predictions
  fallback_to_rules: true           # Use rule-based if ML fails
  
  categories:                       # Custom category definitions
    feature: ["feat", "add", "implement", "create"]
    bugfix: ["fix", "bug", "resolve", "patch"]
    refactor: ["refactor", "cleanup", "optimize"]
    documentation: ["docs", "readme", "comment"]
```

## 🎫 JIRA Integration (Optional)

### Basic JIRA Settings
```yaml
jira:
  server_url: "https://company.atlassian.net"
  username: "${JIRA_ACCESS_USER}"
  api_token: "${JIRA_ACCESS_TOKEN}"
  
  project_keys: ["PROJ", "TEAM"]    # JIRA project keys to track
```

### Advanced JIRA Settings
```yaml
jira:
  server_url: "https://company.atlassian.net"
  username: "${JIRA_ACCESS_USER}"
  api_token: "${JIRA_ACCESS_TOKEN}"
  
  project_keys: ["PROJ", "TEAM", "INFRA"]
  timeout: 30                       # API timeout seconds
  retry_attempts: 3                 # Number of retries
  
  story_point_field: "customfield_10016"  # Custom field for story points
  epic_link_field: "customfield_10014"    # Epic link field
  
  issue_types: ["Story", "Bug", "Task", "Epic"]  # Issue types to analyze
```

## 📝 Logging Configuration

### Basic Logging
```yaml
logging:
  level: "INFO"                     # Log level (DEBUG, INFO, WARNING, ERROR)
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Advanced Logging
```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  handlers:
    console:
      enabled: true
      level: "INFO"
      
    file:
      enabled: true
      filename: "gitflow-analytics.log"
      level: "DEBUG"
      max_size: "10MB"
      backup_count: 3
      
    syslog:
      enabled: false
      facility: "local0"
      level: "WARNING"
```

## 🚀 Performance Configuration

### Cache Settings
```yaml
cache:
  directory: "./.gitflow-cache"     # Cache directory
  commit_cache_ttl: 86400           # Commit cache TTL (seconds)
  identity_cache_ttl: 604800        # Identity cache TTL (seconds)
  ml_cache_ttl: 2592000             # ML prediction cache TTL (seconds)
  
  auto_cleanup: true                # Automatic cache cleanup
  cleanup_older_than_days: 30       # Remove cache entries older than N days
```

### Processing Settings
```yaml
processing:
  max_workers: 4                    # Maximum parallel workers
  batch_size: 1000                  # Commit processing batch size
  memory_limit_mb: 2048             # Memory usage limit
  
  git_timeout: 300                  # Git operation timeout (seconds)
  api_timeout: 30                   # API request timeout (seconds)
```

## 🔧 Environment Variable Support

All string values support environment variable substitution:

```yaml
# Environment variable formats
github:
  token: "${GITHUB_TOKEN}"          # Standard substitution
  owner: "${GITHUB_OWNER:-default}" # With default value
  
jira:
  username: "$JIRA_USER"            # Short format
  api_token: "${JIRA_TOKEN:?Required}" # With error if missing
```

### Environment File Support
GitFlow Analytics automatically loads `.env` files from the same directory as the configuration file:

```bash
# .env file
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
JIRA_ACCESS_USER=user@company.com
JIRA_ACCESS_TOKEN=xxxxxxxxxxxxxxxxxxxx
```

## ✅ Configuration Validation

### Required Fields Validation
GitFlow Analytics validates required configuration fields:
- `github.token` - GitHub personal access token
- Either `github.organization` OR `repositories` list

### Type Validation  
All configuration values are validated for correct types:
- Integers: `weeks`, `timeout`, `batch_size`
- Booleans: `enabled`, `auto_analysis`, `include_merge_commits`
- Strings: `token`, `owner`, `name`, `local_path`
- Arrays: `repositories`, `formats`, `project_keys`

### Value Range Validation
Some values have acceptable ranges:
- `weeks`: 1-104 (2 years maximum)
- `confidence_threshold`: 0.0-1.0
- `fuzzy_threshold`: 0.0-1.0
- `timeout`: 1-300 seconds

## 🔍 Configuration Examples

### Minimal Configuration
```yaml
# Simplest possible configuration
github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"
```

### Development Configuration  
```yaml
# Development and testing setup
github:
  token: "${GITHUB_TOKEN}"
  repositories:
    - owner: "myorg"
      name: "test-repo" 
      local_path: "./test-repo"

analysis:
  weeks: 2                          # Short period for testing
  
reports:
  output_directory: "./test-reports"
  formats: ["csv"]
  
logging:
  level: "DEBUG"                    # Verbose logging
```

### Production Configuration
```yaml
# Enterprise production setup
github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"
  timeout: 60
  retry_attempts: 5

analysis:
  weeks: 8
  identity:
    auto_analysis: true
    fuzzy_threshold: 0.9
    manual_mappings:
      # ... detailed identity mappings
      
ml_categorization:
  enabled: true
  model_name: "en_core_web_md"
  confidence_threshold: 0.8
  
reports:
  output_directory: "/var/lib/gitflow/reports"
  formats: ["csv", "json", "markdown"]
  include_untracked_analysis: true

jira:
  server_url: "https://company.atlassian.net"
  username: "${JIRA_ACCESS_USER}"
  api_token: "${JIRA_ACCESS_TOKEN}"
  project_keys: ["PROJ", "INFRA", "DATA"]

cache:
  directory: "/var/cache/gitflow"
  auto_cleanup: true
  cleanup_older_than_days: 14

processing:
  max_workers: 8
  batch_size: 2000
  memory_limit_mb: 4096
  
logging:
  level: "INFO"
  handlers:
    file:
      enabled: true
      filename: "/var/log/gitflow/analytics.log"
      max_size: "50MB"
      backup_count: 5
```

## 🆘 Configuration Troubleshooting

### Common YAML Errors
- **Tab characters**: YAML requires spaces, not tabs
- **Inconsistent indentation**: Use 2 or 4 spaces consistently
- **Missing colons**: Required after key names
- **Unclosed quotes**: Ensure string delimiters match

### Environment Variable Issues  
- **Variable not set**: Use `${VAR:?Required}` to catch missing variables
- **Default values**: Use `${VAR:-default}` for optional variables
- **Quotes needed**: Wrap variables in quotes for string contexts

### Validation Commands
```bash
# Test configuration file
gitflow-analytics validate -c config.yaml

# Check specific components
gitflow-analytics validate -c config.yaml --check-tokens --check-repos
```

## 📚 Related Documentation

- **[Configuration Guide](../guides/configuration.md)** - User-friendly configuration tutorial  
- **[CLI Commands](cli-commands.md)** - Complete command-line reference
- **[Getting Started](../getting-started/)** - Installation and first steps
- **[Troubleshooting](../guides/troubleshooting.md)** - Common issues and solutions
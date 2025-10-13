# GitFlow Analytics Configuration Guide

This comprehensive guide covers all configuration features in GitFlow Analytics, from basic setup to advanced features like configuration profiles, extending base configurations, and modular configuration management.

## 🚀 Quick Start

1. **Copy the example files:**
   ```bash
   cp config-sample.yaml my-config.yaml
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials:**
   ```bash
   GITHUB_TOKEN=your_github_personal_access_token
   GITHUB_OWNER=your_github_username_or_org
   ```

3. **Run the analysis:**
   ```bash
   gitflow-analytics -c my-config.yaml
   ```

## 📋 Basic Configuration

### GitHub Authentication

The `github` section supports both direct tokens and environment variables, plus organization-based repository discovery:

```yaml
github:
  token: "${GITHUB_TOKEN}"     # From environment variable
  owner: "${GITHUB_OWNER}"     # Default owner for repository-based config
  organization: "myorg"        # For organization-based discovery
  # token: "ghp_direct_token_here"  # Or direct token (not recommended)
```

#### Organization-based Configuration

When `organization` is specified, GitFlow Analytics automatically discovers all non-archived repositories:

```yaml
version: "1.0"

github:
  token: "${GITHUB_TOKEN}"
  organization: "myorg"  # Automatically discovers repositories

analysis:
  weeks: 4

reports:
  output_directory: "./reports"
```

#### Repository-based Configuration

For specific repositories, use the `repositories` list:

```yaml
version: "1.0"

github:
  token: "${GITHUB_TOKEN}"
  repositories:
    - owner: "myorg"
      name: "repo1"
      local_path: "./repos/repo1"
    - owner: "myorg"
      name: "repo2"
      local_path: "./repos/repo2"

analysis:
  weeks: 4

reports:
  output_directory: "./reports"
```

### Analysis Configuration

```yaml
analysis:
  weeks: 4                    # Number of weeks to analyze
  branch_strategy: "smart"    # Branch analysis strategy: main_only, smart, all
  max_branches: 50           # Maximum branches per repository (smart strategy)
  include_merges: true       # Include merge commits in analysis
  exclude_bots: true         # Exclude bot commits
```

### Reports Configuration

```yaml
reports:
  output_directory: "./reports"
  formats: ["csv", "markdown", "json"]  # Output formats
  include_charts: true                  # Generate charts (requires matplotlib)
  anonymize: false                      # Anonymize developer names
```

### PM Platform Configuration

GitFlow Analytics supports multiple project management platforms for ticket tracking:

```yaml
# Configure which platforms to track
analysis:
  ticket_platforms:
    - jira        # Track JIRA tickets (PROJ-123)
    - linear      # Track Linear issues (ENG-123)
    - clickup     # Track ClickUp tasks (CU-abc123)
    - github      # Track GitHub Issues (#123, GH-123)

# Platform-specific configuration
pm:
  jira:
    access_user: "${JIRA_ACCESS_USER}"
    access_token: "${JIRA_ACCESS_TOKEN}"
    base_url: "https://company.atlassian.net"

  linear:
    api_key: "${LINEAR_API_KEY}"
    team_ids:  # Optional: filter by team
      - "team_123abc"

  clickup:
    api_token: "${CLICKUP_API_TOKEN}"
    workspace_url: "https://app.clickup.com/12345/v/"

# GitHub Issues uses github.token automatically - no separate config needed
github:
  token: "${GITHUB_TOKEN}"

# Optional: JIRA story point integration
jira_integration:
  enabled: true
  fetch_story_points: true
  story_point_fields:
    - "Story point estimate"
    - "customfield_10016"
```

See the [PM Platform Setup Guide](../guides/pm-platform-setup.md) for detailed setup instructions for each platform.

## Configuration Profiles

Configuration profiles provide pre-configured settings optimized for specific use cases. This allows you to quickly set up GitFlow Analytics for different scenarios without manually configuring every setting.

### Available Profiles

#### Performance Profile
Optimized for speed with large repositories:
- Analyzes main branch only
- Disables ML/LLM features for speed
- Larger batch sizes for processing
- CSV output only
- Extended cache duration (2 weeks)

```yaml
version: "1.0"
profile: performance
```

#### Quality Profile  
Maximum analysis depth and accuracy:
- Smart branch analysis (up to 100 branches)
- Enables all ML/LLM features
- Higher confidence thresholds
- All output formats (CSV, Markdown, JSON)
- Shorter cache for data freshness

```yaml
version: "1.0"
profile: quality
```

#### Balanced Profile (Default)
Good balance between performance and quality:
- Smart branch analysis (up to 50 branches)
- ML enabled with moderate thresholds
- CSV and Markdown output
- Standard cache duration (1 week)

```yaml
version: "1.0"
profile: balanced
```

#### Minimal Profile
Essential features only for quick overview:
- Main branch analysis only
- No ML/LLM features
- CSV output only
- Long cache duration (30 days)
- No automatic identity analysis

```yaml
version: "1.0"
profile: minimal
```

### Overriding Profile Settings

You can override specific profile settings while keeping the rest of the profile defaults:

```yaml
version: "1.0"
profile: performance  # Start with performance profile

# Override specific settings
analysis:
  ml_categorization:
    enabled: true  # Enable ML despite performance profile
    min_confidence: 0.8
  branch_analysis:
    max_branches_per_repo: 25  # Analyze more branches

output:
  formats: ["csv", "markdown"]  # Add markdown output
```

## Extending Configurations

The `extends` feature allows you to create base configurations that can be shared and extended by multiple projects. This is useful for:
- Sharing common settings across teams
- Creating organization-wide defaults
- Managing environment-specific configurations

### Basic Extension

Create a base configuration:

```yaml
# config-base.yaml
version: "1.0"
github:
  owner: "my-organization"
  base_url: "https://api.github.com"

cache:
  directory: "/shared/cache"
  ttl_hours: 168

analysis:
  exclude:
    authors:
      - "dependabot[bot]"
      - "renovate[bot]"
  identity:
    similarity_threshold: 0.85
```

Extend from the base:

```yaml
# config-project.yaml
version: "1.0"
extends: "./config-base.yaml"

repositories:
  - name: "project-repo"
    path: "./repos/project"
    github_repo: "my-organization/project-repo"

# Override base settings if needed
cache:
  ttl_hours: 336  # Use longer cache for this project
```

### Combining Extends with Profiles

You can use both extends and profiles together. The merge order is:
1. Profile defaults are applied first
2. Base configuration is merged
3. Current configuration overrides are applied last

```yaml
version: "1.0"
extends: "./config-base.yaml"
profile: quality  # Apply quality profile

repositories:
  - name: "critical-project"
    path: "./repos/critical"

# Final overrides
analysis:
  ml_categorization:
    min_confidence: 0.9  # Even stricter for critical project
```

### Path Resolution

The `extends` path can be:
- **Relative**: Resolved relative to the current config file
- **Absolute**: Full system path

```yaml
# Relative path (recommended)
extends: "./base.yaml"
extends: "../shared/config-base.yaml"

# Absolute path
extends: "/opt/gitflow/configs/base.yaml"
```

## Configuration Architecture

The configuration system has been refactored into focused modules for better maintainability:

### Module Structure

```
config/
├── __init__.py       # Public API exports
├── loader.py         # YAML loading and environment expansion
├── schema.py         # Configuration dataclasses
├── validator.py      # Validation logic
├── repository.py     # Repository discovery
├── profiles.py       # Configuration profiles
└── errors.py         # Error handling
```

### Error Handling

The new configuration system provides enhanced error messages with:
- Clear problem identification
- Actionable fix suggestions
- File location context
- Common YAML syntax issue detection

Example error output:
```
❌ YAML configuration error in config.yaml at line 5, column 3

🚫 Tab characters are not allowed in YAML files!

💡 Fix: Replace all tab characters with spaces (usually 2 or 4 spaces).
   Most editors can show whitespace characters and convert tabs to spaces.
   In VS Code: View → Render Whitespace, then Edit → Convert Indentation to Spaces

📁 File: /path/to/config.yaml
```

## Best Practices

### 1. Use Profiles for Quick Setup
Start with a profile that matches your use case, then customize as needed:

```yaml
version: "1.0"
profile: balanced  # Good starting point
# Add your specific configuration...
```

### 2. Create Shared Base Configurations
For teams, create a shared base configuration:

```yaml
# team-base.yaml
version: "1.0"
github:
  organization: "our-company"

# Multi-platform PM configuration
pm:
  jira:
    access_user: "${JIRA_ACCESS_USER}"
    access_token: "${JIRA_ACCESS_TOKEN}"
    base_url: "https://company.atlassian.net"

  linear:
    api_key: "${LINEAR_API_KEY}"
    team_ids: ["team_123"]

analysis:
  ticket_platforms:
    - jira
    - linear
    - github

  exclude:
    authors: ["bot-accounts"]

  identity:
    manual_mappings:
      - name: "John Smith"
        primary_email: "john@company.com"
        aliases: ["jsmith@old-email.com"]
```

### 3. Environment-Specific Configs
Use extends for environment-specific configurations:

```yaml
# config-dev.yaml
extends: "./config-base.yaml"
profile: minimal  # Fast for development

# config-prod.yaml  
extends: "./config-base.yaml"
profile: quality  # Thorough for production
```

### 4. Override Precedence
Remember the override precedence (last wins):
1. Profile defaults (lowest priority)
2. Extended base configuration
3. Current file settings (highest priority)

### 5. Validate Configurations
Use the `--validate-only` flag to test configurations:

```bash
gitflow-analytics -c config.yaml --validate-only
```

## Migration Guide

### From Monolithic to Modular

The configuration system maintains full backward compatibility. Existing configurations will continue to work without changes. The modular structure is internal and doesn't affect the configuration file format.

### Adding Profiles to Existing Configs

To add a profile to an existing configuration:

1. Add the `profile` field at the top level:
```yaml
version: "1.0"
profile: balanced  # Add this line
# ... rest of your config
```

2. Remove settings that are now handled by the profile (optional)
3. Keep only your specific overrides

### Creating Reusable Configurations

To refactor existing configs for reusability:

1. Extract common settings to a base file
2. Add `extends` to child configurations
3. Remove duplicated settings from child configs
4. Test with `--validate-only`

## Performance Considerations

### Profile Selection Impact

- **Performance profile**: ~2-3x faster for large repos
- **Quality profile**: ~2-3x slower but more accurate
- **Balanced profile**: Good default for most cases
- **Minimal profile**: ~5x faster, basic metrics only

### Cache Configuration

Profiles set appropriate cache durations:
- Performance: 2 weeks (fewer cache misses)
- Quality: 3 days (fresher data)
- Balanced: 1 week (good compromise)
- Minimal: 30 days (maximum caching)

### Branch Analysis Strategy

Controlled by `analysis.branch_analysis.strategy`:
- `main_only`: Fastest, analyzes only main/master
- `smart`: Balanced, analyzes active branches
- `all`: Slowest, analyzes everything

## Troubleshooting

### Profile Not Applied

If profile settings aren't being applied:
1. Check profile name spelling (case-insensitive)
2. Ensure profile is specified before overrides
3. Verify no syntax errors with `--validate-only`

### Extends Not Working

If base configuration isn't being loaded:
1. Check file path (relative to current config)
2. Verify base file exists and is readable
3. Check for circular dependencies
4. Validate base file syntax

### Override Not Taking Effect

If your overrides aren't working:
1. Check YAML indentation (must match structure)
2. Verify field names and nesting
3. Remember profiles are applied first
4. Use `--debug` flag to see merged config

## Examples

### Example 1: Large Organization

```yaml
version: "1.0"
profile: performance
extends: "./org-base.yaml"

github:
  organization: "large-corp"
  
# Only analyze main branches for 1000+ repos
analysis:
  branch_analysis:
    strategy: main_only
    max_branches_per_repo: 1
    
output:
  directory: "/fast-nvme/reports"
```

### Example 2: Small Team Quality Focus

```yaml
version: "1.0"
profile: quality

repositories:
  - name: "core-service"
    path: "./repos/core-service"
    github_repo: "team/core-service"

analysis:
  ml_categorization:
    min_confidence: 0.8  # Stricter quality
  commit_classification:
    confidence_threshold: 0.7
    
output:
  formats: ["csv", "markdown", "json"]
```

### Example 3: CI/CD Pipeline

```yaml
version: "1.0"
profile: minimal  # Fast for CI
extends: "./ci-base.yaml"

repositories:
  - name: "${REPO_NAME}"  # From CI environment
    path: "${WORKSPACE}"    # From CI environment
    
cache:
  directory: "${CI_CACHE_DIR}/.gitflow-cache"
  
output:
  directory: "${CI_ARTIFACTS_DIR}"
  formats: ["json"]  # For automated processing
```
# Interactive Launcher Guide

The interactive launcher provides a streamlined workflow for running GitFlow Analytics with persistent preferences and intuitive repository selection.

## Overview

The `gitflow-analytics run` command launches an interactive session that guides you through:
- Repository selection (multi-select)
- Analysis period configuration
- Cache management
- Identity analysis preferences
- Automatic preference saving

## Quick Start

```bash
# Launch with default config discovery
gitflow-analytics run

# Launch with specific config
gitflow-analytics run -c config.yaml
```

## Interactive Workflow

### 1. Configuration Loading

The launcher automatically searches for configuration files in this order:
- `config.yaml`
- `config.yml`
- `gitflow-config.yaml`
- `gitflow-config.yml`
- `.gitflow.yaml`

If no config is found, it prompts you to run `gitflow-analytics install`.

### 2. Repository Selection

Select repositories to analyze using one of these methods:

**Number Selection:**
```
Selection: 1,3,5  # Analyze repositories 1, 3, and 5
```

**All Repositories:**
```
Selection: all  # Analyze all configured repositories
```

**Previous Selection:**
```
Selection: [Enter]  # Use your last selection (shown with ✓)
```

**Default (First Run):**
```
Selection: [Enter]  # Analyzes all repositories if no previous selection
```

### 3. Analysis Period

Specify the number of weeks to analyze (1-52):
```
Number of weeks to analyze [4]: 8
```

The default value is your previously saved preference or 4 weeks.

### 4. Cache Management

Choose whether to clear the cache before analysis:
```
Clear cache before analysis? [y/N]: n
```

This preference is saved for future runs.

### 5. Identity Analysis

Choose whether to skip identity analysis:
```
Skip identity analysis? [y/N]: n
```

Skipping identity analysis speeds up the run if you've already resolved identities.

## Preferences Storage

All your selections are automatically saved to the `launcher` section of your configuration file:

```yaml
launcher:
  last_selected_repos:
    - frontend-app
    - mobile-app
  default_weeks: 8
  auto_clear_cache: false
  skip_identity_analysis: false
  last_run: "2025-10-06T19:00:00Z"
```

These preferences are used as defaults in future launcher sessions.

## Example Session

```bash
$ gitflow-analytics run

🚀 GitFlow Analytics Interactive Launcher

📁 Loading configuration from: ./config.yaml
✅ Configuration loaded

📂 Available Repositories:

   [✓] 1. frontend-app (/Users/masa/repos/frontend-app)
   [ ] 2. backend-service (/Users/masa/repos/backend-service)
   [✓] 3. mobile-app (/Users/masa/repos/mobile-app)

📝 Select repositories:
  • Enter numbers (comma-separated): 1,3,5
  • Enter 'all' for all repositories
  • Press Enter to use previous selection

Selection: 1,3
✅ Selected 2 repositories

📅 Number of weeks to analyze [4]: 8

🗑️  Clear cache before analysis? [y/N]: n

🔍 Skip identity analysis? [y/N]: n

💾 Saving preferences...
✅ Preferences saved to config.yaml

🚀 Starting analysis...
   Repositories: frontend-app, mobile-app
   Period: 8 weeks
   Clear cache: No
   Skip identity: No

[Analysis output...]

✅ Analysis complete!
```

## Advanced Usage

### Custom Configuration Path

```bash
gitflow-analytics run -c /path/to/custom-config.yaml
```

### Integration with Other Commands

The launcher automatically invokes the `analyze` command with your selections. All standard analysis features are available:
- Report generation
- Identity resolution
- Metric calculation
- PM platform integration

### Preference Management

To reset preferences, simply delete the `launcher` section from your config file, or edit individual values:

```yaml
launcher:
  default_weeks: 12  # Change default analysis period
  auto_clear_cache: true  # Always clear cache
```

## Troubleshooting

### Config Not Found

If the launcher can't find your configuration:
```bash
gitflow-analytics run -c /path/to/config.yaml
```

Or create a new configuration:
```bash
gitflow-analytics install
```

### Invalid Repository Selection

If you enter an invalid selection, the launcher will prompt you again. Valid formats:
- Single: `1`
- Multiple: `1,3,5`
- Range (not supported): Use explicit numbers
- All: `all`

### Subprocess Execution Issues

The launcher executes analysis in a subprocess. If you encounter issues:
1. Verify your config file is valid: `gitflow-analytics analyze -c config.yaml --validate-only`
2. Check Python environment: `which python` or `which python3`
3. Run analysis directly: `gitflow-analytics analyze -c config.yaml --weeks 8`

## Benefits

- **Faster workflow**: Skip typing long commands
- **Persistent preferences**: Your choices are remembered
- **Multi-select**: Easily analyze subsets of repositories
- **Visual feedback**: Clear indicators of previous selections
- **Error prevention**: Interactive prompts reduce typos

## See Also

- [Configuration Guide](./configuration.md)
- [Identity Resolution Guide](./identity-resolution.md)
- [Analysis Guide](./analysis.md)

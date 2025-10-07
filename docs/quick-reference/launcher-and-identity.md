# Quick Reference: Interactive Launcher & Enhanced Identity Detection

## Interactive Launcher

### Basic Usage
```bash
# Launch interactive mode
gitflow-analytics run

# With specific config
gitflow-analytics run -c config.yaml
```

### Repository Selection Formats
| Input | Action |
|-------|--------|
| `1,3,5` | Select repositories 1, 3, and 5 |
| `all` | Select all repositories |
| `[Enter]` | Use previous selection (shown with ✓) |
| `[Enter]` (first run) | Select all repositories |

### Preferences Saved
```yaml
launcher:
  last_selected_repos: [repo1, repo2]
  default_weeks: 8
  auto_clear_cache: false
  skip_identity_analysis: false
  last_run: "2025-10-06T19:00:00Z"
```

## Enhanced Identity Detection

### Basic Usage
```bash
# Analyze identities (90% confidence threshold)
gitflow-analytics identities -c config.yaml --weeks 12

# Auto-apply high-confidence suggestions
gitflow-analytics identities -c config.yaml --apply
```

### Confidence Levels
| Indicator | Confidence | Meaning |
|-----------|------------|---------|
| 🟢 | 95%+ | Very high confidence - safe to apply |
| 🟡 | 90-95% | High confidence - review and apply |
| 🟠 | <90% | Rejected - manual investigation needed |

### Example Output
```
🟢 Cluster 1 (Confidence: 95.3%):
   Primary: john.doe@company.com
   Alias:   150280367+johndoe@users.noreply.github.com
   Reason:  Same person based on name patterns...
```

### Manual Mappings Format
```yaml
analysis:
  manual_identity_mappings:
    - name: "John Doe"  # Optional display name
      primary_email: "john.doe@company.com"
      aliases:
        - "150280367+johndoe@users.noreply.github.com"
      confidence: 0.953  # Informational
      reasoning: "..."   # Informational
```

## Configuration

### Launcher Defaults
```yaml
# Set default analysis period
launcher:
  default_weeks: 12

# Always clear cache
launcher:
  auto_clear_cache: true

# Skip identity analysis by default
launcher:
  skip_identity_analysis: true
```

### Identity Confidence Threshold
```yaml
# Require 95% confidence (default: 90%)
analysis:
  similarity_threshold: 0.95
```

### OpenRouter API Key (for LLM analysis)
```yaml
qualitative:
  enabled: true
  openrouter_api_key: ${OPENROUTER_API_KEY}
  model: "openai/gpt-4o-mini"
```

Or in `.env` file:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

## Common Workflows

### First-Time Setup
```bash
# 1. Install and configure
gitflow-analytics install

# 2. Run identity analysis
gitflow-analytics identities -c config.yaml --weeks 12

# 3. Review and apply suggestions
# (Answer 'y' to approval prompt)

# 4. Use interactive launcher for ongoing analysis
gitflow-analytics run
```

### Regular Analysis Workflow
```bash
# Use launcher for quick analysis
gitflow-analytics run

# Select repositories interactively
# Configure period (default: your previous setting)
# Analysis runs automatically
```

### Identity Maintenance
```bash
# Weekly identity check
gitflow-analytics identities -c config.yaml --weeks 1

# Review new duplicates
# Apply high-confidence suggestions

# Clear identity cache if needed
rm .gitflow-cache/identities.db
```

## Troubleshooting

### Launcher Issues
| Issue | Solution |
|-------|----------|
| Config not found | Use `-c /path/to/config.yaml` or run `gitflow-analytics install` |
| Invalid selection | Use format: `1,3,5` or `all` |
| Subprocess error | Run analysis directly: `gitflow-analytics analyze -c config.yaml` |

### Identity Detection Issues
| Issue | Solution |
|-------|----------|
| No OpenRouter key | Configure in `qualitative.openrouter_api_key` or use heuristic fallback |
| Low confidence | Review full report: `cat .gitflow-cache/identity_analysis_*.yaml` |
| Incorrect suggestions | Don't apply; add manual mappings to prevent future suggestions |
| API rate limit | Run on smaller time periods or use caching |

## Key Differences from Previous Version

### Interactive Launcher (New)
- **Before**: `gitflow-analytics analyze -c config.yaml --weeks 8 --clear-cache`
- **After**: `gitflow-analytics run` (interactive prompts)

### Identity Detection (Enhanced)
- **Before**: 80% confidence threshold, no reasoning display
- **After**: 90% confidence threshold with color-coded confidence and reasoning

### Configuration Format (Updated)
- **Before**: `canonical_email` in identity mappings
- **After**: `primary_email` with confidence/reasoning (backward compatible)

## Best Practices

1. **Use launcher for regular analysis**: Faster and more convenient than typing full commands
2. **Review identity suggestions weekly**: Catch new duplicates early
3. **Keep confidence threshold at 90%+**: Prevents false positives
4. **Save launcher preferences**: Let the tool remember your common settings
5. **Review reasoning**: Understand why LLM suggested identity matches

## See Also

- [Interactive Launcher Guide](../guides/interactive-launcher.md)
- [Enhanced Identity Resolution Guide](../guides/identity-resolution-enhanced.md)
- [Configuration Guide](../guides/configuration.md)

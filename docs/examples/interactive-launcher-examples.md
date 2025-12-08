# Interactive Launcher Usage Examples

This document provides comprehensive examples of using the GitFlow Analytics interactive launcher for common workflows.

## Prerequisites

- GitFlow Analytics installed and configured
- At least one repository configured in `config.yaml`
- Basic familiarity with GitFlow Analytics concepts

## Example 1: First-Time Interactive Launcher Usage

```bash
$ gitflow-analytics run

🚀 GitFlow Analytics Interactive Launcher

📁 Loading configuration from: ./config.yaml
✅ Configuration loaded

📂 Available Repositories:

   [ ] 1. frontend-web (/Users/masa/repos/frontend-web)
   [ ] 2. backend-api (/Users/masa/repos/backend-api)
   [ ] 3. mobile-ios (/Users/masa/repos/mobile-ios)
   [ ] 4. mobile-android (/Users/masa/repos/mobile-android)
   [ ] 5. shared-components (/Users/masa/repos/shared-components)

📝 Select repositories:
  • Enter numbers (comma-separated): 1,3,5
  • Enter 'all' for all repositories
  • Press Enter to use previous selection

Selection: 1,2,3
✅ Selected 3 repositories

📅 Number of weeks to analyze [4]: 8

🗑️  Clear cache before analysis? [y/N]: n

🔍 Skip identity analysis? [y/N]: n

💾 Saving preferences...
✅ Preferences saved to config.yaml

🚀 Starting analysis...
   Repositories: frontend-web, backend-api, mobile-ios
   Period: 8 weeks
   Clear cache: No
   Skip identity: No

[Analysis proceeds normally...]

✅ Analysis complete!
```

**Generated Preferences** (in config.yaml):
```yaml
launcher:
  last_selected_repos:
    - frontend-web
    - backend-api
    - mobile-ios
  default_weeks: 8
  auto_clear_cache: false
  skip_identity_analysis: false
  last_run: "2025-10-06T19:30:00Z"
```

## Example 2: Subsequent Launcher Run (Using Saved Preferences)

```bash
$ gitflow-analytics run

🚀 GitFlow Analytics Interactive Launcher

📁 Loading configuration from: ./config.yaml
✅ Configuration loaded

📂 Available Repositories:

   [✓] 1. frontend-web (/Users/masa/repos/frontend-web)
   [✓] 2. backend-api (/Users/masa/repos/backend-api)
   [✓] 3. mobile-ios (/Users/masa/repos/mobile-ios)
   [ ] 4. mobile-android (/Users/masa/repos/mobile-android)
   [ ] 5. shared-components (/Users/masa/repos/shared-components)

📝 Select repositories:
  • Enter numbers (comma-separated): 1,3,5
  • Enter 'all' for all repositories
  • Press Enter to use previous selection

Selection: [Press Enter]
✅ Using previous selection: 3 repositories

📅 Number of weeks to analyze [8]: [Press Enter]

🗑️  Clear cache before analysis? [y/N]: [Press Enter]

🔍 Skip identity analysis? [y/N]: [Press Enter]

💾 Saving preferences...
✅ Preferences saved to config.yaml

🚀 Starting analysis...
   Repositories: frontend-web, backend-api, mobile-ios
   Period: 8 weeks
   Clear cache: No
   Skip identity: No

[Analysis proceeds...]
```

**Note**: All defaults come from saved preferences!

## Example 3: Selecting All Repositories

```bash
$ gitflow-analytics run

[...]

Selection: all
✅ Selected all 5 repositories

[Analysis proceeds with all repositories...]
```

## Example 4: Enhanced Identity Detection - High Confidence

```bash
$ gitflow-analytics identities -c config.yaml --weeks 12

🔍 Analyzing repositories for developer identities...
✅ Found 347 commits

📄 Analysis report saved to: .gitflow-cache/identity_analysis_20251006.yaml

⚠️  Found 4 potential identity clusters:

📋 Suggested identity mappings:

   🟢 Cluster 1 (Confidence: 97.8%):
      Primary: john.smith@company.com
      Alias:   150280367+jsmith@users.noreply.github.com
      Alias:   john.smith@contractor.com
      Reason:  Same person: GitHub noreply matches commit patterns, contractor email used befor...

   🟢 Cluster 2 (Confidence: 96.2%):
      Primary: sarah.johnson@company.com
      Alias:   150280368+sjohnson@users.noreply.github.com
      Reason:  GitHub noreply address with matching name pattern and commit timing correlation

   🟡 Cluster 3 (Confidence: 93.5%):
      Primary: mike.wilson@company.com
      Alias:   m.wilson@gmail.com
      Reason:  Name abbreviation matches, consistent commit patterns across both addresses

   🟡 Cluster 4 (Confidence: 91.2%):
      Primary: emily.chen@company.com
      Alias:   emily.chen@freelance.com
      Reason:  Same developer transitioning from freelance to employee based on timing

🤖 Found 3 bot accounts to exclude:
   - dependabot[bot]
   - renovate[bot]
   - github-actions[bot]

Apply these identity mappings to your configuration? [Y/n]: y

✅ Applied identity mappings to configuration
```

## Tips & Best Practices

### 1. First Run Strategy
- Start with a subset of repositories
- Use 4-8 weeks for initial analysis
- Review identity suggestions carefully
- Build up preferences gradually

### 2. Regular Workflow
- Use `gitflow-analytics run` for regular analysis
- Let preferences save time
- Run identity analysis weekly
- Clear cache when config changes

### 3. Identity Management
- Review high confidence (🟢) suggestions first
- Be cautious with medium confidence (🟡)
- Check full report for rejected clusters
- Add manual mappings for known duplicates

### 4. Performance Tips
- Use launcher to select active repos only
- Skip identity analysis if recently run
- Clear cache when you need fresh data
- Use smaller week ranges for faster results

## See Also

- [Interactive Launcher Guide](../guides/interactive-launcher.md)
- [Enhanced Identity Resolution Guide](../guides/identity-resolution-enhanced.md)
- [Quick Reference](../quick-reference/launcher-and-identity.md)

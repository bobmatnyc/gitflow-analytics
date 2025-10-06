# Usage Examples - Interactive Launcher & Enhanced Identity Detection

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

**Generated Config**:
```yaml
analysis:
  manual_identity_mappings:
    - primary_email: "john.smith@company.com"
      aliases:
        - "150280367+jsmith@users.noreply.github.com"
        - "john.smith@contractor.com"
      confidence: 0.978
      reasoning: "Same person: GitHub noreply matches commit patterns, contractor email used before..."

    - primary_email: "sarah.johnson@company.com"
      aliases:
        - "150280368+sjohnson@users.noreply.github.com"
      confidence: 0.962
      reasoning: "GitHub noreply address with matching name pattern and commit timing correlation"

    - primary_email: "mike.wilson@company.com"
      aliases:
        - "m.wilson@gmail.com"
      confidence: 0.935
      reasoning: "Name abbreviation matches, consistent commit patterns across both addresses"

    - primary_email: "emily.chen@company.com"
      aliases:
        - "emily.chen@freelance.com"
      confidence: 0.912
      reasoning: "Same developer transitioning from freelance to employee based on timing"

  exclude:
    authors:
      - "dependabot[bot]"
      - "renovate[bot]"
      - "github-actions[bot]"
```

## Example 5: Identity Detection - Some Clusters Rejected

```bash
$ gitflow-analytics identities -c config.yaml --weeks 12

🔍 Analyzing repositories for developer identities...
✅ Found 247 commits

📄 Analysis report saved to: .gitflow-cache/identity_analysis_20251006.yaml

⚠️  Found 2 potential identity clusters:

📋 Suggested identity mappings:

   🟢 Cluster 1 (Confidence: 95.7%):
      Primary: alex.brown@company.com
      Alias:   150280369+abrown@users.noreply.github.com
      Reason:  GitHub noreply address matches name and commit patterns

ℹ️  Note: 1 potential cluster was rejected (confidence < 90%)
    Review full report: .gitflow-cache/identity_analysis_20251006.yaml

Apply these identity mappings to your configuration? [Y/n]:
```

**Full Report** (identity_analysis_20251006.yaml):
```yaml
analysis_metadata:
  total_commits: 247
  unique_identities: 12
  clusters_found: 2
  clusters_rejected: 1
  rejection_reason: "Confidence below threshold (90%)"

identity_clusters:
  - canonical_name: "Alex Brown"
    canonical_email: "alex.brown@company.com"
    confidence: 0.957
    reasoning: "GitHub noreply address matches name and commit patterns"
    total_commits: 87
    aliases:
      - name: "Alex Brown"
        email: "150280369+abrown@users.noreply.github.com"
        commit_count: 87

rejected_clusters:
  - cluster:
      - name: "J. Doe"
        email: "jdoe@company.com"
      - name: "Jane Doe"
        email: "jane.doe@company.com"
    confidence: 0.82
    reasoning: "Name similarity but different commit patterns suggest different developers"
    rejection_reason: "Confidence 82% < threshold 90%"
```

## Example 6: Workflow - Setup to Analysis

```bash
# Step 1: Install and configure
$ gitflow-analytics install
[Interactive installation wizard...]

# Step 2: Run initial identity analysis
$ gitflow-analytics identities -c config.yaml --weeks 12
[Review and apply suggestions...]

# Step 3: Use interactive launcher for ongoing analysis
$ gitflow-analytics run
[Select repos and run analysis...]

# Step 4: Review reports
$ ls reports/
weekly_metrics_20251006.csv
developer_stats_20251006.csv
narrative_report_20251006.md
untracked_commits_20251006.csv

# Step 5: Periodic identity maintenance
$ gitflow-analytics identities -c config.yaml --weeks 1
[Check for new duplicates weekly...]
```

## Example 7: Advanced - Changing Launcher Defaults

**Modify config.yaml**:
```yaml
launcher:
  default_weeks: 12  # Analyze 12 weeks by default
  auto_clear_cache: true  # Always clear cache
  skip_identity_analysis: false  # Always run identity analysis
```

**Next run**:
```bash
$ gitflow-analytics run

[...]

📅 Number of weeks to analyze [12]: [Press Enter]

🗑️  Clear cache before analysis? [Y/n]: [Press Enter]

🔍 Skip identity analysis? [y/N]: [Press Enter]

[Uses new defaults automatically]
```

## Example 8: Troubleshooting - Config Not Found

```bash
$ gitflow-analytics run

🚀 GitFlow Analytics Interactive Launcher

❌ No configuration file found!

Searched for: config.yaml, config.yml, gitflow-config.yaml

💡 Run 'gitflow-analytics install' to create a configuration
```

**Solution**:
```bash
# Option 1: Install wizard
$ gitflow-analytics install

# Option 2: Specify config path
$ gitflow-analytics run -c /path/to/config.yaml
```

## Example 9: Identity Detection Without OpenRouter

```bash
$ gitflow-analytics identities -c config.yaml --weeks 12

🔍 Analyzing repositories for developer identities...
✅ Found 247 commits

⚠️  OpenRouter API key not configured
    Falling back to heuristic-only analysis

📄 Analysis report saved to: .gitflow-cache/identity_analysis_20251006.yaml

⚠️  Found 3 potential identity clusters:

📋 Suggested identity mappings:

   Cluster 1:
      Primary: john.doe@company.com
      Alias:   150280367+johndoe@users.noreply.github.com
      Method:  Heuristic (GitHub noreply pattern)

   Cluster 2:
      Primary: jane.smith@company.com
      Alias:   j.smith@company.com
      Method:  Heuristic (Name similarity 92%)

[Note: No confidence scores or LLM reasoning with heuristic analysis]
```

## Example 10: Rapid Testing Workflow

```bash
# Quick test with specific repos
$ gitflow-analytics run

Selection: 1,2  # Select subset
[... configure ...]

# Next run - same selection
$ gitflow-analytics run
Selection: [Enter]  # Reuse selection
[... quick re-run ...]

# Change to all repos
$ gitflow-analytics run
Selection: all
[... analyze everything ...]
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

### 4. Preference Optimization
- Set `default_weeks` to your most common period
- Enable `auto_clear_cache` only if needed
- Keep `skip_identity_analysis: false` for comprehensive analysis

### 5. Performance Tips
- Use launcher to select active repos only
- Skip identity analysis if recently run
- Clear cache when you need fresh data
- Use smaller week ranges for faster results

---

**See Also**:
- [Interactive Launcher Guide](docs/guides/interactive-launcher.md)
- [Enhanced Identity Resolution Guide](docs/guides/identity-resolution-enhanced.md)
- [Quick Reference](docs/quick-reference/launcher-and-identity.md)

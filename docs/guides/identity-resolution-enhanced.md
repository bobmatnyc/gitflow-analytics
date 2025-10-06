# Enhanced Identity Resolution Guide

GitFlow Analytics uses LLM-powered identity analysis to detect and consolidate duplicate developer identities with 90% confidence threshold.

## Overview

The enhanced identity resolution system:
- Uses LLM (GPT-4o-mini via OpenRouter) for intelligent duplicate detection
- Requires 90% confidence for identity matches (up from 80%)
- Displays confidence scores and reasoning for each suggestion
- Color-codes suggestions based on confidence level
- Preserves manual review and approval workflow

## Quick Start

```bash
# Analyze identities with enhanced detection
gitflow-analytics identities -c config.yaml --weeks 12

# Auto-apply high-confidence suggestions
gitflow-analytics identities -c config.yaml --apply
```

## How It Works

### 1. Identity Collection

The system analyzes commits from the specified time period and extracts all unique developer identities (name + email combinations).

### 2. Pre-Clustering

Heuristic-based pre-clustering groups obviously similar identities:
- Same name with different email domains
- GitHub noreply addresses
- Common name variations (J. Smith vs John Smith)

### 3. LLM Analysis

For pre-clustered groups, the LLM analyzes:
- Name similarity patterns
- Email address relationships
- Commit timing and patterns
- Repository overlap

### 4. Confidence Scoring

The LLM assigns a confidence score (0.0-1.0) to each identity cluster:
- **≥ 0.95**: Very high confidence (🟢 green indicator)
- **≥ 0.90**: High confidence (🟡 yellow indicator)
- **< 0.90**: Medium confidence (🟠 orange indicator, rejected by default)

### 5. Manual Review

High-confidence suggestions are presented for manual review with:
- Confidence percentage
- Reasoning explanation
- All email addresses in the cluster

## Example Output

```bash
$ gitflow-analytics identities -c config.yaml --weeks 12

🔍 Analyzing repositories for developer identities...
✅ Found 247 commits

📄 Analysis report saved to: .gitflow-cache/identity_analysis_20251006.yaml

⚠️  Found 3 potential identity clusters:

📋 Suggested identity mappings:

   🟢 Cluster 1 (Confidence: 95.3%):
      Primary: john.doe@company.com
      Alias:   150280367+johndoe@users.noreply.github.com
      Alias:   j.doe@gmail.com
      Reason:  Same person based on name patterns and commit timing correlation

   🟡 Cluster 2 (Confidence: 92.1%):
      Primary: jane.smith@company.com
      Alias:   jane.smith@contractor.com
      Reason:  Same developer using different email domains during contractor period

   🟡 Cluster 3 (Confidence: 91.5%):
      Primary: bob.wilson@company.com
      Alias:   150280368+bwilson@users.noreply.github.com
      Reason:  GitHub noreply address matches commit patterns and name

🤖 Found 2 bot accounts to exclude:
   - dependabot[bot]
   - renovate[bot]

Apply these identity mappings to your configuration? [Y/n]:
```

## Configuration

### Confidence Threshold

The default 90% confidence threshold is set in the analyzer initialization:

```python
# In src/gitflow_analytics/identity_llm/analyzer.py
confidence_threshold: float = 0.9  # 90% confidence required
```

To use a different threshold, modify your configuration:

```yaml
analysis:
  similarity_threshold: 0.95  # Require 95% confidence
```

### OpenRouter API Key

LLM analysis requires an OpenRouter API key:

```yaml
qualitative:
  enabled: true
  openrouter_api_key: ${OPENROUTER_API_KEY}
  model: "openai/gpt-4o-mini"
```

Or set in your `.env` file:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

### Manual Identity Mappings Format

Generated mappings include confidence and reasoning:

```yaml
analysis:
  manual_identity_mappings:
    - name: "John Doe"  # Optional display name override
      primary_email: "john.doe@company.com"
      aliases:
        - "150280367+johndoe@users.noreply.github.com"
        - "j.doe@gmail.com"
      confidence: 0.953
      reasoning: "Same person based on name patterns..."
```

**Note**: The `confidence` and `reasoning` fields are informational and not used during analysis - they help you understand why the LLM suggested the mapping.

## Fallback to Heuristic Analysis

If OpenRouter API key is not configured, the system falls back to heuristic-only analysis:
- Name similarity using fuzzy matching (85% threshold)
- Email domain analysis
- GitHub noreply address detection
- Bot account identification

Heuristic analysis is less accurate but requires no API key or costs.

## Confidence Level Guidelines

### 🟢 95%+ Confidence (Very High)
- **Typical Patterns**:
  - Same name with GitHub noreply address
  - Same person using personal and work email with similar names
  - Clear contractor-to-employee transitions
- **Recommendation**: Safe to auto-apply

### 🟡 90-95% Confidence (High)
- **Typical Patterns**:
  - Name variations (John vs J., Smith vs Smithe)
  - Email domain changes during company transitions
  - Multiple email addresses with consistent commit patterns
- **Recommendation**: Review and apply

### 🟠 <90% Confidence (Medium)
- **Typical Patterns**:
  - Significant name differences
  - Unrelated email domains
  - Inconsistent commit patterns
- **Recommendation**: Rejected automatically, manual investigation needed

## Best Practices

### 1. Regular Identity Analysis

Run identity analysis periodically to catch new duplicate patterns:
```bash
# Weekly identity check
gitflow-analytics identities -c config.yaml --weeks 1
```

### 2. Review Low-Confidence Suggestions

Check the full analysis report for rejected clusters:
```bash
cat .gitflow-cache/identity_analysis_YYYYMMDD.yaml
```

### 3. Manual Mapping Override

For cases you know are the same person but have low LLM confidence:
```yaml
analysis:
  manual_identity_mappings:
    - primary_email: "developer@company.com"
      aliases:
        - "freelance@gmail.com"
      # LLM may have rejected this, but you know they're the same
```

### 4. Bot Exclusion Verification

Always review bot exclusions before applying:
```yaml
analysis:
  exclude:
    authors:
      - "dependabot[bot]"
      - "renovate[bot]"
      # Verify these are actually bots, not developers with unusual names
```

### 5. Cache Management

Identity analysis results are cached for 7 days. To force re-analysis:
```bash
rm .gitflow-cache/identities.db
gitflow-analytics identities -c config.yaml
```

## Troubleshooting

### No OpenRouter API Key

```
⚠️  OpenRouter API key not configured
    Falling back to heuristic-only analysis
```

**Solution**: Configure OpenRouter API key in qualitative config section.

### Low Confidence Suggestions

```
⚠️  Found 5 potential identity clusters:
    (All rejected due to confidence < 90%)
```

**Solution**:
1. Review the full analysis report
2. Lower the confidence threshold temporarily if needed
3. Add manual mappings for known duplicates

### Incorrect Suggestions

If the LLM suggests incorrect mappings:
1. Don't apply the suggestion
2. Add explicit manual mappings to prevent future suggestions
3. Consider adjusting the confidence threshold

### API Rate Limiting

OpenRouter has rate limits. For large organizations:
1. Run identity analysis on smaller time periods
2. Use caching to avoid re-analyzing the same commits
3. Consider upgrading OpenRouter plan

## See Also

- [Interactive Launcher Guide](./interactive-launcher.md)
- [Configuration Guide](./configuration.md)
- [LLM Integration Guide](./llm-integration.md)

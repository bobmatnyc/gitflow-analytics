# Incremental Processing in GitFlow Analytics

GitFlow Analytics now includes intelligent incremental processing to avoid reprocessing data when schemas haven't changed. This significantly improves performance for repeated analyses.

## How It Works

### Schema Versioning

The system tracks the schema version for each component:

- **Qualitative Analysis**: NLP/LLM configuration and field definitions
- **GitHub API**: Rate limiting, field extraction, and PR/issue schemas  
- **JIRA API**: Story point fields, project configuration
- **Identity Resolution**: Manual mappings and similarity thresholds
- **Core**: Story points, ticket references, file changes

### Components Tracked

1. **Qualitative Data Processing**
   ```python
   # Schema includes NLP config, LLM settings, confidence thresholds
   schema = {
       'nlp_config': {...},
       'llm_config': {...}, 
       'confidence_threshold': 0.7,
       'max_llm_fallback_pct': 0.15
   }
   ```

2. **External API Data** (GitHub, JIRA)
   ```python
   # Schema includes API settings and field configurations
   github_schema = {
       'rate_limit_retries': 3,
       'backoff_factor': 2,
       'allowed_ticket_platforms': ['jira', 'github']
   }
   ```

3. **Identity Resolution**
   ```python
   # Schema includes manual mappings and analysis settings
   identity_schema = {
       'manual_mappings': [...],
       'similarity_threshold': 0.85,
       'auto_analysis': true
   }
   ```

## Performance Benefits

### Before Incremental Processing
```
Run 1: Process 10,000 commits + GitHub API calls + Qualitative analysis (5 minutes)
Run 2: Process same 10,000 commits + API calls + analysis (5 minutes) 
Run 3: Process same 10,000 commits + API calls + analysis (5 minutes)
```

### After Incremental Processing  
```
Run 1: Process 10,000 commits + GitHub API calls + Qualitative analysis (5 minutes)
Run 2: Process 500 new commits + incremental API calls (30 seconds)
Run 3: Process 200 new commits + incremental API calls (15 seconds)
```

## When Full Reprocessing Occurs

The system automatically triggers full reprocessing when:

1. **Configuration Changes**
   - NLP model settings change
   - LLM confidence thresholds change  
   - Story point field configurations change
   - Rate limiting settings change

2. **Schema Updates**
   - New fields added to qualitative analysis
   - API response format changes
   - Database schema migrations

3. **Manual Override**
   - `--clear-cache` flag used
   - Schema reset via CLI

## Example Scenarios

### Scenario 1: Daily Analysis Run
```bash
# First run - full processing
gitflow-analytics analyze -c config.yaml --weeks 4
# ✅ Processed 2,000 commits, 500 PRs, qualitative analysis (2 minutes)

# Next day - incremental processing  
gitflow-analytics analyze -c config.yaml --weeks 4
# ⚡ Processed 50 new commits, 12 new PRs, incremental analysis (10 seconds)
```

### Scenario 2: Configuration Change
```yaml
# Original config
qualitative:
  confidence_threshold: 0.7
  
# Updated config  
qualitative:
  confidence_threshold: 0.8  # Changed!
```

```bash
gitflow-analytics analyze -c config.yaml --weeks 4
# 🔄 Qualitative schema changed, reprocessing all commits (2 minutes)
```

### Scenario 3: Adding New Data Sources
```yaml
# Added new JIRA story point field
jira_integration:
  story_point_fields:
    - "Story Points"
    - "customfield_10021"  # New field added
```

```bash
gitflow-analytics analyze -c config.yaml --weeks 4  
# 🔄 JIRA schema changed, fetching all data since start date
```

## Technical Implementation

### Schema Hash Generation
Each component's schema is hashed including:
- Field definitions
- Configuration values  
- Processing parameters

```python
schema_hash = hashlib.sha256(
    json.dumps(schema_definition, sort_keys=True).encode()
).hexdigest()[:16]
```

### Date-Based Tracking
```sql
-- Schema versions table
CREATE TABLE schema_versions (
    component TEXT PRIMARY KEY,        -- 'qualitative', 'github', etc.
    version_hash TEXT NOT NULL,        -- Hash of current schema
    schema_definition TEXT NOT NULL,   -- JSON schema definition  
    created_at DATETIME,
    last_processed_date DATETIME       -- Last date processed with this schema
);
```

### Incremental Logic
```python
def should_process_data(component, date, config):
    # Check if schema changed
    if schema_manager.has_schema_changed(component, config):
        return True
        
    # Check if date is after last processed
    last_processed = schema_manager.get_last_processed_date(component)
    return date > last_processed if last_processed else True
```

## Monitoring and Debugging

### Schema Status Command
```bash
gitflow-analytics schema-status -c config.yaml
```

Output:
```
Component Status:
├── qualitative: ✅ Up to date (last processed: 2025-08-01)
├── github: ⚡ Incremental (since: 2025-07-30) 
├── identity: 🔄 Schema changed (will reprocess)
└── core: ✅ Up to date (last processed: 2025-08-01)
```

### Force Full Reprocessing
```bash
# Clear all schema tracking
gitflow-analytics analyze -c config.yaml --clear-cache

# Reset specific component
gitflow-analytics reset-schema -c config.yaml --component qualitative
```

## Benefits Summary

1. **Faster Analysis**: 10-20x speedup for incremental runs
2. **Reduced API Calls**: Only fetch new data since last run
3. **Cost Savings**: Fewer LLM API calls for qualitative analysis
4. **Automatic**: No manual configuration required
5. **Safe**: Automatically detects when full reprocessing is needed
6. **Transparent**: Clear logging of what's being processed incrementally

This system ensures that GitFlow Analytics scales efficiently for daily use while maintaining accuracy and completeness of analysis.
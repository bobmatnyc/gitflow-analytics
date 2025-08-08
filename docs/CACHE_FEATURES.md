# Cache Features and Optimization Guide

This document describes GitFlow Analytics' comprehensive caching system and the new cache optimization features.

## Overview

GitFlow Analytics uses SQLite-based caching to dramatically improve performance on subsequent runs. The cache stores analyzed commit data, pull request information, and issue tracking data.

## New Cache Features

### 1. Cache Warming (`--warm-cache`)

Pre-loads all commits from your repositories into the cache for maximum performance on subsequent runs.

```bash
# Warm cache for 4 weeks of history
gitflow-analytics -c config.yaml --warm-cache --weeks 4

# Warm cache and run analysis
gitflow-analytics -c config.yaml --warm-cache --weeks 8
```

**Benefits:**
- Subsequent runs are dramatically faster (cache hit rates >95%)
- Ideal for CI/CD environments with repeated analysis
- Batch processing optimizes database performance

**When to use:**
- First run on a new system
- After clearing cache
- Before running multiple analyses on the same data

### 2. Cache Validation (`--validate-cache`)

Validates cache integrity and identifies potential issues.

```bash
# Validate cache only
gitflow-analytics -c config.yaml --validate-cache

# Validate cache and warm if needed
gitflow-analytics -c config.yaml --validate-cache --warm-cache
```

**Validation checks:**
- Missing required fields
- Duplicate entries
- Data integrity issues
- Very old entries (older than 2×TTL)
- Negative change counts

**Example output:**
```
✅ Cache validation passed
Cache contains 1,247 commits
Warning: Found 3 very old cache entries (older than 336h)
```

### 3. Enhanced Cache Statistics

Detailed cache performance metrics displayed at the end of every run.

**Rich Display (default):**
```
📊 Cache Performance Summary
  Total requests: 856
  Cache hits: 823 (96.1%)
  Cache misses: 33
  Time saved: 1.4 minutes

💾 Cache Storage  
  Cached commits: 1,247
  Database size: 12.3 MB
```

**Simple Display:**
```
📊 Cache Performance:
   - Total requests: 856
   - Cache hits: 823 (96.1%)
   - Cache misses: 33
   - Time saved: 1.4 minutes
   - Cached commits: 1,247
   - Database size: 12.3 MB
```

### 4. Debug Mode (`GITFLOW_DEBUG=1`)

Enhanced debugging output for cache operations and performance analysis.

```bash
# Enable debug mode for detailed cache logging
GITFLOW_DEBUG=1 gitflow-analytics -c config.yaml --weeks 2
```

**Debug output includes:**
- Individual cache hits/misses for each commit
- Bulk cache lookup statistics
- Progress bar tracking details
- Cache validation verbose output

**Example debug output:**
```
DEBUG: Cache HIT for a1b2c3d4 in /repos/myproject
DEBUG: Cache MISS for e5f6g7h8 in /repos/myproject
DEBUG: Bulk cache lookup - 95 hits, 5 misses for 100 commits
DEBUG: Batch: 100 commits, Progress: 100/856, Processed: 100
```

### 5. Fixed Progress Bar

Resolved the issue where progress bars would show incorrect totals (e.g., "190/95").

**Improvements:**
- Accurate progress tracking with safety checks
- Better batch processing visualization
- Real-time cache hit rate display
- Debug information in postfix

**Progress bar format:**
```
Analyzing myproject: 100%|████████| 856/856 [00:15<00:00, cache_hit_rate=96.1%, processed=856/856]
```

## Cache Architecture

### Database Structure

The cache uses three main SQLite tables:

1. **`cached_commits`** - Commit analysis results
2. **`pull_request_cache`** - PR metadata and metrics
3. **`issue_cache`** - Issue tracking data from various platforms

### Cache Location

- **Default:** `.gitflow-cache/` in config file directory
- **Commits:** `.gitflow-cache/gitflow_cache.db`
- **Identities:** `.gitflow-cache/identities.db`
- **ML Predictions:** `.gitflow-cache/ml_predictions.db` (if ML enabled)

### Cache TTL (Time To Live)

- **Default:** 168 hours (7 days)
- **Configuration:** `cache.ttl_hours` in config
- **Disable expiration:** Set to 0

```yaml
cache:
  ttl_hours: 336  # 2 weeks
  directory: .gitflow-cache
```

## Performance Optimization Tips

### 1. Cache Warming Strategy

For optimal performance, follow this workflow:

```bash
# Initial setup (first time)
gitflow-analytics -c config.yaml --warm-cache --weeks 12

# Daily/regular analysis (fast)
gitflow-analytics -c config.yaml --weeks 2

# Weekly deep analysis
gitflow-analytics -c config.yaml --weeks 8
```

### 2. Batch Size Optimization

The analyzer processes commits in batches for optimal memory usage and database performance:

- **Default batch size:** 1000 commits
- **Memory vs. Performance:** Larger batches use more memory but reduce database round-trips
- **Configurable** through analyzer initialization (advanced usage)

### 3. Cache Maintenance

Regular maintenance for optimal performance:

```bash
# Validate cache health monthly
gitflow-analytics -c config.yaml --validate-cache

# Clear old cache if needed
gitflow-analytics -c config.yaml --clear-cache

# Re-warm after major repository changes
gitflow-analytics -c config.yaml --warm-cache --clear-cache --weeks 8
```

## Troubleshooting

### Cache Issues

**Problem:** Cache validation fails with integrity errors
```bash
# Solution: Clear and rebuild cache
gitflow-analytics -c config.yaml --clear-cache --warm-cache
```

**Problem:** Poor cache hit rates (<50%)
```bash
# Solution: Warm cache for longer history
gitflow-analytics -c config.yaml --warm-cache --weeks 12
```

**Problem:** Large database file sizes
```bash
# Check cache statistics
gitflow-analytics -c config.yaml --validate-cache

# Consider reducing TTL if needed
# Edit config.yaml: cache.ttl_hours: 72  # 3 days
```

### Debug Mode Troubleshooting

Enable debug mode to identify specific issues:

```bash
GITFLOW_DEBUG=1 gitflow-analytics -c config.yaml --validate-cache
```

Common debug scenarios:
- **High cache miss rate:** Shows which commits are not cached
- **Progress bar issues:** Displays batch processing details
- **Performance problems:** Shows detailed timing information

## API Usage (Advanced)

For programmatic access to cache features:

```python
from gitflow_analytics.core.cache import GitAnalysisCache
from pathlib import Path

# Initialize cache
cache = GitAnalysisCache(Path(".gitflow-cache"))

# Get statistics
stats = cache.get_cache_stats()
print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
print(f"Database size: {stats['database_size_mb']:.1f} MB")

# Validate cache
validation = cache.validate_cache()
if not validation["is_valid"]:
    print("Cache validation failed:", validation["issues"])

# Warm cache
repo_paths = ["/path/to/repo1", "/path/to/repo2"]
warming_result = cache.warm_cache(repo_paths, weeks=4)
print(f"Cached {warming_result['commits_cached']} new commits")
```

## Configuration Examples

### High-Performance Configuration

```yaml
cache:
  ttl_hours: 336  # 2 weeks retention
  directory: .gitflow-cache

analysis:
  # Optimize for speed
  exclude_paths:
    - "*.log"
    - "node_modules/*"
    - ".git/*"
```

### CI/CD Configuration

```yaml
cache:
  ttl_hours: 0  # Never expire (for build reproducibility)
  directory: /cache/gitflow  # Persistent cache location
```

### Development Configuration

```yaml
cache:
  ttl_hours: 24  # Short retention for active development
  directory: .dev-cache
```

## Performance Benchmarks

Typical performance improvements with caching:

| Repository Size | First Run | Cached Run | Improvement |
|----------------|-----------|------------|-------------|
| Small (100 commits) | 5s | 1s | 5x faster |
| Medium (1K commits) | 30s | 3s | 10x faster |
| Large (10K commits) | 300s | 15s | 20x faster |

Cache warming adds initial overhead but provides consistent fast performance:

| Operation | Time | Cache Hit Rate |
|-----------|------|----------------|
| Cold cache | 100s | 0% |
| Warm cache | 120s | 0% (warming) |
| Subsequent runs | 5s | 95%+ |

The break-even point is typically after 2-3 runs on the same dataset.
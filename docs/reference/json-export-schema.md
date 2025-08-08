# Comprehensive JSON Export Schema

The GitFlow Analytics comprehensive JSON export provides a complete, structured data format that consolidates all analytics data into a single file optimized for web consumption and API integration.

## Export Structure

The JSON export contains the following main sections:

### 1. Metadata
```json
{
  "metadata": {
    "generated_at": "2025-08-04T01:58:17.226632+00:00",
    "format_version": "2.0.0",
    "generator": "GitFlow Analytics Comprehensive JSON Exporter",
    "analysis_period": {
      "start_date": "2025-06-09T01:58:17.226625+00:00",
      "end_date": "2025-08-04T01:58:17.226625+00:00",
      "weeks_analyzed": 8,
      "total_days": 56
    },
    "data_summary": {
      "total_commits": 50,
      "total_prs": 10,
      "total_developers": 3,
      "repositories_analyzed": 2,
      "projects_identified": 2,
      "repositories": ["repo1", "repo2"],
      "projects": ["FRONTEND", "BACKEND"]
    },
    "export_settings": {
      "anonymized": true,
      "timezone": "UTC"
    }
  }
}
```

### 2. Executive Summary
High-level metrics, trends, and insights:

```json
{
  "executive_summary": {
    "key_metrics": {
      "commits": {"total": 50, "trend_percent": 15.2, "trend_direction": "increasing"},
      "lines_changed": {"total": 2600, "trend_percent": 25.1, "trend_direction": "increasing"},
      "story_points": {"total": 20, "trend_percent": -5.0, "trend_direction": "decreasing"},
      "developers": {"total": 3, "active_percentage": 100.0},
      "pull_requests": {"total": 10, "trend_percent": 0.0, "trend_direction": "stable"},
      "ticket_coverage": {"percentage": 75.0, "quality_rating": "good"}
    },
    "performance_indicators": {
      "velocity": {"commits_per_week": 6.2, "story_points_per_week": 2.5},
      "quality": {"avg_commit_size": 52.0, "ticket_coverage_pct": 75.0},
      "collaboration": {"developers_per_project": 2.5, "cross_project_contributors": 2}
    },
    "trends": {
      "commits_trend": 15.2,
      "lines_trend": 25.1,
      "story_points_trend": -5.0,
      "prs_trend": 0.0
    },
    "anomalies": [
      {
        "type": "spike",
        "metric": "weekly_commits", 
        "value": 15,
        "expected": 6.2,
        "severity": "high",
        "week_index": 3
      }
    ],
    "wins": [
      {
        "category": "team",
        "title": "Balanced Team Contributions",
        "description": "All team members are actively contributing",
        "impact": "medium"
      }
    ],
    "concerns": [
      {
        "category": "process",
        "title": "Low Ticket Coverage",
        "description": "Only 25% of commits linked to tickets",
        "impact": "high",
        "recommendation": "Improve ticket referencing in commit messages"
      }
    ],
    "health_score": {
      "overall": 78.5,
      "components": {
        "activity_consistency": 85.0,
        "ticket_coverage": 75.0,
        "collaboration": 80.0,
        "code_quality": 70.0,
        "velocity": 82.5
      },
      "weights": {
        "activity_consistency": 0.3,
        "ticket_coverage": 0.25,
        "collaboration": 0.2,
        "code_quality": 0.15,
        "velocity": 0.1
      },
      "rating": "good"
    }
  }
}
```

### 3. Projects
Project-level data with health scores and contributor details:

```json
{
  "projects": {
    "FRONTEND": {
      "summary": {
        "total_commits": 25,
        "total_contributors": 2,
        "lines_changed": 1300,
        "story_points": 10,
        "files_touched": 45,
        "pull_requests": 5
      },
      "health_score": {
        "overall": 82.3,
        "components": {"activity": 85.0, "contributor_diversity": 80.0, "consistency": 82.0},
        "rating": "excellent"
      },
      "contributors": [
        {
          "id": "dev-1",
          "name": "Developer1",
          "commits": 15,
          "commits_percentage": 60.0,
          "lines_changed": 780,
          "role": "primary"
        }
      ],
      "activity_patterns": {
        "commits_per_week": 3.1,
        "peak_activity_day": "Tuesday",
        "commit_size_distribution": {
          "mean": 52.0,
          "median": 45.0,
          "small_commits": 10,
          "medium_commits": 12,
          "large_commits": 3
        }
      },
      "trends": {
        "commits_trend": 20.0,
        "lines_trend": 15.5,
        "contributors_trend": 0.0
      },
      "anomalies": [],
      "focus_metrics": {
        "primary_contributors": ["Developer1", "Developer2"],
        "contribution_distribution": {
          "gini_coefficient": 0.35,
          "concentration_level": "low",
          "top_contributor_percentage": 60.0,
          "contributor_count": 2
        }
      }
    }
  }
}
```

### 4. Developers
Comprehensive developer profiles:

```json
{
  "developers": {
    "dev-1": {
      "identity": {
        "name": "Developer1",
        "canonical_id": "dev-1",
        "primary_email": "dev1@example.com",
        "github_username": "alice-dev",
        "aliases_count": 1
      },
      "summary": {
        "total_commits": 20,
        "total_story_points": 8,
        "projects_contributed": 2,
        "first_seen": "2025-06-09T01:58:17+00:00",
        "last_seen": "2025-07-28T01:58:17+00:00",
        "days_active": 49
      },
      "health_score": {
        "overall": 78.2,
        "components": {"activity": 75.0, "consistency": 85.0, "engagement": 80.0},
        "rating": "good"
      },
      "projects": {
        "FRONTEND": {
          "commits": 12,
          "commits_percentage": 60.0,
          "lines_changed": 624,
          "story_points": 5,
          "first_commit": "2025-06-09T01:58:17+00:00",
          "last_commit": "2025-07-25T01:58:17+00:00",
          "days_active": 46
        }
      },
      "contribution_patterns": {
        "total_commits": 20,
        "avg_commit_size": 52.0,
        "commit_size_stddev": 15.2,
        "peak_hour": 14,
        "time_distribution": "afternoon_focused",
        "peak_day": "Wednesday",
        "work_pattern": "mostly_weekdays",
        "consistency_score": 85.0
      },
      "collaboration": {
        "projects_count": 2,
        "potential_collaborators": 2,
        "cross_project_work": true,
        "collaboration_score": 60
      },
      "achievements": [
        {
          "type": "productivity",
          "title": "High Productivity",
          "description": "20 commits in analysis period",
          "badge": "prolific_contributor"
        }
      ],
      "improvement_areas": [
        {
          "category": "process",
          "title": "Improve Ticket Linking",
          "description": "Only 40% of commits reference tickets",
          "priority": "medium",
          "suggestion": "Include ticket references in commit messages"
        }
      ],
      "activity_timeline": [
        {
          "week": "2025-06-09",
          "commits": 3,
          "lines_changed": 156,
          "projects": 1,
          "project_list": ["FRONTEND"]
        }
      ]
    }
  }
}
```

### 5. Workflow Analysis
Git workflow patterns and PM integration:

```json
{
  "workflow_analysis": {
    "branching_strategy": {
      "merge_commits": 5,
      "merge_rate_percent": 10.0,
      "strategy": "feature_branches",
      "complexity_rating": "low"
    },
    "commit_patterns": {
      "peak_hour": 14,
      "peak_hour_commits": 8,
      "time_distribution": {
        "morning_pct": 20.0,
        "afternoon_pct": 50.0,
        "evening_pct": 25.0,
        "night_pct": 5.0
      },
      "peak_day": "Wednesday",
      "peak_day_commits": 12,
      "weekday_pct": 85.0,
      "weekend_pct": 15.0
    },
    "pr_workflow": {
      "avg_lifetime_hours": 24.0,
      "median_lifetime_hours": 18.0,
      "avg_pr_size": 70.0,
      "median_pr_size": 65.0,
      "avg_review_comments": 1.5,
      "prs_with_reviews": 7,
      "review_rate_pct": 70.0
    },
    "git_pm_correlation": {
      "total_correlations": 25,
      "confidence_distribution": {"high": 15, "medium": 8, "low": 2},
      "confidence_rates": {"high_pct": 60.0, "medium_pct": 32.0, "low_pct": 8.0},
      "correlation_methods": {"exact_match": 15, "fuzzy_match": 8, "pattern_match": 2},
      "story_point_analysis": {"coverage_pct": 80.0, "accuracy_pct": 85.0},
      "platforms": ["jira", "github"]
    },
    "process_health": {
      "ticket_linking_rate": 75.0,
      "merge_commit_rate": 10.0,
      "commit_message_quality": {
        "avg_message_length_words": 8.5,
        "ticket_reference_rate_pct": 75.0,
        "conventional_commit_rate_pct": 45.0,
        "overall_rating": "good"
      }
    }
  }
}
```

### 6. Time Series Data
Chart-ready time series data:

```json
{
  "time_series": {
    "weekly": {
      "labels": ["2025-06-09", "2025-06-16", "2025-06-23", "2025-06-30"],
      "datasets": {
        "commits": {
          "label": "Commits",
          "data": [6, 7, 8, 5],
          "backgroundColor": "rgba(54, 162, 235, 0.2)",
          "borderColor": "rgba(54, 162, 235, 1)"
        },
        "lines_changed": {
          "label": "Lines Changed",
          "data": [312, 364, 416, 260],
          "backgroundColor": "rgba(255, 99, 132, 0.2)", 
          "borderColor": "rgba(255, 99, 132, 1)"
        },
        "story_points": {
          "label": "Story Points",
          "data": [2, 4, 0, 2],
          "backgroundColor": "rgba(75, 192, 192, 0.2)",
          "borderColor": "rgba(75, 192, 192, 1)"
        },
        "active_developers": {
          "label": "Active Developers",
          "data": [2, 3, 3, 2],
          "backgroundColor": "rgba(153, 102, 255, 0.2)",
          "borderColor": "rgba(153, 102, 255, 1)"
        }
      }
    },
    "daily": {
      "labels": ["2025-06-09", "2025-06-10", "2025-06-11"],
      "datasets": {
        "commits": {
          "label": "Daily Commits",
          "data": [2, 1, 3],
          "backgroundColor": "rgba(54, 162, 235, 0.1)",
          "borderColor": "rgba(54, 162, 235, 1)"
        }
      }
    }
  }
}
```

### 7. Insights
Quantitative and qualitative insights:

```json
{
  "insights": {
    "quantitative": [
      {
        "category": "productivity",
        "type": "metric",
        "title": "Weekly Commit Rate",
        "description": "Team averages 6.2 commits per week",
        "value": 6.2,
        "trend": "increasing",
        "priority": "medium"
      }
    ],
    "qualitative": [
      {
        "category": "code_quality",
        "type": "qualitative",
        "title": "Code Review Practices",
        "description": "Strong peer review culture with detailed feedback",
        "priority": "low",
        "confidence": 0.85
      }
    ],
    "prioritized": [
      {
        "category": "team",
        "type": "concern",
        "title": "Unbalanced Contributions",
        "description": "Work is concentrated among few developers",
        "priority": "high",
        "recommendation": "Consider distributing work more evenly"
      }
    ],
    "insight_categories": {
      "productivity": [/* insights */],
      "team": [/* insights */],
      "quality": [/* insights */]
    },
    "actionable_recommendations": [
      {
        "title": "Improve Ticket Coverage",
        "action": "Implement consistent ticket referencing in commits and PRs",
        "priority": "high",
        "category": "process",
        "expected_impact": "high"
      }
    ]
  }
}
```

### 8. Raw Data Summary
Reference data and quality metrics:

```json
{
  "raw_data": {
    "commits_sample": [/* First 5 commits */],
    "prs_sample": [/* First 3 PRs */],
    "developer_stats_schema": {
      "fields": ["canonical_id", "primary_name", "total_commits"],
      "sample_record": {/* Sample developer record */}
    },
    "dora_metrics": {/* Complete DORA metrics */},
    "data_quality": {
      "commits_with_timestamps": 50,
      "commits_with_projects": 48,
      "commits_with_tickets": 12,
      "developers_with_github": 3
    }
  }
}
```

### 9. PM Integration (Optional)
PM platform integration summary:

```json
{
  "pm_integration": {
    "platforms": ["jira", "azure"],
    "total_issues": 45,
    "story_point_coverage": 80.0,
    "correlations_count": 25,
    "correlation_quality": {
      "total_correlations": 25,
      "average_confidence": 0.82,
      "high_confidence_correlations": 15
    },
    "issue_types": {
      "story": 20,
      "bug": 15,
      "task": 10
    },
    "platform_summary": {
      "jira": {
        "total_issues": 30,
        "linked_issues": 22,
        "coverage_percentage": 73.3
      }
    }
  }
}
```

## Usage with Chart Libraries

The JSON structure is optimized for popular charting libraries:

### Chart.js Example
```javascript
const data = jsonExport.time_series.weekly;
const chartConfig = {
  type: 'line',
  data: {
    labels: data.labels,
    datasets: [data.datasets.commits, data.datasets.lines_changed]
  },
  options: {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }
};
```

### D3.js Example
```javascript
const weeklyData = jsonExport.time_series.weekly;
const commits = weeklyData.labels.map((date, i) => ({
  date: new Date(date),
  commits: weeklyData.datasets.commits.data[i]
}));
```

## Health Score Interpretation

Health scores range from 0-100:
- **90-100**: Excellent - Optimal performance
- **80-89**: Good - Strong performance with minor areas for improvement
- **60-79**: Fair - Acceptable performance but improvement recommended
- **40-59**: Needs Improvement - Several areas require attention
- **0-39**: Poor - Significant issues requiring immediate attention

## Anomaly Types

- **spike**: Unusually high activity (2x+ normal)
- **drop**: Unusually low activity (<30% normal)
- **concentration**: Uneven work distribution (Gini > 0.8)
- **volatility**: Inconsistent patterns (high standard deviation)

## Export Settings

- **anonymized**: When true, developer names and emails are anonymized
- **timezone**: All timestamps are normalized to UTC
- **format_version**: Schema version for compatibility checking
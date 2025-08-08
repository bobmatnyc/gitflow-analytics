# Commit Classification Training Guide

This guide explains how to use GitFlow Analytics' commit classification training system to create custom models tailored to your project's specific patterns and ticket types.

## Overview

The training system integrates with your existing PM platforms (JIRA, GitHub Issues, etc.) to automatically label commits based on their associated tickets, then trains machine learning models to classify commits without explicit ticket references.

## Prerequisites

1. **PM Platform Integration**: Must be configured and enabled
2. **Dependencies**: Install scikit-learn (`pip install scikit-learn`)
3. **Training Data**: Sufficient commits with ticket references (default minimum: 50)

## Quick Start

### 1. Configure PM Integration

First, ensure your configuration includes PM platform integration:

```yaml
# config.yaml
pm_integration:
  enabled: true
  primary_platform: "jira"
  platforms:
    jira:
      enabled: true
      config:
        base_url: "https://your-company.atlassian.net"
        username: "${JIRA_USERNAME}"
        api_token: "${JIRA_API_TOKEN}"
        project_keys: ["PROJ", "TEAM"]
```

### 2. Run Training

Train a model using your repository data:

```bash
# Basic training (12 weeks of data)
gitflow-analytics train -c config.yaml

# Extended training with more data
gitflow-analytics train -c config.yaml --weeks 24 --min-examples 100

# Save training data for inspection
gitflow-analytics train -c config.yaml --save-training-data

# Custom model configuration
gitflow-analytics train -c config.yaml \
  --model-type random_forest \
  --validation-split 0.25 \
  --session-name "production_model_v1"
```

### 3. View Training Results

Check training statistics and model performance:

```bash
# View training history
gitflow-analytics train-stats -c config.yaml
```

### 4. Use Trained Models

Once trained, models are automatically used in analysis:

```bash
# Analysis will use trained models when available
gitflow-analytics analyze -c config.yaml --weeks 4
```

## How It Works

### 1. Data Collection
- Extracts commits from configured repositories
- Matches commits with tickets via references (e.g., "PROJ-123")
- Collects ticket metadata from PM platforms

### 2. Label Mapping
The system maps PM platform ticket types to classification categories:

| PM Ticket Type | Classification Category |
|----------------|------------------------|
| Bug            | bug_fix               |
| Story          | feature               |
| Task           | maintenance           |
| Epic           | feature               |
| Improvement    | feature               |
| Documentation  | documentation         |
| Test           | test                  |
| Refactoring    | refactor             |
| Hotfix         | bug_fix              |
| Security       | bug_fix              |

### 3. Multi-Ticket Handling
When commits reference multiple tickets:
- Same category: High confidence
- Mixed categories: Uses priority order (bug_fix > feature > maintenance)
- Lower confidence for mixed categories

### 4. Model Training
- Feature extraction using TF-IDF vectorization
- Train/validation/test splits (default: 70/20/10)
- Cross-validation for robust performance estimates
- Model storage with versioning

### 5. Integration
- Trained models integrate seamlessly with existing ML categorization
- Falls back to rule-based classification if models unavailable
- Performance monitoring and usage statistics

## Configuration Options

### Training Parameters

```bash
# Minimum training examples required
--min-examples 50

# Validation data fraction  
--validation-split 0.2

# Model algorithm
--model-type random_forest  # or svm, naive_bayes

# Time period to analyze
--weeks 12

# Incremental training (add to existing data)
--incremental

# Export training data as CSV
--save-training-data
```

### Category Mapping

Customize ticket type to category mapping in training configuration:

```python
# In training pipeline configuration
category_mapping = {
    'Bug': 'bug_fix',
    'Story': 'feature', 
    'Custom Task Type': 'maintenance',
    # Add your custom mappings
}
```

## Best Practices

### Data Quality
1. **Consistent Ticket References**: Ensure commits consistently reference tickets
2. **Ticket Type Accuracy**: Keep PM platform ticket types accurate and consistent
3. **Sufficient History**: Use at least 12-24 weeks of data for training
4. **Regular Retraining**: Retrain models periodically as patterns evolve

### Model Performance
1. **Monitor Accuracy**: Aim for >80% validation accuracy
2. **Category Balance**: Ensure reasonable distribution across categories
3. **Validation**: Review training data CSV exports for quality
4. **Testing**: Validate model predictions on recent commits

### Integration
1. **Gradual Rollout**: Test trained models on historical data first
2. **Fallback Strategy**: Rule-based classification provides backup
3. **Performance Monitoring**: Track model usage and accuracy
4. **Version Management**: Keep multiple model versions for rollback

## Troubleshooting

### Common Issues

**"Insufficient training data"**
- Increase `--weeks` to analyze more history
- Ensure commits reference ticket IDs (e.g., PROJ-123)
- Check PM platform connectivity and permissions
- Verify ticket reference patterns match your format

**"PM integration must be enabled"**
- Add `pm_integration` section to configuration
- Enable at least one PM platform
- Verify credentials and permissions

**"No trained models available"**
- Run training command first
- Check for training errors in logs
- Verify model files exist in cache directory

**Poor model accuracy**
- Increase training data size
- Review category mapping accuracy
- Check ticket type consistency
- Consider data quality issues

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
gitflow-analytics train -c config.yaml --log DEBUG
```

## Example Workflow

Here's a complete workflow for setting up commit classification:

```bash
# 1. Initial training with debugging
gitflow-analytics train -c config.yaml \
  --weeks 24 \
  --min-examples 100 \
  --save-training-data \
  --log INFO

# 2. Review training data quality
# Check the exported CSV file for label accuracy

# 3. Check training statistics
gitflow-analytics train-stats -c config.yaml

# 4. Test with analysis
gitflow-analytics analyze -c config.yaml --weeks 2

# 5. Monitor and retrain as needed
# Retrain monthly or when patterns change
gitflow-analytics train -c config.yaml --incremental
```

## Performance Metrics

The training system provides comprehensive metrics:

- **Overall Accuracy**: Percentage of correct predictions
- **Per-Category Metrics**: Precision, recall, F1-score for each category
- **Cross-Validation**: Robust performance estimates
- **Confusion Matrix**: Detailed error analysis
- **Training Coverage**: Percentage of commits with tickets

## Advanced Usage

### Custom Model Types

Extend the training pipeline to support additional model types:

```python
# In training pipeline
if model_type == 'custom_classifier':
    classifier = YourCustomClassifier(parameters)
```

### Feature Engineering

Add custom features for better accuracy:

```python
# In feature extraction
def extract_custom_features(commit):
    return {
        'has_breaking_change': 'BREAKING' in commit['message'],
        'author_experience': calculate_author_experience(commit['author']),
        # Add domain-specific features
    }
```

### Model Ensembles

Combine multiple models for better performance:

```python
# Advanced: Ensemble multiple models
def ensemble_predict(models, message):
    predictions = [model.predict(message) for model in models]
    return majority_vote(predictions)
```

## Integration with CI/CD

Automate training as part of your development workflow:

```yaml
# GitHub Actions example
name: Retrain Classification Model
on:
  schedule:
    - cron: '0 0 1 * *'  # Monthly
  
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install gitflow-analytics scikit-learn
      - name: Train model
        run: |
          gitflow-analytics train -c .github/config.yaml \
            --weeks 12 \
            --min-examples 50
        env:
          JIRA_USERNAME: ${{ secrets.JIRA_USERNAME }}
          JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
```

This completes the comprehensive training system for GitFlow Analytics commit classification!
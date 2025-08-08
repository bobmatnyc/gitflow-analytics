# Commit Classification System

The GitFlow Analytics commit classification system uses machine learning to automatically categorize git commits into meaningful types such as features, bug fixes, refactoring, documentation, and more. This provides deeper insights into development patterns and team productivity.

## Overview

The classification system consists of several key components:

1. **LinguistAnalyzer**: Analyzes file changes to determine programming languages and development activities
2. **FeatureExtractor**: Extracts 68-dimensional feature vectors from commit data
3. **CommitClassificationModel**: Random Forest classifier with model persistence
4. **CommitClassifier**: Main orchestrator that coordinates the entire pipeline

## Features

### 68-Dimensional Feature Extraction

The system extracts comprehensive features from each commit:

- **Keyword Features (20 dimensions)**: Semantic analysis of commit messages across 20 categories
- **File-based Features (20 dimensions)**: Programming languages, development activities, file patterns
- **Commit Statistics (15 dimensions)**: Size metrics, complexity indicators, formatting patterns  
- **Temporal Features (8 dimensions)**: Time-based patterns (hour, day, weekend, business hours)
- **Author Features (5 dimensions)**: Developer behavior and experience indicators

### Supported Classification Categories

The system can classify commits into 15 categories:

| Category | Description |
|----------|-------------|
| `feature` | New functionality or capabilities |
| `bugfix` | Bug fixes and error corrections |
| `refactor` | Code restructuring and optimization |
| `docs` | Documentation changes and updates |
| `test` | Testing-related changes |
| `config` | Configuration and settings changes |
| `chore` | Maintenance and housekeeping tasks |
| `security` | Security-related changes |
| `hotfix` | Emergency production fixes |
| `style` | Code style and formatting changes |
| `build` | Build system and dependency changes |
| `ci` | Continuous integration changes |
| `revert` | Reverts of previous changes |
| `merge` | Merge commits and integration |
| `wip` | Work in progress commits |

### File Analysis Capabilities

The LinguistAnalyzer provides comprehensive file analysis:

- **50+ Programming Languages**: Python, JavaScript, Java, Go, TypeScript, and more
- **10+ Development Activities**: UI, API, database, testing, documentation, infrastructure
- **Generated File Detection**: Automatically identifies compiled, minified, and generated files
- **Multi-language Projects**: Handles projects with multiple programming languages

## Configuration

Add the commit classification configuration to your `config.yaml`:

```yaml
analysis:
  commit_classification:
    enabled: true
    confidence_threshold: 0.6
    batch_size: 100
    auto_retrain: true
    retrain_threshold_days: 30
    
    model:
      n_estimators: 100
      max_depth: 20
      min_samples_split: 5
      min_samples_leaf: 2
      random_state: 42
      n_jobs: -1
    
    feature_extraction:
      enable_temporal_features: true
      enable_author_features: true
      enable_file_analysis: true
    
    training:
      validation_split: 0.2
      min_training_samples: 20
      cross_validation_folds: 5
      class_weight: "balanced"
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable commit classification |
| `confidence_threshold` | float | `0.5` | Minimum confidence for reliable predictions |
| `batch_size` | integer | `100` | Number of commits processed per batch |
| `auto_retrain` | boolean | `true` | Automatically check if model needs retraining |
| `retrain_threshold_days` | integer | `30` | Days after which to suggest retraining |

#### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | `100` | Number of trees in the Random Forest |
| `max_depth` | `20` | Maximum depth of decision trees |
| `min_samples_split` | `5` | Minimum samples required to split a node |
| `min_samples_leaf` | `2` | Minimum samples required at a leaf node |
| `random_state` | `42` | Random seed for reproducible results |
| `n_jobs` | `-1` | Number of CPU cores to use (-1 = all) |

## Usage

### Basic Classification

```python
from gitflow_analytics.classification import CommitClassifier

# Create classifier with configuration
config = {
    'enabled': True,
    'confidence_threshold': 0.6,
    'batch_size': 100
}

classifier = CommitClassifier(config=config)

# Classify commits
results = classifier.classify_commits(commit_data)

for result in results:
    print(f"Commit: {result['commit_hash']}")
    print(f"Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Primary Language: {result['file_analysis']['primary_language']}")
```

### Training a Custom Model

```python
# Prepare training data as list of (commit_data, label) tuples
training_data = [
    (commit1, 'feature'),
    (commit2, 'bugfix'),
    (commit3, 'docs'),
    # ... more training examples
]

# Train the model
training_results = classifier.train_model(training_data, validation_split=0.2)

print(f"Training accuracy: {training_results['accuracy']:.3f}")
print(f"Cross-validation score: {training_results['cv_accuracy_mean']:.3f}")
```

### Analyzing Commit Patterns

```python
# Analyze patterns across a collection of commits
pattern_analysis = classifier.analyze_commit_patterns(commits)

print(f"Total commits: {pattern_analysis['total_commits_analyzed']}")
print(f"Classification distribution: {pattern_analysis['classification_distribution']}")
print(f"Average confidence: {pattern_analysis['average_confidence']:.3f}")
print(f"Most common class: {pattern_analysis['most_common_class']}")
```

### Feature Importance

```python
# Get the most important features for classification
importance = classifier.get_feature_importance(top_n=10)

print("Top 10 most important features:")
for feature_name, score in importance:
    print(f"  {feature_name}: {score:.4f}")
```

## Model Training

### Training Data Requirements

- **Minimum Samples**: At least 20 labeled examples per class
- **Balanced Classes**: Aim for roughly equal representation of each category
- **Quality Labels**: Accurate, consistent labeling is crucial for good performance
- **Diverse Examples**: Include commits from different developers, time periods, and projects

### Training Process

1. **Data Preparation**: Extract features from labeled commits
2. **Train/Validation Split**: Automatically splits data for validation
3. **Model Training**: Trains Random Forest classifier with cross-validation
4. **Performance Evaluation**: Calculates accuracy, precision, recall metrics
5. **Model Persistence**: Saves trained model for future use

### Model Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Overall Accuracy**: Percentage of correctly classified commits
- **Cross-Validation Score**: Average accuracy across k-fold validation
- **Per-Class Metrics**: Precision, recall, and F1-score for each category
- **Confusion Matrix**: Detailed breakdown of classification performance
- **Feature Importance**: Most influential features for classification

## Integration with GitFlow Analytics

The commit classification system integrates seamlessly with existing GitFlow Analytics workflows:

### Reports

Classification results are included in generated reports:

- **CSV Reports**: Additional columns with predicted class and confidence
- **Markdown Reports**: Classification summaries and pattern analysis
- **JSON Exports**: Complete classification metadata for API integration

### Caching

- **Model Cache**: Trained models are cached for reuse
- **Prediction Cache**: Classification results are cached to avoid recomputation
- **Feature Cache**: Extracted features are cached for batch processing efficiency

### Performance Optimization

- **Batch Processing**: Commits processed in configurable batches
- **Parallel Processing**: Uses multiple CPU cores for feature extraction
- **Graceful Degradation**: Falls back to rule-based classification if ML unavailable

## Troubleshooting

### Common Issues

#### "scikit-learn not available"

Install the required dependency:
```bash
pip install scikit-learn>=1.0.0
```

#### "Need at least 20 labeled examples"

Ensure you have sufficient training data:
- Minimum 20 examples per classification category
- Use diverse, well-labeled examples
- Consider using the export/import functionality to create training data

#### "Model may need retraining"

The model automatically checks if it needs retraining based on age:
- Models older than 30 days (configurable) show this warning
- Retrain with recent commit data to maintain accuracy
- Use `auto_retrain: false` to disable automatic checking

### Performance Optimization

#### Large Repositories

For large repositories with many commits:
- Increase `batch_size` to process more commits at once
- Use `n_jobs: -1` to utilize all CPU cores
- Consider filtering out merge commits and automated commits

#### Memory Usage

To reduce memory usage:
- Decrease `batch_size` if running out of memory
- Process commits in smaller time windows
- Use `enable_author_features: false` if author data is large

## Dependencies

The commit classification system requires:

- **scikit-learn** >= 1.0.0: Machine learning algorithms
- **numpy** >= 1.20.0: Numerical computing
- **pandas** >= 1.3.0: Data manipulation (optional, for advanced features)

Install with:
```bash
pip install scikit-learn numpy
```

## API Reference

### CommitClassifier

Main interface for commit classification.

#### Methods

- `classify_commits(commits)`: Classify a batch of commits
- `classify_single_commit(commit)`: Classify a single commit  
- `train_model(training_data, validation_split)`: Train the model
- `analyze_commit_patterns(commits)`: Analyze patterns in commits
- `get_feature_importance(top_n)`: Get feature importance rankings
- `get_model_status()`: Get model status and capabilities

### FeatureExtractor

Extracts 68-dimensional features from commits.

#### Methods

- `extract_features(commit_data, author_stats)`: Extract features from single commit
- `extract_batch_features(commits, author_stats)`: Extract features from batch
- `get_feature_names()`: Get human-readable feature names

### LinguistAnalyzer

Analyzes file changes for language and activity detection.

#### Methods

- `analyze_commit_files(file_paths)`: Analyze files in a commit
- `get_language_category(language)`: Get high-level language category
- `get_supported_languages()`: List supported programming languages
- `get_supported_activities()`: List supported development activities

## Examples

See the `test_classification_system.py` script for comprehensive examples of using the commit classification system.

## Future Enhancements

Planned improvements to the classification system:

1. **Deep Learning Models**: Experiment with neural networks for improved accuracy
2. **Active Learning**: Semi-supervised learning to reduce labeling requirements  
3. **Multi-label Classification**: Support commits that span multiple categories
4. **Custom Categories**: Allow users to define project-specific categories
5. **Integration with Issue Trackers**: Automatically classify based on linked issues
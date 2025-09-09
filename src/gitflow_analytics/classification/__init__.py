"""Commit classification module for GitFlow Analytics.

This module provides machine learning-based commit classification capabilities,
analyzing commits to categorize them into meaningful types such as:
- Feature development
- Bug fixes
- Refactoring
- Documentation
- Testing
- Configuration changes
- And more

The classification system uses a combination of:
- File pattern analysis via LinguistAnalyzer
- 68-dimensional feature extraction
- Random Forest classification
- Temporal and author-based features

Usage:
    from gitflow_analytics.classification import CommitClassifier

    classifier = CommitClassifier(config)
    predictions = classifier.predict_batch(commits)
"""

from .classifier import CommitClassifier
from .feature_extractor import FeatureExtractor
from .linguist_analyzer import LinguistAnalyzer
from .model import CommitClassificationModel

__all__ = ["CommitClassifier", "FeatureExtractor", "LinguistAnalyzer", "CommitClassificationModel"]

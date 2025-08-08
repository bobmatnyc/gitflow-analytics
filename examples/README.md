# Examples Directory

This directory contains example configurations and utility scripts to help you get started with GitFlow Analytics.

## Contents

### config/
Sample configuration files demonstrating various use cases:

- **config-sample.yaml** - Basic configuration template
- **config-sample-ml.yaml** - Configuration with ML-enhanced commit categorization
- **config-qualitative-sample.yaml** - Configuration with qualitative analysis features
- **config-pm-sample.yaml** - Configuration with project management tool integration
- **config-jira-resilient.yaml** - Robust JIRA integration configuration
- **config-classification-example.yaml** - Commit classification example
- **config-recess.yaml** - Example using recess-recreo test repositories

### scripts/
Utility scripts for debugging and configuration management:

- **debug_jira_auth.py** - Test and debug JIRA authentication
- **fix_pm_config.py** - Fix project management configuration issues
- **validate_classification_integration.py** - Validate classification system integration

## Usage

1. Copy a sample configuration file from `config/` to your project root
2. Rename it to match your needs (e.g., `config.yaml`)
3. Modify the configuration according to your repository setup
4. Run GitFlow Analytics: `gitflow-analytics -c config.yaml --weeks 8`

For more detailed instructions, see the main [README.md](../README.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
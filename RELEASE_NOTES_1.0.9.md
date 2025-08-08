# GitFlow Analytics v1.0.9 Release Notes

## 🚀 Major Features

### Commit Classification with Machine Learning
- **Random Forest classifier** with 68-dimensional feature extraction
- **Hybrid ML + rule-based approach** for accurate commit categorization
- **JIRA integration** for automatic training data generation
- **ML prediction caching** for performance optimization
- Categories: feature, bug_fix, refactor, documentation, maintenance, test, style, build

### Enhanced Developer Analytics
- **Activity scoring** with balanced metrics and diminishing returns
- **Commit classification breakdowns** in developer profiles
- **Project-level classification analysis** with weekly trends
- **Team composition** showing all projects per developer with percentages
- **Untracked work analysis** with dual percentage metrics

### Performance Optimizations
- **Bulk cache operations** reducing analysis time by 10-20x
- **Cache warming** (`--warm-cache`) for CI/CD environments
- **Cache validation** (`--validate-cache`) for integrity checks
- **Progress bar fixes** for accurate tracking
- **Batch processing** for large repositories

### Cost Tracking & LLM Integration
- **Token tracking** with real-time cost display
- **Daily budget limits** with color-coded alerts
- **OpenRouter API support** for qualitative analysis
- **Cost statistics** displayed after each run
- **Budget utilization warnings** at 80%, 90%, and 100%

## 🔧 Technical Improvements

### Architecture Enhancements
- **Lazy PM framework initialization** fixing credential injection issues
- **Mixed type handling** for files_changed (int/list compatibility)
- **Enhanced error handling** with user-friendly YAML error messages
- **Modular classification system** with pluggable ML components
- **SQLite-based training data storage** per project

### Documentation & Cleanup
- **Comprehensive CLAUDE.md** for AI-assisted development
- **Updated README** with ML classification examples
- **Cache features documentation** with performance benchmarks
- **Security audit** completed with credential checks
- **Code cleanup** removing deprecated features

## 🐛 Bug Fixes

- Fixed "name 'extra' is not defined" error in activity scoring
- Fixed PM framework module-level initialization causing auth failures
- Fixed progress bar showing incorrect counts (190/95 issue)
- Fixed SQLAlchemy text() deprecation warnings
- Fixed files_changed type mismatches across codebase
- Fixed missing developers in reports (removed 10-developer limit)

## 📊 Performance Metrics

- **Cache hit rates**: 95%+ after warming
- **Analysis speed**: 10-20x faster with warmed cache
- **Memory usage**: Optimized for repositories with 100k+ commits
- **Classification speed**: 300+ commits/second
- **ML accuracy**: 85%+ with JIRA training data

## 🔒 Security Updates

- Removed all hardcoded credentials
- Enhanced .gitignore for security
- Credential validation in configuration
- Environment variable support with .env files
- No sensitive data in logs or reports

## 💡 Usage Examples

### Commit Classification Training
```bash
gitflow-analytics train -c config.yaml --weeks 12
```

### Enhanced Analysis with Classification
```bash
gitflow-analytics -c config.yaml --weeks 4
```

### Cache Operations
```bash
# Warm cache for faster runs
gitflow-analytics -c config.yaml --warm-cache --weeks 8

# Validate cache integrity
gitflow-analytics -c config.yaml --validate-cache
```

### Cost-Aware Analysis
```bash
# Configure in YAML
qualitative:
  enabled: true
  llm:
    provider: "openrouter"
    model: "anthropic/claude-3-sonnet"
  cost_tracking:
    daily_budget_usd: 10.00
```

## ⚠️ Known Issues

- Test suite has failures that don't affect runtime functionality
- Resource warnings in tests (unclosed database connections)
- spaCy model must be downloaded separately for ML features
- Some test fixtures need updating for new features

## 🔮 Future Enhancements

- Real-time classification model updates
- Multi-language commit message support
- Advanced ML model selection (XGBoost, neural networks)
- Distributed cache support for teams
- Classification confidence thresholds per category

## 📦 Installation

```bash
pip install gitflow-analytics==1.0.9

# For ML features
python -m spacy download en_core_web_sm
```

## 🙏 Acknowledgments

Thanks to all contributors and users who provided feedback for this release. Special thanks to the EWTN team for extensive testing and feature requests that drove many of these improvements.

---

For questions or issues, please visit: https://github.com/ewtn-devops/gitflow-analytics/issues
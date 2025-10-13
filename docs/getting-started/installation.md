# Installation Guide

This guide walks you through installing GitFlow Analytics and setting up the basic requirements.

## 🚀 Quick Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Install the latest stable version
pip install gitflow-analytics

# Or install with pipx (recommended for CLI tools)
pipx install gitflow-analytics

# Verify installation
gitflow-analytics --version
```

### Option 2: Install from GitHub (Development)

```bash
# Install latest development version
pip install git+https://github.com/bobmatnyc/gitflow-analytics.git

# Or clone and install locally
git clone https://github.com/bobmatnyc/gitflow-analytics.git
cd gitflow-analytics
pip install -e ".[dev]"
```

## 📋 System Requirements

### Python Version
- **Python 3.9 or higher** (3.11+ recommended)
- Check your version: `python --version`

### Operating System Support
- ✅ **Linux** (Ubuntu 18.04+, CentOS 7+)
- ✅ **macOS** (10.14+)
- ✅ **Windows** (10+, PowerShell or WSL recommended)

### Hardware Requirements
- **4GB+ RAM** (8GB+ recommended for large repositories)
- **2GB+ disk space** for dependencies and cache
- **Git 2.20+** for repository analysis

### Dependencies
GitFlow Analytics will automatically install required dependencies:
- `PyGitHub` - GitHub API integration
- `GitPython` - Git repository analysis
- `pandas` - Data processing
- `pyyaml` - Configuration handling
- `rich` - Beautiful terminal output
- `click` - Command-line interface

## 🔑 Authentication Setup

### GitHub Personal Access Token

1. **Create a GitHub Token**:
   - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click "Generate new token (classic)"
   - Select scopes: `repo`, `read:org` (for organization analysis)
   - Copy the generated token

2. **Set Environment Variable**:
   ```bash
   # Option 1: Set in your shell profile (.bashrc, .zshrc, etc.)
   export GITHUB_TOKEN="ghp_your_token_here"

   # Option 2: Create a .env file (recommended)
   echo "GITHUB_TOKEN=ghp_your_token_here" > .env
   ```

3. **Verify Token**:
   ```bash
   # Test GitHub API access
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
   ```

### PM Platform Setup (Optional)

GitFlow Analytics supports multiple project management platforms for enhanced ticket tracking:

- **JIRA** - Enterprise project management
- **Linear** - Modern issue tracking
- **ClickUp** - All-in-one productivity
- **GitHub Issues** - Native GitHub integration (auto-configured with GitHub token)

**Quick Setup**: Use the interactive launcher:
```bash
gitflow-analytics launcher
```

Select **Profile 1 (Standard)** and choose which PM platforms to configure.

**Manual Setup**: See the [PM Platform Setup Guide](../guides/pm-platform-setup.md) for detailed instructions on obtaining and configuring credentials for each platform.

## 🧪 Verify Installation

### Basic Verification
```bash
# Check version
gitflow-analytics --version

# View help
gitflow-analytics --help

# Test with a public repository (no token required)
gitflow-analytics --help analyze
```

### Complete Test Run
```bash
# Create a test configuration
cat > test-config.yaml << EOF
github:
  token: "${GITHUB_TOKEN}"
  repositories:
    - owner: "octocat"
      name: "Hello-World"
      local_path: "./test-repo"
EOF

# Run a quick test analysis (this will clone Hello-World repo)
gitflow-analytics -c test-config.yaml --weeks 4 --validate-only
```

## 🔧 Optional: ML Features Setup

If you want to use machine learning categorization features:

```bash
# Install spaCy language model
python -m spacy download en_core_web_sm

# Verify spaCy installation
python -c "import spacy; print('spaCy installed successfully')"
```

## 📁 Recommended Directory Structure

Set up a clean workspace:

```bash
# Create project directory
mkdir gitflow-analysis
cd gitflow-analysis

# Create subdirectories
mkdir config repos reports cache

# Your structure should look like:
# gitflow-analysis/
# ├── config/          # Configuration files
# ├── repos/           # Cloned repositories  
# ├── reports/         # Generated reports
# └── cache/           # Analysis cache
```

## 🐛 Troubleshooting

### Common Issues

**"Command not found: gitflow-analytics"**
```bash
# If installed with pip, ensure pip binary location is in PATH
python -m pip show gitflow-analytics

# If installed with pipx, ensure pipx bin directory is in PATH
pipx list
```

**"Permission denied" errors**
```bash
# On Unix systems, you might need to adjust permissions
chmod +x ~/.local/bin/gitflow-analytics

# Or install with --user flag
pip install --user gitflow-analytics
```

**"GitHub API rate limit exceeded"**
- Ensure your GitHub token is set correctly
- Authenticated requests have higher rate limits (5000/hour vs 60/hour)

**"SSL Certificate verification failed"**
```bash
# Corporate networks might need certificate verification disabled
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org gitflow-analytics
```

**"dyld: Library not loaded" (macOS) or broken pipx installation**
```bash
# This error occurs when Homebrew upgrades Python, breaking pipx virtual environments
# The error typically shows: "Library not loaded: /opt/homebrew/Cellar/python@3.12..."

# Solution 1: Recreate the pipx virtual environment (recommended)
pipx uninstall gitflow-analytics
pipx install gitflow-analytics

# Solution 2: If uninstall fails, force reinstall all pipx packages
pipx reinstall-all

# Solution 3: Manual cleanup if the above fail
rm -rf ~/.local/pipx/venvs/gitflow-analytics
rm -rf ~/.local/pipx/shared
pipx install gitflow-analytics

# Explanation: When Homebrew upgrades Python (e.g., 3.12.11_1 → 3.12.12),
# existing pipx virtual environments still reference the old Python version,
# causing dynamic library loading failures. Recreating the venv fixes this.
```

### Dependency Issues

**"No module named 'git'"**
```bash
# Ensure Git is installed and accessible
git --version

# On Ubuntu/Debian
sudo apt-get install git

# On macOS with Homebrew
brew install git
```

**spaCy model download fails**
```bash
# Download model manually
python -m spacy download en_core_web_sm

# Or download alternative model
python -m spacy download en_core_web_md
```

### Platform-Specific Notes

**Windows Users**:
- Use PowerShell or Windows Subsystem for Linux (WSL)
- Ensure Python is in your PATH
- Use forward slashes in paths: `./repos/project` not `.\\repos\\project`

**macOS Users**:
- You might need to install Xcode command line tools: `xcode-select --install`
- Consider using Homebrew for Python: `brew install python`

**Linux Users**:
- Some distributions require separate `python3-dev` package
- Ubuntu/Debian: `sudo apt-get install python3-dev`
- CentOS/RHEL: `sudo yum install python3-devel`

## ✅ Installation Complete!

You're ready to move on to the [Quick Start Tutorial](quickstart.md).

If you encountered any issues, please check our [Troubleshooting Guide](../guides/troubleshooting.md) or [file an issue](https://github.com/bobmatnyc/gitflow-analytics/issues).

## 🔄 What's Next?

- **[Quick Start Tutorial](quickstart.md)** - 5-minute walkthrough
- **[Your First Analysis](first-analysis.md)** - Run your first repository analysis  
- **[Configuration Guide](../guides/configuration.md)** - Learn about advanced configuration options
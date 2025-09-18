#!/bin/bash
# Development version launcher for GitFlow Analytics
# This ensures we always use the development version from the editable install

# Check if we're in the right environment
if ! command -v gitflow-analytics &> /dev/null; then
    echo "❌ GitFlow Analytics not found. Please install with:"
    echo "   pip install -e /Users/masa/Projects/managed/gitflow-analytics"
    exit 1
fi

# Check version to ensure we're using the development version
VERSION=$(gitflow-analytics --version 2>/dev/null | grep -o "version [0-9.]*" | cut -d' ' -f2)
if [[ "$VERSION" != "2.1.0" ]]; then
    echo "⚠️  Warning: Using version $VERSION instead of expected 2.1.0"
    echo "   Consider reinstalling with:"
    echo "   pip uninstall -y gitflow-analytics && pip install -e /Users/masa/Projects/managed/gitflow-analytics"
fi

# Run the command with all arguments passed through
exec gitflow-analytics "$@"

#!/usr/bin/env python3
"""
Test script to demonstrate the loading screen functionality.

This script can be used to test the loading screen without running a full
GitFlow Analytics analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from gitflow_analytics.tui.app import GitFlowAnalyticsApp
    
    def test_loading_screen():
        """Test the loading screen functionality."""
        print("🚀 Starting GitFlow Analytics TUI with loading screen test...")
        print("✨ The loading screen should appear immediately showing:")
        print("   - Loading spinner animation")
        print("   - Progress bar with startup steps")
        print("   - Status messages for each loading phase")
        print("   - Configuration discovery (if config files exist)")
        print("   - spaCy model loading (if qualitative analysis enabled)")
        print("")
        print("⌨️  Press Ctrl+C during loading to test cancellation")
        print("🎯 After loading, the main screen should appear with loaded configuration")
        print("")
        print("Loading GitFlow Analytics TUI...")
        
        app = GitFlowAnalyticsApp()
        app.run()
    
    if __name__ == "__main__":
        test_loading_screen()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Make sure to run this from the project root directory")
    print("📁 Expected directory structure:")
    print("   gitflow-analytics/")
    print("   ├── src/gitflow_analytics/...")
    print("   └── test_loading_screen.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
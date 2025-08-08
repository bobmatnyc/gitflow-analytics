#!/usr/bin/env python3
"""Debug JIRA authentication in PM framework."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gitflow_analytics.config import ConfigLoader
from gitflow_analytics.pm_framework.adapters.jira_adapter import JIRAAdapter

# Load config
config_path = Path.home() / "Clients/EWTN/gfa/config.yaml"
config = ConfigLoader.load(config_path)

print("=== Debugging JIRA Authentication ===")
print(f"Config path: {config_path}")

# Check environment variables
print(f"\nEnvironment variables loaded:")
print(f"JIRA_ACCESS_USER: {os.getenv('JIRA_ACCESS_USER', 'NOT SET')}")
print(f"JIRA_ACCESS_TOKEN length: {len(os.getenv('JIRA_ACCESS_TOKEN', ''))}")

# Check what's in the PM integration config
if hasattr(config, 'pm_integration'):
    print(f"\nPM Integration enabled: {config.pm_integration.enabled}")
    if hasattr(config.pm_integration, 'platforms'):
        jira_config = config.pm_integration.platforms['jira']
        print(f"JIRA platform enabled: {jira_config.enabled}")
        print(f"JIRA config type: {type(jira_config.config)}")
        
        # Get the actual config values - config is a dict
        jira_params = jira_config.config
        print(f"\nJIRA config parameters:")
        print(f"  base_url: {jira_params.get('base_url', 'NOT SET')}")
        print(f"  username: {jira_params.get('username', 'NOT SET')}")
        print(f"  api_token: {'SET' if jira_params.get('api_token') else 'NOT SET'}")
        print(f"  api_token length: {len(jira_params.get('api_token', ''))}")
        
        # Try to create JIRA adapter directly
        print("\nTrying to create JIRA adapter...")
        try:
            adapter = JIRAAdapter(jira_params)
            print(f"✓ Adapter created")
            print(f"  Adapter base_url: {adapter.base_url}")
            print(f"  Adapter username: {adapter.username}")
            print(f"  Adapter api_token length: {len(adapter.api_token)}")
            
            # Try to authenticate
            print("\nTrying to authenticate...")
            if adapter.authenticate():
                print("✅ Authentication successful!")
            else:
                print("❌ Authentication failed")
        except Exception as e:
            print(f"❌ Failed to create adapter: {e}")
            import traceback
            traceback.print_exc()
else:
    print("PM integration not found in config!")
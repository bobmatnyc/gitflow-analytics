#!/usr/bin/env python3
"""Fix PM integration config by expanding environment variables."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
config_path = Path.home() / "Clients/EWTN/gfa/config.yaml"
env_path = config_path.parent / ".env"
load_dotenv(env_path)

# Read config
with open(config_path) as f:
    config = yaml.safe_load(f)

# Fix PM integration config
if 'pm_integration' in config and 'platforms' in config['pm_integration']:
    jira_config = config['pm_integration']['platforms']['jira']['config']
    
    # Expand environment variables
    if jira_config['username'] == '${JIRA_ACCESS_USER}':
        jira_config['username'] = os.getenv('JIRA_ACCESS_USER')
    
    if jira_config['api_token'] == '${JIRA_ACCESS_TOKEN}':
        jira_config['api_token'] = os.getenv('JIRA_ACCESS_TOKEN')

# Save back
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✅ Fixed PM integration config with actual credentials")
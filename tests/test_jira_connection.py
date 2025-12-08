#!/usr/bin/env python3
"""Test JIRA connection."""

import base64
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
env_path = Path.home() / "Clients/EWTN/gfa/.env"
load_dotenv(env_path)

username = os.getenv("JIRA_ACCESS_USER")
api_token = os.getenv("JIRA_ACCESS_TOKEN")
base_url = "https://ewtn.atlassian.net"

print(f"Username: {username}")
print(f"Token length: {len(api_token) if api_token else 0}")
print(f"Base URL: {base_url}")

# Test connection
credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
headers = {"Authorization": f"Basic {credentials}", "Accept": "application/json"}

try:
    response = requests.get(f"{base_url}/rest/api/3/myself", headers=headers)
    print(f"\nResponse status: {response.status_code}")

    if response.status_code == 200:
        user_data = response.json()
        print(f"✅ Successfully authenticated as: {user_data.get('displayName', 'Unknown')}")
        print(f"   Email: {user_data.get('emailAddress', 'N/A')}")
    else:
        print(f"❌ Authentication failed: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")

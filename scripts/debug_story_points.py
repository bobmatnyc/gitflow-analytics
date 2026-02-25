#!/usr/bin/env python3
"""Debug script to check story points in JIRA tickets."""

import os
import sys

import requests
from requests.auth import HTTPBasicAuth

# Add src to path
sys.path.insert(0, "src")


def check_jira_credentials():
    """Check if JIRA credentials are available."""
    user = os.getenv("JIRA_ACCESS_USER")
    token = os.getenv("JIRA_ACCESS_TOKEN")

    if not user or not token:
        print("❌ JIRA credentials not found!")
        print("   Set JIRA_ACCESS_USER and JIRA_ACCESS_TOKEN environment variables")
        return False

    print(f"✅ JIRA credentials found for user: {user}")
    return True


def get_ticket_details(ticket_id):
    """Get detailed information about a JIRA ticket."""
    user = os.getenv("JIRA_ACCESS_USER")
    token = os.getenv("JIRA_ACCESS_TOKEN")
    base_url = "https://ewtn.atlassian.net"

    try:
        response = requests.get(
            f"{base_url}/rest/api/3/issue/{ticket_id}",
            auth=HTTPBasicAuth(user, token),
            params={"fields": "*all"},
            timeout=30,
        )

        if response.status_code == 200:
            issue = response.json()
            fields = issue.get("fields", {})

            # Check all the story point fields we're configured to look for
            story_point_fields = [
                "customfield_10016",  # Primary field from config
                "customfield_10063",  # Backup field from config
                "Story Points",  # Field name fallback
                "timeestimate",  # Remaining Estimate
                "timeoriginalestimate",  # Original estimate
            ]

            print(f"\n🎫 Ticket: {ticket_id}")
            print(f"   Summary: {fields.get('summary', 'No summary')}")
            print(f"   Status: {fields.get('status', {}).get('name', 'Unknown')}")
            print(f"   Issue Type: {fields.get('issuetype', {}).get('name', 'Unknown')}")

            print("\n📊 Story Point Fields:")
            found_story_points = False

            for field_id in story_point_fields:
                if field_id in fields:
                    value = fields[field_id]
                    print(f"   ✅ {field_id}: {value} (type: {type(value).__name__})")
                    if value is not None and value != 0:
                        found_story_points = True
                else:
                    print(f"   ❌ {field_id}: Not found")

            # Also check for any field that might contain "story" or "point"
            print("\n🔍 All fields containing 'story' or 'point':")
            for field_id, value in fields.items():
                if field_id.startswith("customfield_") and value is not None:
                    # Get field name if possible
                    field_name = f"customfield_{field_id.split('_')[-1]}"
                    if any(term in str(value).lower() for term in ["story", "point"]) or any(
                        term in field_id.lower() for term in ["story", "point"]
                    ):
                        print(f"   🔍 {field_id}: {value} (type: {type(value).__name__})")

            if not found_story_points:
                print("   ⚠️  No story points found in any configured field")

            return True

        else:
            print(f"❌ Failed to fetch {ticket_id}: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False

    except Exception as e:
        print(f"❌ Error fetching {ticket_id}: {e}")
        return False


def main():
    """Debug story points in JIRA tickets."""
    print("🔍 Debug Story Points in JIRA Tickets")
    print("=" * 50)

    if not check_jira_credentials():
        return

    # Sample tickets from the analysis output
    sample_tickets = ["RMVP-884", "RMVP-648", "RMVP-611", "RMVP-660", "RMVP-627"]

    print(f"\n📋 Checking {len(sample_tickets)} sample tickets...")

    success_count = 0
    for ticket_id in sample_tickets:
        if get_ticket_details(ticket_id):
            success_count += 1
        print("-" * 50)

    print(f"\n✅ Successfully checked {success_count}/{len(sample_tickets)} tickets")


if __name__ == "__main__":
    main()

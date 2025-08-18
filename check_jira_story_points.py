#!/usr/bin/env python3
"""
Quick script to check story points in the JIRA tickets that were fetched.
"""

import os
import sys
import requests
from requests.auth import HTTPBasicAuth

# JIRA configuration
JIRA_BASE_URL = "https://ewtn.atlassian.net"
JIRA_USER = os.getenv('JIRA_ACCESS_USER')
JIRA_TOKEN = os.getenv('JIRA_ACCESS_TOKEN')

# Story point fields to check
STORY_POINT_FIELDS = [
    "customfield_10063",  # Story Points (primary)
    "customfield_10016",  # Story point estimate (backup)
    "timeestimate",       # Remaining Estimate
    "timeoriginalestimate"  # Original estimate
]

# Sample tickets from the analysis output
SAMPLE_TICKETS = [
    "RMVP-458", "RMVP-648", "RMVP-941", "RMVP-515", "RMVP-537",
    "RMVP-997", "RMVP-554", "RMVP-274", "SITE-83", "RMVP-838"
]

def check_jira_credentials():
    """Check if JIRA credentials are available."""
    if not JIRA_USER or not JIRA_TOKEN:
        print("❌ JIRA credentials not found in environment variables")
        print("   Set JIRA_ACCESS_USER and JIRA_ACCESS_TOKEN")
        return False
    return True

def get_ticket_story_points(ticket_id):
    """Get story points for a specific ticket."""
    if not check_jira_credentials():
        return None
        
    try:
        auth = HTTPBasicAuth(JIRA_USER, JIRA_TOKEN)
        response = requests.get(
            f"{JIRA_BASE_URL}/rest/api/2/issue/{ticket_id}",
            auth=auth,
            timeout=10
        )
        
        if response.status_code == 200:
            issue = response.json()
            fields = issue.get('fields', {})
            
            # Check each story point field
            story_points_info = {}
            for field_id in STORY_POINT_FIELDS:
                if field_id in fields:
                    value = fields[field_id]
                    story_points_info[field_id] = value
            
            return {
                'ticket_id': ticket_id,
                'summary': fields.get('summary', 'No summary'),
                'status': fields.get('status', {}).get('name', 'Unknown'),
                'issue_type': fields.get('issuetype', {}).get('name', 'Unknown'),
                'story_points_fields': story_points_info,
                'story_points': extract_story_points(story_points_info)
            }
        else:
            return {
                'ticket_id': ticket_id,
                'error': f"HTTP {response.status_code}",
                'story_points': None
            }
            
    except Exception as e:
        return {
            'ticket_id': ticket_id,
            'error': str(e),
            'story_points': None
        }

def extract_story_points(story_points_info):
    """Extract story points from the field values."""
    for field_id, value in story_points_info.items():
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                continue
    return None

def main():
    """Check story points in sample JIRA tickets."""
    print("🔍 Checking Story Points in JIRA Tickets")
    print("=" * 50)
    
    if not check_jira_credentials():
        return
    
    print(f"📋 Checking {len(SAMPLE_TICKETS)} sample tickets...")
    print()
    
    tickets_with_points = 0
    total_story_points = 0
    
    for ticket_id in SAMPLE_TICKETS:
        print(f"🎫 {ticket_id}:")
        ticket_info = get_ticket_story_points(ticket_id)
        
        if 'error' in ticket_info:
            print(f"   ❌ Error: {ticket_info['error']}")
        else:
            print(f"   📝 {ticket_info['summary'][:60]}...")
            print(f"   📊 Type: {ticket_info['issue_type']}")
            print(f"   🔄 Status: {ticket_info['status']}")
            
            if ticket_info['story_points']:
                print(f"   ✅ Story Points: {ticket_info['story_points']}")
                tickets_with_points += 1
                total_story_points += ticket_info['story_points']
            else:
                print(f"   ❌ No story points found")
                
            # Show which fields were checked
            fields_info = ticket_info.get('story_points_fields', {})
            if fields_info:
                print(f"   🔍 Fields checked:")
                for field_id, value in fields_info.items():
                    status = "✅" if value is not None else "❌"
                    print(f"      {status} {field_id}: {value}")
        
        print()
    
    print("=" * 50)
    print("📊 Summary:")
    print(f"   Total tickets checked: {len(SAMPLE_TICKETS)}")
    print(f"   Tickets with story points: {tickets_with_points}")
    print(f"   Total story points: {total_story_points}")
    print(f"   Success rate: {tickets_with_points/len(SAMPLE_TICKETS)*100:.1f}%")
    
    if tickets_with_points == 0:
        print()
        print("💡 Recommendations:")
        print("   1. Check if story points are being set in JIRA tickets")
        print("   2. Verify the correct story point field IDs in your JIRA instance")
        print("   3. Consider using JIRA admin tools to find the correct custom field IDs")
        print("   4. Test with tickets that you know have story points assigned")

if __name__ == "__main__":
    main()

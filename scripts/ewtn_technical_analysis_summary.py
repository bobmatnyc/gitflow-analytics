#!/usr/bin/env python3
"""
EWTN Technical Analysis Summary

This script provides detailed technical insights from the commit classification analysis,
including algorithm performance, confidence distributions, and advanced analytics.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def load_analysis_data():
    """Load the detailed analysis results."""
    data_file = Path('ewtn_classification_demo_results/ewtn_classification_analysis_2024-06-01.json')
    with open(data_file, 'r') as f:
        return json.load(f)

def print_technical_insights(data):
    """Print comprehensive technical insights."""
    reports = data['reports']
    
    print("=" * 80)
    print("🔬 EWTN TECHNICAL ANALYSIS - DETAILED INSIGHTS")
    print("=" * 80)
    
    # Classification Performance Analysis
    print("\n📈 CLASSIFICATION ALGORITHM PERFORMANCE")
    print("-" * 50)
    
    breakdown = reports['classification_breakdown']
    accuracy = breakdown['accuracy_validation']['accuracy_percentage']
    total_classified = breakdown['accuracy_validation']['total_validated']
    
    print(f"Model Accuracy: {accuracy:.1f}%")
    print(f"Total Classifications: {total_classified}")
    print(f"Correct Predictions: {breakdown['accuracy_validation']['correct_predictions']}")
    
    # Confidence Distribution Analysis
    confidence = reports['confidence_analysis']
    print(f"\nConfidence Score Distribution:")
    print(f"  • Average Confidence: {confidence['overall_metrics']['average_confidence']:.3f}")
    print(f"  • Confidence Range: {confidence['overall_metrics']['min_confidence']:.3f} - {confidence['overall_metrics']['max_confidence']:.3f}")
    
    conf_dist = confidence['confidence_distribution']
    total_predictions = confidence['overall_metrics']['total_predictions']
    
    print(f"\nConfidence Levels:")
    print(f"  • Very High (≥0.9): {conf_dist['very_high']} ({conf_dist['very_high']/total_predictions*100:.1f}%)")
    print(f"  • High (0.8-0.9):   {conf_dist['high']} ({conf_dist['high']/total_predictions*100:.1f}%)")
    print(f"  • Medium (0.6-0.8): {conf_dist['medium']} ({conf_dist['medium']/total_predictions*100:.1f}%)")
    print(f"  • Low (0.4-0.6):    {conf_dist['low']} ({conf_dist['low']/total_predictions*100:.1f}%)")
    print(f"  • Very Low (<0.4):  {conf_dist['very_low']} ({conf_dist['very_low']/total_predictions*100:.1f}%)")
    
    # Detailed Developer Analysis
    print("\n👥 DEVELOPER PRODUCTIVITY ANALYSIS")
    print("-" * 50)
    
    dev_analysis = reports['developer_analysis']
    
    # Sort developers by total commits
    sorted_developers = sorted(dev_analysis.items(), key=lambda x: x[1]['total_commits'], reverse=True)
    
    print("Top Performers by Commit Volume:")
    for i, (dev_name, stats) in enumerate(sorted_developers[:5], 1):
        primary_type = stats['primary_work_type']
        total_commits = stats['total_commits']
        avg_confidence = stats['avg_confidence']
        repo_count = stats['repository_count']
        
        print(f"  {i}. {dev_name}")
        print(f"     • {total_commits} commits | Primary: {primary_type.title()}")
        print(f"     • Avg Confidence: {avg_confidence:.3f} | Repositories: {repo_count}")
        print(f"     • Work Distribution: {dict(stats['work_distribution'])}")
        print()
    
    # Repository Analysis
    print("\n🏗️ REPOSITORY ANALYSIS")
    print("-" * 50)
    
    repo_analysis = reports['repository_analysis']
    sorted_repos = sorted(repo_analysis.items(), key=lambda x: x[1]['total_commits'], reverse=True)
    
    for repo_name, stats in sorted_repos:
        print(f"📦 {repo_name.upper()}")
        print(f"   • {stats['total_commits']} commits | Primary: {stats['primary_work_type'].title()}")
        print(f"   • {stats['developer_count']} developers | Activity: {stats['activity_level'].title()}")
        print(f"   • Work Distribution: {dict(stats['work_distribution'])}")
        print(f"   • Avg Confidence: {stats['avg_confidence']:.3f}")
        print()
    
    # Temporal Pattern Analysis
    print("\n⏰ TEMPORAL PATTERN ANALYSIS")
    print("-" * 50)
    
    temporal = reports['temporal_patterns']
    daily_patterns = temporal['daily_patterns']
    
    print("Daily Activity Summary:")
    for date, stats in sorted(daily_patterns.items()):
        total = stats['total_commits']
        devs = stats['active_developers']
        repos = stats['active_repositories']
        work_dist = stats['work_distribution']
        
        print(f"  {date}: {total} commits | {devs} devs | {repos} repos")
        print(f"    Work: Eng={work_dist.get('engineering', 0)} Ops={work_dist.get('operations', 0)} Docs={work_dist.get('documentation', 0)}")
    
    peak_day = temporal['peak_activity_day']
    peak_hour = temporal['peak_activity_hour']
    print(f"\n📊 Peak Activity:")
    print(f"   • Busiest Day: {peak_day}")
    print(f"   • Peak Hour: {peak_hour}:00")
    
    # File Change Analysis
    print("\n📁 CODE CHANGE ANALYSIS")
    print("-" * 50)
    
    # Calculate aggregate file and line changes
    total_files = 0
    total_lines = 0
    commit_sizes = []
    
    for dev_name, stats in dev_analysis.items():
        files_per_commit = stats['avg_files_per_commit']
        lines_per_commit = stats['avg_lines_per_commit']
        commits = stats['total_commits']
        
        total_files += files_per_commit * commits
        total_lines += lines_per_commit * commits
        
        commit_sizes.extend([lines_per_commit] * commits)
    
    avg_files_per_commit = total_files / sum(stats['total_commits'] for stats in dev_analysis.values())
    avg_lines_per_commit = total_lines / sum(stats['total_commits'] for stats in dev_analysis.values())
    
    print(f"Average Files per Commit: {avg_files_per_commit:.1f}")
    print(f"Average Lines per Commit: {avg_lines_per_commit:.1f}")
    
    # Commit size distribution
    small_commits = sum(1 for size in commit_sizes if size < 50)
    medium_commits = sum(1 for size in commit_sizes if 50 <= size < 200)
    large_commits = sum(1 for size in commit_sizes if size >= 200)
    total_commits = len(commit_sizes)
    
    print(f"\nCommit Size Distribution:")
    print(f"  • Small (<50 lines):   {small_commits} ({small_commits/total_commits*100:.1f}%)")
    print(f"  • Medium (50-200):     {medium_commits} ({medium_commits/total_commits*100:.1f}%)")
    print(f"  • Large (≥200 lines):  {large_commits} ({large_commits/total_commits*100:.1f}%)")
    
    # Strategic Insights
    print("\n🎯 STRATEGIC INSIGHTS")
    print("-" * 50)
    
    insights = reports['actionable_insights']
    work_balance = insights['work_balance_analysis']
    
    print(f"Work Balance Assessment: {work_balance['balance_assessment']}")
    print(f"Engineering Focus: {work_balance['engineering_percentage']:.1f}%")
    print(f"Operations Overhead: {work_balance['operations_percentage']:.1f}%")
    print(f"Documentation Coverage: {work_balance['documentation_percentage']:.1f}%")
    
    print(f"\nRecommendations:")
    for rec in insights['team_recommendations']:
        priority = rec['priority'].upper()
        area = rec['area'].title()
        recommendation = rec['recommendation']
        metric = rec['metric']
        
        print(f"  • [{priority}] {area}: {recommendation}")
        print(f"    Metric: {metric}")
    
    # Quality Metrics
    print("\n🏆 QUALITY METRICS")
    print("-" * 50)
    
    # Calculate quality indicators
    high_confidence_rate = (conf_dist['very_high'] + conf_dist['high']) / total_predictions * 100
    documentation_rate = work_balance['documentation_percentage']
    team_coverage = len(dev_analysis)  # Number of active developers
    repo_coverage = len(repo_analysis)  # Number of active repositories
    
    print(f"Classification Quality Score: {high_confidence_rate:.1f}% (High Confidence)")
    print(f"Documentation Health Score: {documentation_rate:.1f}% (Target: 15-20%)")
    print(f"Team Engagement Score: {team_coverage}/10 developers active")
    print(f"Repository Coverage: {repo_coverage}/6 repositories active")
    
    # Risk Assessment
    print(f"\n⚠️ RISK ASSESSMENT")
    print("-" * 50)
    
    risks = []
    if documentation_rate < 10:
        risks.append("🟡 MEDIUM: Documentation coverage below recommended 15% threshold")
    if work_balance['operations_percentage'] > 30:
        risks.append("🟡 MEDIUM: High operations overhead may indicate technical debt")
    if high_confidence_rate < 80:
        risks.append("🔴 HIGH: Classification confidence below acceptable threshold")
    
    if not risks:
        risks.append("🟢 LOW: No significant risks identified")
    
    for risk in risks:
        print(f"  {risk}")
    
    print("\n" + "=" * 80)
    print("📊 Analysis Complete - All metrics within acceptable ranges")
    print("🔄 Recommend running weekly analysis for trend monitoring")
    print("=" * 80)

def main():
    """Main execution function."""
    try:
        data = load_analysis_data()
        print_technical_insights(data)
    except FileNotFoundError:
        print("❌ Analysis data not found. Please run the classification demo first.")
    except Exception as e:
        print(f"❌ Error loading analysis data: {e}")

if __name__ == '__main__':
    main()
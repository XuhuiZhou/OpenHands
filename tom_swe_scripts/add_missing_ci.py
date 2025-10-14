#!/usr/bin/env python3
"""
Add missing confidence intervals to a user_satisfaction.json file
"""
import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append('/home/xuhuizhou/OpenHands')
from evaluation.benchmarks.swe_bench.eval_user_satisfaction import calculate_confidence_intervals, calculate_task_success_correlation, calculate_contingency_analysis

def add_missing_ci(file_path: str):
    """Add CI data to a file that's missing it"""

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return False

    # Load the existing data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Find the main result entry
    main_key = None
    main_data = None
    for key, value in data.items():
        if value and 'detailed_results' in value:
            main_key = key
            main_data = value
            break

    if not main_data:
        print("Error: No detailed_results found in file")
        return False

    detailed_results = main_data['detailed_results']

    # Check if CI already exists
    if 'confidence_intervals' in main_data:
        print("Confidence intervals already exist in this file")
        return False

    print(f"Computing confidence intervals for {len(detailed_results)} results...")

    # Calculate confidence intervals
    ci_data = calculate_confidence_intervals(detailed_results)

    # Calculate correlation analysis if not present
    correlation_data = None
    if 'correlation_analysis' not in main_data:
        print("Computing correlation analysis...")
        correlation_data = calculate_task_success_correlation(detailed_results)

    # Calculate contingency analysis if not present
    contingency_data = None
    if 'contingency_analysis' not in main_data:
        print("Computing contingency analysis...")
        contingency_data = calculate_contingency_analysis(detailed_results)

    # Add the new data to the structure
    if ci_data:
        main_data['confidence_intervals'] = ci_data
        print("✓ Added confidence intervals")

    if correlation_data:
        main_data['correlation_analysis'] = correlation_data
        print("✓ Added correlation analysis")

    if contingency_data:
        main_data['contingency_analysis'] = contingency_data
        print("✓ Added contingency analysis")

    # Write back to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Successfully updated {file_path}")
    return True

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python add_missing_ci.py <path_to_user_satisfaction.json>")
        sys.exit(1)

    file_path = sys.argv[1]
    success = add_missing_ci(file_path)
    sys.exit(0 if success else 1)
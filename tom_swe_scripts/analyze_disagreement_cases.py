#!/usr/bin/env python3
"""
Analyze concrete examples of disagreement cases (F+High and S+Med)
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import random

def extract_disagreement_examples():
    """Extract concrete examples of F+High and S+Med cases"""

    base_path = "/home/xuhuizhou/OpenHands/evaluation/evaluation_outputs/outputs/cmu-lti__stateful-test"
    disagreement_cases = {
        'f_high': [],  # Failed + High satisfaction
        's_med': []    # Success + Medium satisfaction
    }

    configurations = []

    # Find all user_satisfaction.json files
    for root, dirs, files in os.walk(base_path):
        if "user_satisfaction.json" in files:
            file_path = os.path.join(root, "user_satisfaction.json")

            # Extract agent and model from path
            path_parts = root.replace(base_path + "/", "").split("/")
            agent = path_parts[0]
            model_config = path_parts[1]

            # Parse model config
            if "qwen3-coder-480b" in model_config:
                model = "qwen3-coder-480b"
            elif "claude-sonnet-4-20250514" in model_config:
                model = "claude-sonnet-4"
            elif "claude-3-7-sonnet-20250219" in model_config:
                model = "claude-3.7-sonnet"
            else:
                model = "unknown"

            # Check for RAG
            rag = "rag" in model_config
            rag_suffix = "+RAG" if rag else ""
            config_name = f"{agent}-{model}{rag_suffix}"

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Get detailed results
                for file_key, file_data in data.items():
                    if file_data and 'detailed_results' in file_data:
                        detailed_results = file_data['detailed_results']

                        for result in detailed_results:
                            # Access the correct structure
                            instance_id = result.get('instance_id', 'unknown')
                            metrics = result.get('metrics', {})
                            task_success = result.get('task_resolved', False)
                            overall_sat = metrics.get('overall_satisfaction', 0)

                            # Failed + High satisfaction (>3.5)
                            if not task_success and overall_sat > 3.5:
                                disagreement_cases['f_high'].append({
                                    'config': config_name,
                                    'instance_id': instance_id,
                                    'overall_satisfaction': overall_sat,
                                    'satisfaction_details': metrics,
                                    'task_resolved': task_success,
                                    'file_path': file_path
                                })

                            # Success + Medium satisfaction (2.0-3.5)
                            elif task_success and 2.0 <= overall_sat <= 3.5:
                                disagreement_cases['s_med'].append({
                                    'config': config_name,
                                    'instance_id': instance_id,
                                    'overall_satisfaction': overall_sat,
                                    'satisfaction_details': metrics,
                                    'task_resolved': task_success,
                                    'file_path': file_path
                                })

                        configurations.append(config_name)
                        break

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return disagreement_cases, list(set(configurations))

def analyze_themes(disagreement_cases: Dict[str, List[Dict[str, Any]]]):
    """Analyze and categorize themes in disagreement cases"""

    print("DISAGREEMENT CASE ANALYSIS")
    print("=" * 100)

    f_high_cases = disagreement_cases['f_high']
    s_med_cases = disagreement_cases['s_med']

    print(f"Total F+High cases (Failed + High satisfaction): {len(f_high_cases)}")
    print(f"Total S+Med cases (Success + Medium satisfaction): {len(s_med_cases)}")

    # Analyze F+High cases
    print(f"\n1. FAILED + HIGH SATISFACTION CASES ({len(f_high_cases)} total)")
    print("-" * 70)

    # Group by configuration
    f_high_by_config = {}
    for case in f_high_cases:
        config = case['config']
        if config not in f_high_by_config:
            f_high_by_config[config] = []
        f_high_by_config[config].append(case)

    for config, cases in sorted(f_high_by_config.items()):
        print(f"\n{config}: {len(cases)} cases")

        # Sample a few cases for detailed analysis
        sample_cases = random.sample(cases, min(3, len(cases)))
        for i, case in enumerate(sample_cases):
            satisfaction = case['satisfaction_details']
            print(f"  Example {i+1}: {case['instance_id']}")
            print(f"    Overall Satisfaction: {case['overall_satisfaction']:.1f}/5")
            print(f"    Communication Quality: {satisfaction.get('communication_quality', 'N/A'):.1f}/5")
            print(f"    Problem Solving: {satisfaction.get('problem_solving_approach', 'N/A'):.1f}/5")
            print(f"    Efficiency: {satisfaction.get('efficiency', 'N/A'):.1f}/5")
            print(f"    User Preference: {satisfaction.get('user_preference_alignment', 'N/A'):.1f}/5")
            print(f"    File: {case['file_path'].split('/')[-3]}/{case['file_path'].split('/')[-2]}")

    # Analyze S+Med cases
    print(f"\n\n2. SUCCESS + MEDIUM SATISFACTION CASES ({len(s_med_cases)} total)")
    print("-" * 70)

    # Group by configuration
    s_med_by_config = {}
    for case in s_med_cases:
        config = case['config']
        if config not in s_med_by_config:
            s_med_by_config[config] = []
        s_med_by_config[config].append(case)

    for config, cases in sorted(s_med_by_config.items()):
        print(f"\n{config}: {len(cases)} cases")

        # Sample a few cases for detailed analysis
        sample_cases = random.sample(cases, min(3, len(cases)))
        for i, case in enumerate(sample_cases):
            satisfaction = case['satisfaction_details']
            print(f"  Example {i+1}: {case['instance_id']}")
            print(f"    Overall Satisfaction: {case['overall_satisfaction']:.1f}/5")
            print(f"    Communication Quality: {satisfaction.get('communication_quality', 'N/A'):.1f}/5")
            print(f"    Problem Solving: {satisfaction.get('problem_solving_approach', 'N/A'):.1f}/5")
            print(f"    Efficiency: {satisfaction.get('efficiency', 'N/A'):.1f}/5")
            print(f"    User Preference: {satisfaction.get('user_preference_alignment', 'N/A'):.1f}/5")
            print(f"    File: {case['file_path'].split('/')[-3]}/{case['file_path'].split('/')[-2]}")

    return f_high_by_config, s_med_by_config

def create_summary_table(f_high_by_config: Dict[str, List[Dict]], s_med_by_config: Dict[str, List[Dict]], configurations: List[str]):
    """Create summary table of disagreement patterns"""

    print(f"\n\n3. DISAGREEMENT SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Configuration':<35} | {'F+High Count':<12} | {'F+High %':<10} | {'S+Med Count':<12} | {'S+Med %':<10}")
    print("-" * 100)

    summary_data = []

    for config in sorted(configurations):
        f_high_count = len(f_high_by_config.get(config, []))
        s_med_count = len(s_med_by_config.get(config, []))

        # Calculate percentages (we'll need to get total failed/success counts)
        # For now, just show raw counts
        summary_data.append({
            'config': config,
            'f_high_count': f_high_count,
            's_med_count': s_med_count
        })

        print(f"{config:<35} | {f_high_count:>11} | {'N/A':>9} | {s_med_count:>11} | {'N/A':>9}")

    return summary_data

def analyze_satisfaction_patterns(f_high_by_config: Dict[str, List[Dict]], s_med_by_config: Dict[str, List[Dict]]):
    """Analyze satisfaction component patterns"""

    print(f"\n\n4. SATISFACTION COMPONENT ANALYSIS")
    print("=" * 100)

    # Analyze F+High cases - what makes users happy despite failure?
    print("F+High Cases - What drives high satisfaction despite failure:")
    print("-" * 70)

    all_f_high = []
    for cases in f_high_by_config.values():
        all_f_high.extend(cases)

    if all_f_high:
        # Calculate average satisfaction components
        components = ['communication_quality', 'problem_solving_approach', 'efficiency', 'user_preference_alignment']

        for component in components:
            values = [case['satisfaction_details'].get(component, 0) for case in all_f_high if component in case['satisfaction_details']]
            if values:
                avg_value = sum(values) / len(values)
                print(f"  Average {component.replace('_', ' ').title()}: {avg_value:.2f}/5")

    # Analyze S+Med cases - what limits satisfaction despite success?
    print("\nS+Med Cases - What limits satisfaction despite success:")
    print("-" * 70)

    all_s_med = []
    for cases in s_med_by_config.values():
        all_s_med.extend(cases)

    if all_s_med:
        for component in components:
            values = [case['satisfaction_details'].get(component, 0) for case in all_s_med if component in case['satisfaction_details']]
            if values:
                avg_value = sum(values) / len(values)
                print(f"  Average {component.replace('_', ' ').title()}: {avg_value:.2f}/5")

def main():
    """Main execution"""
    print("ANALYZING DISAGREEMENT CASES: F+High and S+Med")
    print("=" * 100)

    # Set random seed for reproducible examples
    random.seed(42)

    # Extract examples
    disagreement_cases, configurations = extract_disagreement_examples()

    # Analyze themes
    f_high_by_config, s_med_by_config = analyze_themes(disagreement_cases)

    # Create summary table
    summary_data = create_summary_table(f_high_by_config, s_med_by_config, configurations)

    # Analyze satisfaction patterns
    analyze_satisfaction_patterns(f_high_by_config, s_med_by_config)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")

if __name__ == '__main__':
    main()
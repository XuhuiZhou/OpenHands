#!/usr/bin/env python3
"""
Extract properly filtered user feedback quotes from disagreement cases
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import random

def extract_filtered_quotes():
    """Extract F+High and S+Med cases with proper filtering"""

    base_path = "/home/xuhuizhou/OpenHands/evaluation/evaluation_outputs/outputs/cmu-lti__stateful-test"

    quotes_data = {
        'f_high_quotes': [],
        's_med_quotes': []
    }

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
                            instance_id = result.get('instance_id', 'unknown')
                            metrics = result.get('metrics', {})
                            task_success = result.get('task_resolved', False)
                            overall_sat = metrics.get('overall_satisfaction', 0)

                            # Get the actual user feedback text
                            explanation = result.get('explanation', '')
                            detailed_feedback = result.get('detailed_feedback', {})

                            case_data = {
                                'instance_id': instance_id,
                                'config': config_name,
                                'task_success': task_success,
                                'overall_satisfaction': overall_sat,
                                'metrics': metrics,
                                'explanation': explanation,
                                'detailed_feedback': detailed_feedback
                            }

                            # Apply PROPER filtering criteria
                            # Failed + High satisfaction (>3.5)
                            if task_success == False and overall_sat > 3.5:
                                quotes_data['f_high_quotes'].append(case_data)

                            # Success + Medium satisfaction (2.0-3.5)
                            elif task_success == True and 2.0 <= overall_sat <= 3.5:
                                quotes_data['s_med_quotes'].append(case_data)

                        break

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return quotes_data

def print_filtered_quotes(quotes_data):
    """Print only properly filtered cases"""

    print("PROPERLY FILTERED USER FEEDBACK QUOTES")
    print("=" * 120)

    # F+High Cases (Failed + High Satisfaction >3.5)
    print(f"\n1. FAILED + HIGH SATISFACTION CASES ({len(quotes_data['f_high_quotes'])} total)")
    print("-" * 120)

    count = 0
    import random
    selected_cases = random.sample(quotes_data['f_high_quotes'], min(50, len(quotes_data['f_high_quotes'])))
    for case in selected_cases:
        count += 1

        print(f"\nInstance: {case['instance_id']}")
        print(f"Config: {case['config']}")
        print(f"Task Success: {case['task_success']} | Overall Satisfaction: {case['overall_satisfaction']:.1f}/5")
        print(f"Scores - Communication: {case['metrics'].get('communication_quality', 'N/A'):.1f}, "
              f"Problem Solving: {case['metrics'].get('problem_solving_approach', 'N/A'):.1f}, "
              f"Efficiency: {case['metrics'].get('efficiency', 'N/A'):.1f}, "
              f"User Preference: {case['metrics'].get('user_preference_alignment', 'N/A'):.1f}")

        explanation = case['explanation']
        detailed = case['detailed_feedback']

        if explanation:
            print(f"User Explanation: \"{explanation[:200]}...\"" if len(explanation) > 200 else f"User Explanation: \"{explanation}\"")

        if detailed:
            strengths = detailed.get('strengths', '')
            if strengths:
                print(f"Strengths: \"{strengths[:300]}...\"" if len(strengths) > 300 else f"Strengths: \"{strengths}\"")

        print("-" * 80)

    # S+Med Cases (Success + Medium Satisfaction 2.0-3.5)
    print(f"\n\n2. SUCCESS + MEDIUM SATISFACTION CASES ({len(quotes_data['s_med_quotes'])} total)")
    print("-" * 120)

    count = 0
    selected_cases = random.sample(quotes_data['s_med_quotes'], min(50, len(quotes_data['s_med_quotes'])))
    for case in selected_cases:
        if count >= 3:  # Limit to first 3 examples
            break
        count += 1

        print(f"\nInstance: {case['instance_id']}")
        print(f"Config: {case['config']}")
        print(f"Task Success: {case['task_success']} | Overall Satisfaction: {case['overall_satisfaction']:.1f}/5")
        print(f"Scores - Communication: {case['metrics'].get('communication_quality', 'N/A'):.1f}, "
              f"Problem Solving: {case['metrics'].get('problem_solving_approach', 'N/A'):.1f}, "
              f"Efficiency: {case['metrics'].get('efficiency', 'N/A'):.1f}, "
              f"User Preference: {case['metrics'].get('user_preference_alignment', 'N/A'):.1f}")

        explanation = case['explanation']
        detailed = case['detailed_feedback']

        if explanation:
            print(f"User Explanation: \"{explanation[:200]}...\"" if len(explanation) > 200 else f"User Explanation: \"{explanation}\"")

        if detailed:
            weaknesses = detailed.get('weaknesses', '')
            if weaknesses:
                print(f"Weaknesses: \"{weaknesses[:300]}...\"" if len(weaknesses) > 300 else f"Weaknesses: \"{weaknesses}\"")

        print("-" * 80)

def main():
    """Main execution"""
    print("EXTRACTING PROPERLY FILTERED DISAGREEMENT CASES")
    print("=" * 120)

    # Set random seed for reproducible examples
    random.seed(42)

    # Extract properly filtered quotes
    quotes_data = extract_filtered_quotes()

    print(f"Found {len(quotes_data['f_high_quotes'])} F+High cases (Failed + Satisfaction >3.5)")
    print(f"Found {len(quotes_data['s_med_quotes'])} S+Med cases (Success + Satisfaction 2.0-3.5)")

    # Print filtered quotes
    print_filtered_quotes(quotes_data)

    print("\n" + "=" * 120)
    print("FILTERING COMPLETE - Only showing true disagreement cases")

if __name__ == '__main__':
    main()

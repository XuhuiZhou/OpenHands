#!/usr/bin/env python3
"""
Extract confidence intervals from all user_satisfaction.json files and create a comprehensive table
"""
import json
import os
from pathlib import Path

def extract_ci_data():
    """Extract CI data from all user satisfaction files"""

    base_path = "/home/xuhuizhou/OpenHands/evaluation/evaluation_outputs/outputs/cmu-lti__stateful-test"

    results = []

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

            # Load JSON data
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Get the first (and usually only) result set
                for file_key, file_data in data.items():
                    if file_data and 'confidence_intervals' in file_data:
                        ci_data = file_data['confidence_intervals']
                        overall_sat = ci_data.get('overall_satisfaction', {})

                        if 'ci_lower' in overall_sat and 'ci_upper' in overall_sat:
                            results.append({
                                'agent': agent,
                                'model': model + rag_suffix,
                                'ci_lower': overall_sat['ci_lower'],
                                'ci_upper': overall_sat['ci_upper'],
                                'mean': overall_sat['mean'],
                                'n_samples': overall_sat['n_samples'],
                                'margin_error': overall_sat['margin_of_error']
                            })
                        break

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return results

def print_table(results):
    """Print formatted table"""

    # Sort results by agent then model
    results.sort(key=lambda x: (x['agent'], x['model']))

    print("Confidence Interval Analysis - Overall Satisfaction (95% CI)")
    print("=" * 80)
    print(f"{'Agent':<20} | {'Model':<25} | {'Mean':<6} | {'CI Lower':<8} | {'CI Upper':<8} | {'±Error':<7} | {'n':<5}")
    print("-" * 80)

    for r in results:
        print(f"{r['agent']:<20} | {r['model']:<25} | {r['mean']:.2f} | {r['ci_lower']:.3f}   | {r['ci_upper']:.3f}   | {r['margin_error']:.3f} | {r['n_samples']:<5}")

    print("\nSummary:")
    print(f"Total configurations: {len(results)}")

    # Group by agent
    tom_results = [r for r in results if 'Tom' in r['agent']]
    code_results = [r for r in results if 'Tom' not in r['agent']]

    if tom_results:
        tom_means = [r['mean'] for r in tom_results]
        print(f"TomCodeActAgent - Mean satisfaction range: {min(tom_means):.2f} - {max(tom_means):.2f}")

    if code_results:
        code_means = [r['mean'] for r in code_results]
        print(f"CodeActAgent - Mean satisfaction range: {min(code_means):.2f} - {max(code_means):.2f}")

if __name__ == '__main__':
    results = extract_ci_data()
    print_table(results)
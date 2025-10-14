#!/usr/bin/env python3
"""
Compact visualization of contingency tables across all configurations
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import statistics

def extract_contingency_data():
    """Extract contingency and summary data from all user satisfaction files"""

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
                    if file_data and 'contingency_analysis' in file_data:
                        contingency = file_data['contingency_analysis']
                        correlation = file_data.get('correlation_analysis', {})
                        ci_data = file_data.get('confidence_intervals', {})
                        averages = file_data.get('overall_averages', {})

                        config_data = {
                            'agent': agent,
                            'model': model + rag_suffix,
                            'config_name': f"{agent}-{model}{rag_suffix}",
                            'contingency': contingency,
                            'correlation': correlation,
                            'confidence_intervals': ci_data,
                            'averages': averages,
                            'file_path': file_path
                        }
                        results.append(config_data)
                        break

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return results

def create_summary_table(results: List[Dict[str, Any]]):
    """Create a summary metrics table"""

    print("SUMMARY METRICS TABLE")
    print("=" * 120)
    print(f"{'Configuration':<35} | {'Success':<7} | {'Mean Sat':<8} | {'High Sat':<8} | {'Disagree':<8} | {'Correlation':<11} | {'n':<5}")
    print("-" * 120)

    summary_data = []

    for result in sorted(results, key=lambda x: (x['agent'], x['model'])):
        config_name = result['config_name']

        # Extract key metrics
        correlation = result['correlation']
        contingency = result['contingency']
        averages = result['averages']

        if 'contingency_table' in contingency:
            # Calculate success rate
            totals = contingency['totals']
            success_rate = totals['successful_tasks'] / totals['total_samples'] * 100

            # Get mean satisfaction
            mean_satisfaction = averages.get('overall_satisfaction', 0)

            # Calculate high satisfaction rate (>3.5)
            table = contingency['contingency_table']
            high_sat_failed = table['Failed Tasks (0)']['High (>3.5)']
            high_sat_success = table['Successful Tasks (1)']['High (>3.5)']
            high_sat_rate = (high_sat_failed + high_sat_success) / totals['total_samples'] * 100

            # Get disagreement rate
            disagreement = contingency['disagreement_analysis']
            disagree_rate = disagreement['total_disagreement']['percentage']

            # Get correlation
            corr_value = correlation.get('correlation', 0)

            # Sample size
            n_samples = totals['total_samples']

            summary_data.append({
                'config': config_name,
                'success_rate': success_rate,
                'mean_satisfaction': mean_satisfaction,
                'high_sat_rate': high_sat_rate,
                'disagree_rate': disagree_rate,
                'correlation': corr_value,
                'n_samples': n_samples
            })

            print(f"{config_name:<35} | {success_rate:>6.1f}% | {mean_satisfaction:>7.2f} | {high_sat_rate:>7.1f}% | {disagree_rate:>7.1f}% | {corr_value:>10.3f} | {n_samples:>5}")

    return summary_data

def create_disagreement_analysis(results: List[Dict[str, Any]]):
    """Focus on disagreement cases"""

    print("\n\nDISAGREEMENT ANALYSIS")
    print("=" * 100)
    print(f"{'Configuration':<35} | {'High Sat + Failed':<15} | {'Low Sat + Success':<16} | {'Total Disagree':<13}")
    print("-" * 100)

    for result in sorted(results, key=lambda x: (x['agent'], x['model'])):
        config_name = result['config_name']
        contingency = result['contingency']

        if 'disagreement_analysis' in contingency:
            disagreement = contingency['disagreement_analysis']

            high_sat_failed = disagreement['high_satisfaction_but_failed']
            low_sat_success = disagreement['low_satisfaction_but_successful']
            total_disagree = disagreement['total_disagreement']

            print(f"{config_name:<35} | {high_sat_failed['count']:>4} ({high_sat_failed['percentage']:>4.1f}%) | {low_sat_success['count']:>4} ({low_sat_success['percentage']:>4.1f}%) | {total_disagree['count']:>4} ({total_disagree['percentage']:>4.1f}%)")

def create_contingency_heatmap(results: List[Dict[str, Any]]):
    """Create a compact heatmap-style contingency view with within-group percentages"""

    print("\n\nCONTINGENCY HEATMAP (Within-Group Percentages)")
    print("=" * 140)
    print(f"{'Configuration':<35} | {'F+Low':<6} | {'F+Med':<6} | {'F+High':<7} | {'S+Low':<6} | {'S+Med':<6} | {'S+High':<7}")
    print("-" * 140)

    for result in sorted(results, key=lambda x: (x['agent'], x['model'])):
        config_name = result['config_name']
        contingency = result['contingency']

        if 'contingency_table' in contingency:
            table = contingency['contingency_table']
            percentages = contingency['percentages']  # This already has within-group percentages
            totals = contingency['totals']

            # Use the pre-calculated within-group percentages from the JSON
            f_low_pct = percentages['Failed Tasks (0)']['Low (≤2.0)']
            f_med_pct = percentages['Failed Tasks (0)']['Medium (2.0-3.5)']
            f_high_pct = percentages['Failed Tasks (0)']['High (>3.5)']
            s_low_pct = percentages['Successful Tasks (1)']['Low (≤2.0)']
            s_med_pct = percentages['Successful Tasks (1)']['Medium (2.0-3.5)']
            s_high_pct = percentages['Successful Tasks (1)']['High (>3.5)']

            print(f"{config_name:<35} | {f_low_pct:>5.1f}% | {f_med_pct:>5.1f}% | {f_high_pct:>6.1f}% | {s_low_pct:>5.1f}% | {s_med_pct:>5.1f}% | {s_high_pct:>6.1f}%")

def create_agent_comparison(results: List[Dict[str, Any]]):
    """Compare TomCodeActAgent vs CodeActAgent"""

    print("\n\nAGENT COMPARISON")
    print("=" * 80)

    tom_results = [r for r in results if 'Tom' in r['agent']]
    code_results = [r for r in results if 'Tom' not in r['agent']]

    print(f"TomCodeActAgent Configurations: {len(tom_results)}")
    print(f"CodeActAgent Configurations: {len(code_results)}")

    # Calculate averages
    if tom_results:
        tom_success_rates = []
        tom_satisfactions = []
        tom_disagreements = []

        for r in tom_results:
            if 'contingency_table' in r['contingency']:
                totals = r['contingency']['totals']
                tom_success_rates.append(totals['successful_tasks'] / totals['total_samples'] * 100)
                tom_satisfactions.append(r['averages'].get('overall_satisfaction', 0))
                tom_disagreements.append(r['contingency']['disagreement_analysis']['total_disagreement']['percentage'])

        print(f"\nTomCodeActAgent Averages:")
        print(f"  Success Rate: {statistics.mean(tom_success_rates):.1f}% (range: {min(tom_success_rates):.1f}%-{max(tom_success_rates):.1f}%)")
        print(f"  Mean Satisfaction: {statistics.mean(tom_satisfactions):.2f}/5 (range: {min(tom_satisfactions):.2f}-{max(tom_satisfactions):.2f})")
        print(f"  Disagreement Rate: {statistics.mean(tom_disagreements):.1f}% (range: {min(tom_disagreements):.1f}%-{max(tom_disagreements):.1f}%)")

    if code_results:
        code_success_rates = []
        code_satisfactions = []
        code_disagreements = []

        for r in code_results:
            if 'contingency_table' in r['contingency']:
                totals = r['contingency']['totals']
                code_success_rates.append(totals['successful_tasks'] / totals['total_samples'] * 100)
                code_satisfactions.append(r['averages'].get('overall_satisfaction', 0))
                code_disagreements.append(r['contingency']['disagreement_analysis']['total_disagreement']['percentage'])

        print(f"\nCodeActAgent Averages:")
        print(f"  Success Rate: {statistics.mean(code_success_rates):.1f}% (range: {min(code_success_rates):.1f}%-{max(code_success_rates):.1f}%)")
        print(f"  Mean Satisfaction: {statistics.mean(code_satisfactions):.2f}/5 (range: {min(code_satisfactions):.2f}-{max(code_satisfactions):.2f})")
        print(f"  Disagreement Rate: {statistics.mean(code_disagreements):.1f}% (range: {min(code_disagreements):.1f}%-{max(code_disagreements):.1f}%)")

def main():
    """Main execution"""
    print("CONTINGENCY TABLE ANALYSIS - ALL CONFIGURATIONS")
    print("=" * 150)

    # Extract all data
    results = extract_contingency_data()

    if not results:
        print("No contingency data found!")
        return

    print(f"Found {len(results)} configurations with contingency data\n")

    # Generate all visualizations
    summary_data = create_summary_table(results)
    create_disagreement_analysis(results)
    create_contingency_heatmap(results)
    create_agent_comparison(results)

    print("\n" + "=" * 150)
    print("ANALYSIS COMPLETE")

if __name__ == '__main__':
    main()
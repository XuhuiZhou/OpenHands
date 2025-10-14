#!/usr/bin/env python3
"""
Display contingency analysis from existing user_satisfaction.json results
"""
import json
import sys
from pathlib import Path

def display_contingency_analysis(results_file: str):
    """Load and display contingency analysis from existing results"""

    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        return

    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)

    # Find the first result set that has detailed results
    overall_aggregated = None
    for file_path, file_results in all_results.items():
        if file_results and 'detailed_results' in file_results:
            # Aggregate all results
            all_instance_results = []
            for result_set in all_results.values():
                if result_set and 'detailed_results' in result_set:
                    all_instance_results.extend(result_set['detailed_results'])

            if all_instance_results:
                # Import the aggregation function
                import sys
                sys.path.append('/home/xuhuizhou/OpenHands')
                from evaluation.benchmarks.swe_bench.eval_user_satisfaction import aggregate_results
                overall_aggregated = aggregate_results(all_instance_results)
                break

    if not overall_aggregated:
        print("Error: No detailed results found in the file")
        return

    # Display results
    summary = overall_aggregated['summary']
    averages = overall_aggregated.get('overall_averages', {})
    correlation_analysis = overall_aggregated.get('correlation_analysis', {})
    contingency_analysis = overall_aggregated.get('contingency_analysis', {})
    confidence_intervals = overall_aggregated.get('confidence_intervals', {})

    print(f"EVALUATION SUMMARY ({summary['total_instances']} instances):")
    print(f"  Instances with evaluation results: {summary['instances_with_eval_results']}")

    # Display satisfaction metrics with confidence intervals
    if averages:
        print(f"\nUser Satisfaction Metrics (1-5 scale):")
        print(f"  Overall Satisfaction: {averages['overall_satisfaction']:.2f}/5")
        print(f"  Communication Quality: {averages['communication_quality']:.2f}/5")
        print(f"  Problem Solving Approach: {averages['problem_solving_approach']:.2f}/5")
        print(f"  Efficiency: {averages['efficiency']:.2f}/5")
        print(f"  User Preference Alignment: {averages['user_preference_alignment']:.2f}/5")

    # Display confidence intervals
    if confidence_intervals:
        print(f"\n95% Confidence Intervals:")
        for metric_name, metric_data in confidence_intervals.items():
            if 'error' not in metric_data:
                ci_lower = metric_data['ci_lower']
                ci_upper = metric_data['ci_upper']
                margin_error = metric_data['margin_of_error']
                n_samples = metric_data['n_samples']

                # Format metric name for display
                display_name = metric_name.replace('_', ' ').title()
                print(f"  {display_name}: [{ci_lower:.2f}, {ci_upper:.2f}] (±{margin_error:.2f}, n={n_samples})")
            else:
                display_name = metric_name.replace('_', ' ').title()
                print(f"  {display_name}: {metric_data['error']}")

    # Display correlation analysis
    if correlation_analysis and correlation_analysis.get('correlation') is not None:
        print(f"\nTask Success vs User Satisfaction Correlation:")
        print(f"  Point-biserial correlation: {correlation_analysis['correlation']:.3f}")
        print(f"  P-value: {correlation_analysis['p_value']:.4f}")
        print(f"  Sample size: {correlation_analysis['n_samples']} instances")
        print(f"  Task success rate: {correlation_analysis['task_success_rate']:.1%}")
        print(f"  Mean satisfaction: {correlation_analysis['mean_satisfaction']:.2f}/5")

        # Interpret correlation strength
        corr = abs(correlation_analysis['correlation'])
        if corr < 0.1:
            strength = "negligible"
        elif corr < 0.3:
            strength = "weak"
        elif corr < 0.5:
            strength = "moderate"
        elif corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"

        significance = "significant" if correlation_analysis['p_value'] < 0.05 else "not significant"
        print(f"  Interpretation: {strength} correlation ({significance})")

    # Display contingency analysis
    if contingency_analysis and 'contingency_table' in contingency_analysis:
        print(f"\nContingency Analysis - Task Success vs Satisfaction Levels:")

        # Display contingency table
        table = contingency_analysis['contingency_table']
        totals = contingency_analysis['totals']
        percentages = contingency_analysis['percentages']

        print(f"  {'':20} | {'Low (≤2.0)':>12} | {'Medium (2.0-3.5)':>16} | {'High (>3.5)':>12} | {'Total':>8}")
        print(f"  {'-'*20}|{'-'*14}|{'-'*18}|{'-'*14}|{'-'*10}")

        for task_type in ['Failed Tasks (0)', 'Successful Tasks (1)']:
            counts = table[task_type]
            percs = percentages[task_type]
            total = totals['failed_tasks'] if 'Failed' in task_type else totals['successful_tasks']

            print(f"  {task_type:20}| {counts['Low (≤2.0)']:>4} ({percs['Low (≤2.0)']:>4.1f}%) "
                  f"| {counts['Medium (2.0-3.5)']:>4} ({percs['Medium (2.0-3.5)']:>4.1f}%) "
                  f"| {counts['High (>3.5)']:>4} ({percs['High (>3.5)']:>4.1f}%) | {total:>8}")

        # Display disagreement analysis
        disagreement = contingency_analysis['disagreement_analysis']
        print(f"\nDisagreement Cases (showing task success ≠ satisfaction level):")
        print(f"  High satisfaction but task failed: {disagreement['high_satisfaction_but_failed']['count']} "
              f"({disagreement['high_satisfaction_but_failed']['percentage']:.1f}%)")
        print(f"  Low satisfaction but task succeeded: {disagreement['low_satisfaction_but_successful']['count']} "
              f"({disagreement['low_satisfaction_but_successful']['percentage']:.1f}%)")
        print(f"  Total disagreement cases: {disagreement['total_disagreement']['count']} "
              f"({disagreement['total_disagreement']['percentage']:.1f}%)")

        # Calculate R-squared and variance explained
        r_squared = correlation_analysis['correlation'] ** 2 if correlation_analysis.get('correlation') else 0
        print(f"\nVariance Analysis:")
        print(f"  R² (variance explained by task success): {r_squared:.1%}")
        print(f"  Unexplained variance (other factors): {1-r_squared:.1%}")

    elif contingency_analysis.get('error'):
        print(f"\nContingency Analysis:")
        print(f"  Error: {contingency_analysis['error']}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python display_contingency.py <path_to_user_satisfaction.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    display_contingency_analysis(results_file)
#!/usr/bin/env python3
"""
User Satisfaction Evaluation Script for SWE-Bench

This script evaluates agent performance from a user perspective by analyzing trajectories
and generating satisfaction ratings. It focuses on evaluating stateful mode outputs
where user profiles and preferences are considered.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import scipy.stats
import numpy as np
from litellm import acompletion as litellm_acompletion

from openhands.core.config import get_llm_config_arg
from openhands.events.serialization.event import event_from_dict


@dataclass
class UserSatisfactionMetrics:
    """Metrics for user satisfaction evaluation"""
    overall_satisfaction: float  # 1-5 scale
    communication_quality: float  # 1-5 scale
    problem_solving_approach: float  # 1-5 scale
    efficiency: float  # 1-5 scale
    user_preference_alignment: float  # 1-5 scale (for stateful mode)
    explanation: str
    detailed_feedback: Dict[str, str]


class FakeUserEvaluator:
    """Simulates a user evaluating the agent's performance"""

    def __init__(self, llm_config_name: str = 'llm.fake_user'):
        self.llm_config = get_llm_config_arg(llm_config_name)

    def _format_trajectory_events(self, history: List[Dict[str, Any]]) -> str:
        """Format trajectory events into a readable conversation flow"""
        formatted_events = []

        for i, event in enumerate(history):
            source = event.get('source', 'unknown')
            action = event.get('action', 'unknown')
            args = event.get('args', {})

            if source == 'user' and action == 'message':
                content = args.get('content', '')
                formatted_events.append(f"USER: {content}")
            elif source == 'agent' and action == 'message':
                content = args.get('content', '')
                # Truncate very long messages but preserve important info
                if len(content) > 500:
                    content = content[:500] + "...[truncated]"
                formatted_events.append(f"AGENT: {content}")
            elif source == 'agent' and action in ['run', 'str_replace', 'create', 'edit']:
                # Show key technical actions
                if action == 'run':
                    command = args.get('command', '')
                    formatted_events.append(f"AGENT_ACTION: Run command: {command}")
                elif action == 'str_replace':
                    file_path = args.get('file_path', '')
                    formatted_events.append(f"AGENT_ACTION: Edit file: {file_path}")
                elif action in ['create', 'edit']:
                    file_path = args.get('file_path', '')
                    formatted_events.append(f"AGENT_ACTION: {action.title()} file: {file_path}")

        return "\n".join(formatted_events)

    def _load_evaluation_results(self, instance_id: str, base_dir: str) -> Optional[Dict[str, Any]]:
        """Load evaluation results from report.json if available"""
        report_path = Path(base_dir) / "eval_outputs" / instance_id / "report.json"
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load report.json for {instance_id}: {e}")
        return None

    async def evaluate_trajectory(
        self,
        instance: Dict[str, Any],
        history: List[Dict[str, Any]],
        test_result: Dict[str, Any],
        user_profile: str = None,
        eval_results: Dict[str, Any] = None
    ) -> UserSatisfactionMetrics:
        """
        Evaluate a single trajectory from user perspective

        Args:
            instance: The problem instance data
            history: Complete interaction history
            test_result: Results from the agent's attempt
            user_profile: User roleplay prompt (for stateful mode)
            eval_results: Evaluation results from report.json
        """
        # Build evaluation prompt with full trajectory
        evaluation_prompt = self._build_evaluation_prompt(
            instance, history, test_result, user_profile, eval_results
        )
        # Get LLM evaluation
        response = await litellm_acompletion(
            model=self.llm_config.model,
            messages=[{'role': 'user', 'content': evaluation_prompt}],
            api_key=self.llm_config.api_key.get_secret_value(),
            temperature=1.0,
        )

        # Parse response into metrics
        return self._parse_evaluation_response(response.choices[0].message.content)

    def _build_evaluation_prompt(
        self,
        instance: Dict[str, Any],
        history: List[Dict[str, Any]],
        test_result: Dict[str, Any],
        user_profile: str = None,
        eval_results: Dict[str, Any] = None
    ) -> str:
        """Build the comprehensive evaluation prompt for the LLM"""

        instance_id = instance.get('instance_id', 'Unknown')
        problem_statement = instance.get('problem_statement', 'No problem statement provided')
        original_issue = instance.get('original_issue', 'No original issue provided')

        # Format the full trajectory
        trajectory_text = self._format_trajectory_events(history)

        # Extract git patch
        git_patch = test_result.get('git_patch', '')

        # Format evaluation results if available
        eval_summary = self._format_evaluation_results(eval_results, instance_id) if eval_results else "No technical evaluation results available."

        # User profile context for stateful mode
        profile_section = instance.get('user_roleplay_prompt', 'No user profile provided')

        prompt = f"""You are evaluating an AI coding agent's performance from a USER PERSPECTIVE.

=== ORIGINAL PROBLEM ===
Instance ID: {instance_id}
Problem Statement: {problem_statement}

=== USER PROFILE ===
{profile_section}

Full problem/issue description on user's mind:
{original_issue}

=== TECHNICAL EVALUATION RESULTS ===
{eval_summary}

=== COMPLETE INTERACTION TRAJECTORY ===
{trajectory_text}

=== FINAL OUTPUT ===
Git Patch ({len(git_patch)} characters):
{git_patch[:1000] + "\\n...[truncated]" if len(git_patch) > 1000 else git_patch if git_patch.strip() else "No changes were made to the codebase."}

=== EVALUATION CRITERIA ===
Evaluate the agent from a user's perspective on these dimensions (1-5 scale):

1. OVERALL_SATISFACTION: Based on the user profile, would a user be satisfied with this interaction?
2. COMMUNICATION_QUALITY: How well did the agent communicate? Did the agent ask appropriate clarifying questions only when necessary?
3. PROBLEM_SOLVING_APPROACH: Was the approach systematic and effective?
4. EFFICIENCY: Did the agent work efficiently without violating the user profile's preferences?
5. USER_PREFERENCE_ALIGNMENT: Did the agent respect user preferences and context?

Respond in this EXACT JSON format:
{{
    "detailed_feedback": {{
        "strengths": "What the agent did well",
        "weaknesses": "What could be improved",
        "user_experience": "How this felt from user perspective",
        "question_asking_behavior": "Assessment of when and how agent asked questions"
    }}
    "overall_satisfaction": 4.2,
    "communication_quality": 4.0,
    "problem_solving_approach": 4.5,
    "efficiency": 3.8,
    "user_preference_alignment": 4.1,
    "explanation": "Brief explanation of the overall rating",
}}
"""
        return prompt

    def _format_evaluation_results(self, eval_results: Dict[str, Any], instance_id: str) -> str:
        """Format technical evaluation results from report.json"""
        if not eval_results or instance_id not in eval_results:
            return "No technical evaluation results found."

        result = eval_results[instance_id]

        sections = []
        sections.append(f"Patch Status: {'Applied successfully' if result.get('patch_successfully_applied') else 'Failed to apply'}")
        sections.append(f"Problem Resolved: {'Yes' if result.get('resolved') else 'No'}")

        tests_status = result.get('tests_status', {})
        if tests_status:
            fail_to_pass = tests_status.get('FAIL_TO_PASS', {})
            sections.append(f"Tests Fixed: {len(fail_to_pass.get('success', []))} tests now pass")
            if fail_to_pass.get('failure'):
                sections.append(f"Tests Still Failing: {len(fail_to_pass.get('failure', []))} tests still fail")

        return "\n".join(sections)

    def _parse_evaluation_response(self, response: str) -> UserSatisfactionMetrics:
        """Parse the LLM response into structured metrics"""
        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]

            data = json.loads(response)

            return UserSatisfactionMetrics(
                overall_satisfaction=float(data.get('overall_satisfaction', 0)),
                communication_quality=float(data.get('communication_quality', 0)),
                problem_solving_approach=float(data.get('problem_solving_approach', 0)),
                efficiency=float(data.get('efficiency', 0)),
                user_preference_alignment=float(data.get('user_preference_alignment', 0)),
                explanation=data.get('explanation', ''),
                detailed_feedback=data.get('detailed_feedback', {})
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not parse evaluation response: {e}")
            print(f"Response: {response}")
            return UserSatisfactionMetrics(
                overall_satisfaction=0,
                communication_quality=0,
                problem_solving_approach=0,
                efficiency=0,
                user_preference_alignment=0,
                explanation=f"Failed to parse response: {str(e)}",
                detailed_feedback={}
            )


async def evaluate_instance_batch(
    evaluator: FakeUserEvaluator,
    instances: List[Tuple[int, Dict[str, Any]]],
    base_dir: str
) -> List[Dict[str, Any]]:
    """Evaluate a batch of instances concurrently"""
    async def evaluate_single(line_num: int, instance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            instance_id = instance_data.get('instance_id')
            if not instance_id:
                print(f"Warning: No instance_id found in line {line_num + 1}")
                return None

            # Load evaluation results from report.json
            eval_results = evaluator._load_evaluation_results(instance_id, str(base_dir))

            # Extract task success if available
            task_resolved = None
            if eval_results and instance_id in eval_results:
                task_resolved = eval_results[instance_id].get('resolved', False)

            # Extract components
            history = instance_data.get('history', [])
            instance_dict = instance_data.get('instance', {})
            user_profile = instance_dict.get('user_roleplay_prompt')

            # Evaluate this instance
            metrics = await evaluator.evaluate_trajectory(
                instance=instance_dict,
                history=history,
                test_result=instance_data.get('test_result', {}),
                user_profile=user_profile,
                eval_results=eval_results
            )

            result = {
                'instance_id': instance_id,
                'metrics': {
                    'overall_satisfaction': metrics.overall_satisfaction,
                    'communication_quality': metrics.communication_quality,
                    'problem_solving_approach': metrics.problem_solving_approach,
                    'efficiency': metrics.efficiency,
                    'user_preference_alignment': metrics.user_preference_alignment,
                },
                'explanation': metrics.explanation,
                'detailed_feedback': metrics.detailed_feedback,
                'has_user_profile': user_profile is not None,
                'trajectory_length': len(history),
                'has_eval_results': eval_results is not None,
                'task_resolved': task_resolved
            }

            print(f"Evaluated {instance_id}: Overall satisfaction = {metrics.overall_satisfaction:.2f}")
            return result

        except Exception as e:
            print(f"Error processing line {line_num + 1}: {e}")
            return None

    # Process all instances in the batch concurrently
    tasks = [evaluate_single(line_num, instance_data) for line_num, instance_data in instances]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results and exceptions
    return [r for r in results if r is not None and not isinstance(r, Exception)]

async def evaluate_output_file(
    output_file: str,
    evaluator: FakeUserEvaluator,
    max_instances: int = None,
    batch_size: int = 64
) -> Dict[str, Any]:
    """Evaluate all instances in an output file with async batching"""
    all_results = []
    output_path = Path(output_file)
    base_dir = output_path.parent  # Directory containing output.jsonl

    # Read all instances
    instances = []
    with open(output_file, 'r') as f:
        for line_num, line in enumerate(f):
            if max_instances and line_num >= max_instances:
                break
            try:
                instance_data = json.loads(line)
                instances.append((line_num, instance_data))
            except Exception as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue

    # Process instances in batches
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(instances) + batch_size - 1)//batch_size} ({len(batch)} instances)")

        batch_results = await evaluate_instance_batch(evaluator, batch, str(base_dir))
        all_results.extend(batch_results)

    return aggregate_results(all_results)


def calculate_confidence_intervals(results: List[Dict[str, Any]], confidence_level: float = 0.95) -> Dict[str, Any]:
    """Calculate confidence intervals for satisfaction metrics"""
    metrics = ['overall_satisfaction', 'communication_quality', 'problem_solving_approach', 'efficiency', 'user_preference_alignment']
    confidence_intervals = {}

    alpha = 1 - confidence_level

    for metric in metrics:
        values = [r['metrics'][metric] for r in results if r['metrics'][metric] > 0]

        if len(values) < 2:
            confidence_intervals[metric] = {
                'mean': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'margin_of_error': 0,
                'n_samples': len(values),
                'error': 'Insufficient data'
            }
            continue

        # Calculate basic statistics
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation
        n = len(values)

        # Calculate standard error
        se = std / np.sqrt(n)

        # Calculate t-critical value for given confidence level
        t_critical = scipy.stats.t.ppf(1 - alpha/2, df=n-1)

        # Calculate margin of error
        margin_of_error = t_critical * se

        # Calculate confidence interval
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error

        confidence_intervals[metric] = {
            'mean': mean,
            'std': std,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin_of_error': margin_of_error,
            'n_samples': n,
            'confidence_level': confidence_level
        }

    return confidence_intervals


def calculate_contingency_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate contingency table between task success and satisfaction levels"""
    # Extract paired data
    task_success = []
    satisfaction_scores = []

    for result in results:
        if not result.get('has_eval_results', False) or 'task_resolved' not in result:
            continue

        satisfaction = result['metrics']['overall_satisfaction']
        if satisfaction <= 0:
            continue

        task_success.append(int(result['task_resolved']))
        satisfaction_scores.append(satisfaction)

    if len(task_success) < 2:
        return {'error': 'Insufficient data for contingency analysis'}

    # Categorize satisfaction scores
    satisfaction_categories = []
    for score in satisfaction_scores:
        if score <= 2.0:
            satisfaction_categories.append('Low (≤2.0)')
        elif score <= 3.5:
            satisfaction_categories.append('Medium (2.0-3.5)')
        else:
            satisfaction_categories.append('High (>3.5)')

    # Build contingency table
    contingency = {
        'Failed Tasks (0)': {'Low (≤2.0)': 0, 'Medium (2.0-3.5)': 0, 'High (>3.5)': 0},
        'Successful Tasks (1)': {'Low (≤2.0)': 0, 'Medium (2.0-3.5)': 0, 'High (>3.5)': 0}
    }

    for success, sat_cat in zip(task_success, satisfaction_categories):
        success_label = 'Successful Tasks (1)' if success else 'Failed Tasks (0)'
        contingency[success_label][sat_cat] += 1

    # Calculate percentages and totals
    total_failed = sum(contingency['Failed Tasks (0)'].values())
    total_successful = sum(contingency['Successful Tasks (1)'].values())
    total_samples = total_failed + total_successful

    # Add percentage calculations
    result = {
        'contingency_table': contingency,
        'totals': {
            'failed_tasks': total_failed,
            'successful_tasks': total_successful,
            'total_samples': total_samples
        },
        'percentages': {}
    }

    # Calculate row percentages (within each success category)
    for success_type in contingency:
        total = total_failed if 'Failed' in success_type else total_successful
        result['percentages'][success_type] = {}
        for sat_cat in contingency[success_type]:
            count = contingency[success_type][sat_cat]
            percentage = (count / total * 100) if total > 0 else 0
            result['percentages'][success_type][sat_cat] = percentage

    # Calculate disagreement cases
    # High satisfaction but failed task
    high_sat_failed = contingency['Failed Tasks (0)']['High (>3.5)']
    # Low satisfaction but successful task
    low_sat_success = contingency['Successful Tasks (1)']['Low (≤2.0)']

    result['disagreement_analysis'] = {
        'high_satisfaction_but_failed': {
            'count': high_sat_failed,
            'percentage': (high_sat_failed / total_samples * 100) if total_samples > 0 else 0
        },
        'low_satisfaction_but_successful': {
            'count': low_sat_success,
            'percentage': (low_sat_success / total_samples * 100) if total_samples > 0 else 0
        },
        'total_disagreement': {
            'count': high_sat_failed + low_sat_success,
            'percentage': ((high_sat_failed + low_sat_success) / total_samples * 100) if total_samples > 0 else 0
        }
    }

    return result


def calculate_task_success_correlation(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate point-biserial correlation between task success and user satisfaction"""
    # Extract paired data where both task success and satisfaction are available
    task_success = []
    satisfaction_scores = []

    for result in results:
        if not result.get('has_eval_results', False):
            continue

        instance_id = result['instance_id']
        satisfaction = result['metrics']['overall_satisfaction']

        # Skip if satisfaction score is 0 (indicates parsing failure or no score)
        if satisfaction <= 0:
            continue

        # For this implementation, we need to load the actual eval results
        # Since we don't have direct access here, we'll look for it in the result
        # The eval results should be added to the result dict during evaluation
        if 'task_resolved' in result:
            task_success.append(int(result['task_resolved']))
            satisfaction_scores.append(satisfaction)

    if len(task_success) < 2:
        return {
            'correlation': None,
            'p_value': None,
            'n_samples': len(task_success),
            'error': 'Insufficient data for correlation analysis'
        }

    try:
        correlation, p_value = scipy.stats.pointbiserialr(task_success, satisfaction_scores)
        return {
            'correlation': correlation,
            'p_value': p_value,
            'n_samples': len(task_success),
            'task_success_rate': sum(task_success) / len(task_success),
            'mean_satisfaction': sum(satisfaction_scores) / len(satisfaction_scores)
        }
    except Exception as e:
        return {
            'correlation': None,
            'p_value': None,
            'n_samples': len(task_success),
            'error': f'Error calculating correlation: {str(e)}'
        }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluation results with improved categorization"""
    if not results:
        return {}

    metrics = ['overall_satisfaction', 'communication_quality', 'problem_solving_approach', 'efficiency', 'user_preference_alignment']

    def calculate_averages(subset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate averages for a subset of results"""
        averages = {}
        for metric in metrics:
            values = [r['metrics'][metric] for r in subset if r['metrics'][metric] > 0]
            averages[metric] = sum(values) / len(values) if values else 0
        return averages

    # Categorize results
    stateful_results = [r for r in results if r['has_user_profile']]
    non_stateful_results = [r for r in results if not r['has_user_profile']]
    with_eval_results = [r for r in results if r.get('has_eval_results', False)]

    # Calculate task success correlation, contingency analysis, and confidence intervals
    correlation_analysis = calculate_task_success_correlation(results)
    contingency_analysis = calculate_contingency_analysis(results)
    confidence_intervals = calculate_confidence_intervals(results)

    return {
        'summary': {
            'total_instances': len(results),
            'stateful_instances': len(stateful_results),
            'non_stateful_instances': len(non_stateful_results),
            'instances_with_eval_results': len(with_eval_results),
            'average_trajectory_length': sum(r['trajectory_length'] for r in results) / len(results)
        },
        'overall_averages': calculate_averages(results),
        'stateful_averages': calculate_averages(stateful_results),
        'non_stateful_averages': calculate_averages(non_stateful_results),
        'correlation_analysis': correlation_analysis,
        'contingency_analysis': contingency_analysis,
        'confidence_intervals': confidence_intervals,
        'detailed_results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate user satisfaction from SWE-Bench trajectories')
    parser.add_argument('--input-dir', type=str,
                        help='Full path to directory containing output.jsonl files')
    parser.add_argument('--agent', type=str,
                        help='Agent name (e.g., TomCodeActAgent)')
    parser.add_argument('--model', type=str,
                        help='Model name (e.g., claude-sonnet-4-20250514)')
    parser.add_argument('--rag', action='store_true', default=False,
                        help='Whether to use RAG for evaluation')
    parser.add_argument('--dataset', type=str, default='cmu-lti__stateful-test',
                        help='Dataset name (default: cmu-lti__stateful-test)')
    parser.add_argument('--llm-config', type=str, default='llm.fake_user',
                        help='LLM config name for evaluation')
    parser.add_argument('--max-instances', type=int, default=None,
                        help='Maximum number of instances to evaluate per file')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for concurrent evaluation (default: 64)')
    parser.add_argument('--recursive', action='store_true',
                        help='Search for output.jsonl files recursively')
    parser.add_argument('--include-eval-results', action='store_true', default=True,
                        help='Include technical evaluation results from report.json files (default: True)')

    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_path = Path(args.input_dir)
    elif args.agent and args.model:
        # Construct path automatically
        base_dir = Path("/home/xuhuizhou/OpenHands/evaluation/evaluation_outputs/outputs")
        # Find the directory that matches the pattern
        pattern = f"{args.dataset}/{args.agent}/{args.model}*"
        matches = list(base_dir.glob(pattern))
        if not matches:
            print(f"No directories found matching pattern: {pattern}")
            sys.exit(1)
        input_path = matches[0]  # Use the first match
        print(f"Using directory: {input_path}")
    else:
        parser.error("Either --input-dir or both --agent and --model must be provided")

    # Initialize evaluator
    evaluator = FakeUserEvaluator(args.llm_config)

    # Find all output files
    if args.recursive:
        output_files = list(input_path.rglob('output.jsonl'))
    else:
        output_files = list(input_path.glob('*/output.jsonl'))
        output_files.extend(list(input_path.glob('output.jsonl')))

    # Filter out any output files that don't have corresponding eval_outputs if required
    if args.include_eval_results:
        filtered_files = []
        for f in output_files:
            eval_dir = f.parent / 'eval_outputs'
            if eval_dir.exists():
                filtered_files.append(f)
            else:
                print(f"Warning: No eval_outputs directory found for {f}, skipping...")
        output_files = filtered_files

    if not output_files:
        print(f"No output.jsonl files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(output_files)} output files to evaluate")

    # Evaluate each file
    all_results = {}

    async def evaluate_all_files():
        for output_file in output_files:
            print(f"\nEvaluating {output_file}...")
            file_results = await evaluate_output_file(str(output_file), evaluator, args.max_instances, args.batch_size)
            all_results[str(output_file)] = file_results

    # Run async evaluation
    asyncio.run(evaluate_all_files())

    # Save results to input directory
    output_file = input_path / 'user_satisfaction.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\nEvaluation complete! Results saved to {output_file}")

    # Print comprehensive summary
    total_instances = sum(result['summary']['total_instances'] for result in all_results.values() if result)
    if total_instances > 0:
        print(f"\nCOMPREHENSIVE EVALUATION SUMMARY ({total_instances} instances):")

        # Calculate overall averages across all files
        all_instance_results = []
        for file_results in all_results.values():
            if file_results and 'detailed_results' in file_results:
                all_instance_results.extend(file_results['detailed_results'])

        if all_instance_results:
            overall_aggregated = aggregate_results(all_instance_results)
            summary = overall_aggregated['summary']
            averages = overall_aggregated['overall_averages']

            print(f"\nOverall Results:")
            print(f"  Total instances: {summary['total_instances']}")
            print(f"  Stateful instances: {summary['stateful_instances']}")
            print(f"  Instances with evaluation results: {summary['instances_with_eval_results']}")
            print(f"  Average trajectory length: {summary['average_trajectory_length']:.1f} events")

            print(f"\nUser Satisfaction Metrics (1-5 scale):")
            print(f"  Overall Satisfaction: {averages['overall_satisfaction']:.2f}/5")
            print(f"  Communication Quality: {averages['communication_quality']:.2f}/5")
            print(f"  Problem Solving Approach: {averages['problem_solving_approach']:.2f}/5")
            print(f"  Efficiency: {averages['efficiency']:.2f}/5")
            print(f"  User Preference Alignment: {averages['user_preference_alignment']:.2f}/5")

            # Display confidence intervals
            confidence_intervals = overall_aggregated.get('confidence_intervals', {})
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
            correlation_analysis = overall_aggregated.get('correlation_analysis', {})
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
            elif correlation_analysis.get('error'):
                print(f"\nTask Success vs User Satisfaction Correlation:")
                print(f"  Error: {correlation_analysis['error']}")

            # Display contingency analysis
            contingency_analysis = overall_aggregated.get('contingency_analysis', {})
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
            elif contingency_analysis.get('error'):
                print(f"\nContingency Analysis:")
                print(f"  Error: {contingency_analysis['error']}")

            if summary['stateful_instances'] > 0:
                print(f"\nStateful Mode Analysis ({summary['stateful_instances']} instances):")
                stateful_avg = overall_aggregated['stateful_averages']
                print(f"  Overall Satisfaction: {stateful_avg['overall_satisfaction']:.2f}/5")
                print(f"  User Preference Alignment: {stateful_avg['user_preference_alignment']:.2f}/5")
                print(f"  Communication Quality: {stateful_avg['communication_quality']:.2f}/5")


if __name__ == '__main__':
    main()

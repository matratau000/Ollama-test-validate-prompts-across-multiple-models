import asyncio
import os
import json
import re
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
from ollama import AsyncClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Sample benchmark questions with reference answers
BENCHMARK_QUESTIONS = [
    {
        "category": "general_knowledge",
        "question": "What is the capital of France?",
        "reference_answer": "The capital of France is Paris.",
        "key_points": ["Paris", "capital city", "France"]
    },
    {
        "category": "science",
        "question": "Explain the process of photosynthesis briefly.",
        "reference_answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It converts carbon dioxide and water into glucose and oxygen.",
        "key_points": ["sunlight", "plants", "chlorophyll", "carbon dioxide", "water", "glucose", "oxygen"]
    },
    {
        "category": "math",
        "question": "What is the Pythagorean theorem?",
        "reference_answer": "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse equals the sum of the squares of the other two sides. It can be written as a² + b² = c², where c is the hypotenuse.",
        "key_points": ["right-angled triangle", "hypotenuse", "square", "sum", "a² + b² = c²"]
    },
    {
        "category": "programming",
        "question": "What is the difference between a list and a tuple in Python?",
        "reference_answer": "Lists and tuples are both sequence data types in Python, but lists are mutable (can be changed after creation) while tuples are immutable (cannot be modified after creation). Lists use square brackets [] and tuples use parentheses ().",
        "key_points": ["mutable", "immutable", "sequence", "square brackets", "parentheses"]
    },
    {
        "category": "reasoning",
        "question": "If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?",
        "reference_answer": "The train will travel 150 miles in 2.5 hours. This is calculated by multiplying the speed (60 mph) by the time (2.5 hours): 60 × 2.5 = 150 miles.",
        "key_points": ["150 miles", "multiplication", "distance = speed × time"]
    }
]

def read_responses(directory):
    responses = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            model = filename.split('_response_')[0]
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read().strip()
                if model not in responses:
                    responses[model] = []
                responses[model].append(content)
    return responses

def extract_json(content):
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None

json_prompt = """
You are an AI model analyzing the following responses. Evaluate each response based on the criteria provided and return the results in JSON format. The JSON object should have the following structure:

{{
    "accuracy": float,  # Accuracy of the response (0.0 to 1.0)
    "completeness": float,  # Completeness of the response (0.0 to 1.0)
    "clarity": float,  # Clarity of the response (0.0 to 1.0)
    "relevance": float,  # Relevance to the question (0.0 to 1.0)
    "reasoning": float,  # Quality of reasoning/logic (0.0 to 1.0)
    "factual_correctness": float,  # Factual correctness (0.0 to 1.0)
    "hallucination_score": float,  # Level of hallucination (0.0 = none, 1.0 = severe)
    "bias_score": float,  # Level of bias detected (0.0 = none, 1.0 = severe)
    "creativity": float,  # Level of creativity if applicable (0.0 to 1.0)
    "feedback": str,  # Detailed feedback on the response
    "strengths": [str],  # List of specific strengths
    "weaknesses": [str]  # List of specific weaknesses
}}

Ensure your response contains ONLY the JSON object, with no additional text before or after. Here is the response to analyze:
"""

# Enhanced evaluation prompt with reference answer
json_prompt_with_reference = """
You are an AI model analyzing the following response to a question. Evaluate the response based on the criteria provided, comparing it to the reference answer. Return the results in JSON format. The JSON object should have the following structure:

{{
    "accuracy": float,  # Accuracy of the response compared to reference (0.0 to 1.0)
    "completeness": float,  # Completeness of the response (0.0 to 1.0)
    "clarity": float,  # Clarity of the response (0.0 to 1.0)
    "relevance": float,  # Relevance to the question (0.0 to 1.0)
    "reasoning": float,  # Quality of reasoning/logic (0.0 to 1.0)
    "factual_correctness": float,  # Factual correctness (0.0 to 1.0)
    "hallucination_score": float,  # Level of hallucination (0.0 = none, 1.0 = severe)
    "bias_score": float,  # Level of bias detected (0.0 = none, 1.0 = severe)
    "creativity": float,  # Level of creativity if applicable (0.0 to 1.0)
    "key_points_coverage": float,  # Percentage of key points covered (0.0 to 1.0)
    "feedback": str,  # Detailed feedback on the response
    "strengths": [str],  # List of specific strengths
    "weaknesses": [str],  # List of specific weaknesses
    "missing_key_points": [str]  # Key points from the reference that are missing
}}

Ensure your response contains ONLY the JSON object, with no additional text before or after.

Question: {question}
Reference Answer: {reference_answer}
Key Points to Look For: {key_points}
Model Response: {response}
"""

async def ask_question(client, question, model, idx):
    os.makedirs('responses', exist_ok=True)
    file_path = f'responses/{model.replace(":", "_")}_response_{idx}.txt'
    
    # Track performance metrics
    start_time = time.time()
    token_count = 0
    
    with open(file_path, 'w') as f:
        async for part in (await client.chat(model=model, messages=[{'role': 'user', 'content': question}], stream=True)):
            f.write(part['message']['content'])
            # Count tokens if the information is available
            if 'prompt_eval_count' in part and 'eval_count' in part:
                token_count += part['eval_count']
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save performance metrics
    os.makedirs('metrics', exist_ok=True)
    metrics_file = f'metrics/{model.replace(":", "_")}_metrics_{idx}.json'
    
    metrics = {
        "model": model,
        "processing_time_seconds": processing_time,
        "tokens_generated": token_count,
        "tokens_per_second": token_count / processing_time if processing_time > 0 else 0,
        "question_length": len(question),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def read_metrics(directory):
    """Read performance metrics from JSON files in the metrics directory."""
    metrics = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                try:
                    data = json.load(file)
                    model = data.get("model")
                    if model:
                        if model not in metrics:
                            metrics[model] = []
                        metrics[model].append(data)
                except json.JSONDecodeError:
                    continue
    return metrics

async def analyze_with_retry(client, model, response, question=None, reference_answer=None, key_points=None, max_retries=3):
    for _ in range(max_retries):
        if reference_answer:
            # Use prompt with reference if we have one
            prompt = json_prompt_with_reference.format(
                question=question,
                reference_answer=reference_answer,
                key_points=key_points,
                response=response
            )
        else:
            # Use standard prompt
            prompt = f"{json_prompt}\nResponse: {response}"
            
        result = await client.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are an AI model analyzing responses.'},
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )
        content = result['message']['content']
        json_data = extract_json(content)
        if json_data:
            return json_data
    return {
        "accuracy": None,
        "completeness": None,
        "clarity": None,
        "relevance": None,
        "reasoning": None,
        "factual_correctness": None,
        "hallucination_score": None,
        "bias_score": None,
        "creativity": None,
        "feedback": "Failed to parse JSON after multiple attempts",
        "strengths": [],
        "weaknesses": []
    }

async def analyze_response_with_llm(client, model, responses, questions=None):
    analysis = {}
    tasks = []
    
    for idx, response in enumerate(responses):
        if questions and idx < len(questions):
            # If we have benchmark questions and this index exists
            q = questions[idx]
            tasks.append(analyze_with_retry(
                client, model, response, 
                question=q["question"], 
                reference_answer=q["reference_answer"], 
                key_points=q["key_points"]
            ))
        else:
            # Regular analysis without reference
            tasks.append(analyze_with_retry(client, model, response))
            
    results = await asyncio.gather(*tasks)
    for idx, result in enumerate(results):
        analysis[f"response_{idx}"] = result
    return analysis

async def multi_evaluator_analysis(client, evaluator_models, responses, questions=None):
    """Run analysis using multiple evaluator models and aggregate the results."""
    all_evaluations = {}
    
    for evaluator in evaluator_models:
        print(f"Evaluating responses using {evaluator} as evaluator...")
        all_evaluations[evaluator] = await analyze_response_with_llm(
            client, evaluator, responses, questions
        )
    
    # Aggregate results across evaluators
    aggregated = {}
    for idx in range(len(responses)):
        response_key = f"response_{idx}"
        aggregated[response_key] = {}
        
        # Get all metrics that exist in any evaluator
        all_metrics = set()
        for evaluator in evaluator_models:
            if response_key in all_evaluations[evaluator]:
                all_metrics.update(all_evaluations[evaluator][response_key].keys())
        
        # Skip non-numeric and special fields for aggregation
        skip_fields = ["feedback", "strengths", "weaknesses", "missing_key_points"]
        numeric_metrics = [m for m in all_metrics if m not in skip_fields]
        
        # Average each metric across evaluators
        for metric in numeric_metrics:
            values = []
            for evaluator in evaluator_models:
                if (response_key in all_evaluations[evaluator] and 
                    metric in all_evaluations[evaluator][response_key] and
                    all_evaluations[evaluator][response_key][metric] is not None):
                    values.append(all_evaluations[evaluator][response_key][metric])
            
            if values:
                aggregated[response_key][metric] = sum(values) / len(values)
            else:
                aggregated[response_key][metric] = None
        
        # Concatenate feedback from all evaluators
        for field in ["feedback", "strengths", "weaknesses", "missing_key_points"]:
            if field in all_metrics:
                combined = []
                for evaluator in evaluator_models:
                    if (response_key in all_evaluations[evaluator] and 
                        field in all_evaluations[evaluator][response_key]):
                        value = all_evaluations[evaluator][response_key][field]
                        if isinstance(value, list):
                            combined.extend(value)
                        elif value:
                            combined.append(f"[{evaluator}]: {value}")
                
                if field in ["strengths", "weaknesses", "missing_key_points"]:
                    # Remove duplicates while preserving order
                    aggregated[response_key][field] = list(dict.fromkeys(combined))
                else:
                    aggregated[response_key][field] = "\n".join(combined) if combined else ""
    
    return {
        "aggregated": aggregated,
        "by_evaluator": all_evaluations
    }

def calculate_statistics(model_scores):
    """Calculate mean, standard deviation, and 95% confidence interval for model scores."""
    mean_scores = {}
    std_devs = {}
    conf_intervals = {}
    
    for model, scores in model_scores.items():
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        mean_scores[model] = mean_score
        
        # Only calculate std_dev and confidence intervals if we have more than 1 sample
        if len(scores) > 1:
            std_dev = np.std(scores_array, ddof=1)  # Using ddof=1 for sample standard deviation
            conf_interval = 1.96 * (std_dev / np.sqrt(len(scores)))  # 95% confidence interval
            std_devs[model] = std_dev
            conf_intervals[model] = conf_interval
        else:
            # If only one score, set std_dev to 0 and conf_interval to None
            std_devs[model] = 0
            conf_intervals[model] = 0
    
    return mean_scores, std_devs, conf_intervals

def generate_radar_analysis(model_metrics, metric_names):
    """Generate textual analysis for radar chart data."""
    analysis = "Radar Chart Analysis:\n\n"
    
    # Find the best model for each metric
    best_models = {}
    for metric in metric_names:
        best_score = -1
        best_model = None
        for model, metrics in model_metrics.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model
        if best_model:
            best_models[metric] = (best_model, best_score)
    
    # Add best model for each metric to the analysis
    analysis += "Top performers by metric:\n"
    for metric, (model, score) in best_models.items():
        formatted_metric = metric.replace('_', ' ').title()
        analysis += f"- {formatted_metric}: {model} ({score:.2f})\n"
    
    # Calculate overall scores (average across all metrics)
    overall_scores = {}
    for model, metrics in model_metrics.items():
        valid_metrics = [metrics.get(metric, 0) for metric in metric_names if metric in metrics]
        if valid_metrics:
            overall_scores[model] = sum(valid_metrics) / len(valid_metrics)
    
    # Find the best overall model
    if overall_scores:
        best_overall = max(overall_scores.items(), key=lambda x: x[1])
        analysis += f"\nBest overall model: {best_overall[0]} with an average score of {best_overall[1]:.2f}\n"
    
    # Add insights about model strengths and weaknesses
    analysis += "\nKey insights:\n"
    for model, metrics in model_metrics.items():
        if metrics:
            # Find the model's best and worst metrics
            model_metrics_values = [(metric, metrics.get(metric, 0)) for metric in metric_names if metric in metrics]
            if model_metrics_values:
                best_metric = max(model_metrics_values, key=lambda x: x[1])
                worst_metric = min(model_metrics_values, key=lambda x: x[1])
                
                best_formatted = best_metric[0].replace('_', ' ').title()
                worst_formatted = worst_metric[0].replace('_', ' ').title()
                
                analysis += f"- {model}: Excels at {best_formatted} ({best_metric[1]:.2f}), "
                analysis += f"struggles with {worst_formatted} ({worst_metric[1]:.2f})\n"
    
    return analysis

def generate_radar_chart(model_metrics, metric_names):
    """Generate a radar chart comparing model performance across different metrics."""
    if not model_metrics or not metric_names:
        print("No data available for radar chart")
        return
    
    # Set up the radar chart with gridspec for analysis
    plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
    
    ax = plt.subplot(gs[0], polar=True)
    
    # Number of metrics
    N = len(metric_names)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add labels at each angle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metric_names])
    
    # Draw lines connecting labels through the center
    ax.set_rlabel_position(0)
    
    # Draw y-ticks (radial lines) from the center to the 1.0 circle
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot each model's metrics
    for model, metrics in model_metrics.items():
        values = [metrics.get(metric, 0) for metric in metric_names]
        values += values[:1]  # Close the loop
        
        # Plot the values
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add the title
    plt.title('Model Performance Comparison Across Metrics', size=15, y=1.1)
    
    # Add analysis text
    text_ax = plt.subplot(gs[1])
    text_ax.axis('off')
    
    analysis = generate_radar_analysis(model_metrics, metric_names)
    text_ax.text(0.02, 0.95, analysis, fontsize=10, va='top', ha='left', multialignment='left')
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison_radar.png', dpi=300, bbox_inches='tight')
    print("Radar chart saved to visualizations/model_comparison_radar.png")

def generate_bar_analysis(model_summaries, metric):
    """Generate textual analysis for a bar chart of a specific metric."""
    analysis = f"{metric.replace('_', ' ').title()} Analysis:\n\n"
    
    # Extract scores and models for this metric
    scores = []
    models = []
    for model, metrics in model_summaries.items():
        if metric in metrics:
            scores.append(metrics[metric])
            models.append(model)
    
    if not scores:
        return "No data available for analysis."
    
    # Find the best and worst model for this metric
    best_idx = scores.index(max(scores))
    worst_idx = scores.index(min(scores))
    best_model = models[best_idx]
    worst_model = models[worst_idx]
    
    analysis += f"- Top performer: {best_model} ({scores[best_idx]:.2f})\n"
    analysis += f"- Lowest performer: {worst_model} ({scores[worst_idx]:.2f})\n"
    
    # Calculate average and spread
    avg_score = sum(scores) / len(scores)
    analysis += f"- Average score across all models: {avg_score:.2f}\n"
    
    if len(scores) > 1:
        score_range = max(scores) - min(scores)
        analysis += f"- Score range: {score_range:.2f}\n"
    
    # Add metric-specific analysis
    if metric == 'accuracy':
        analysis += "\nAccuracy Insights:\n"
        if max(scores) > 0.8:
            analysis += f"- {best_model} demonstrates excellent accuracy, suggesting reliable responses for factual queries.\n"
        elif max(scores) > 0.6:
            analysis += f"- {best_model} shows good accuracy but has room for improvement on factual responses.\n"
        else:
            analysis += "- All models show moderate to low accuracy, suggesting caution when using them for factual information.\n"
            
    elif metric == 'coherence':
        analysis += "\nCoherence Insights:\n"
        if max(scores) > 0.8:
            analysis += f"- {best_model} produces highly coherent responses with logical flow and structure.\n"
        else:
            analysis += f"- Even the best model ({best_model}) has room for improvement in response coherence.\n"
            
    elif metric == 'hallucination_score':
        analysis += "\nHallucination Insights:\n"
        if min(scores) < 0.2:
            analysis += f"- {best_model} shows minimal hallucination, making it more reliable for factual use cases.\n"
        elif min(scores) < 0.4:
            analysis += f"- {best_model} demonstrates moderate hallucination levels. Verification of outputs is recommended.\n"
        else:
            analysis += "- All models show significant hallucination tendencies. Use with caution for factual queries.\n"
            
    elif metric == 'relevance':
        analysis += "\nRelevance Insights:\n"
        if max(scores) > 0.8:
            analysis += f"- {best_model} consistently provides highly relevant responses to queries.\n"
        else:
            analysis += "- Models show moderate relevance scores, suggesting they may sometimes drift from the core query topic.\n"
            
    # Add comparison to benchmark if we have one
    if 'benchmark' in models:
        benchmark_idx = models.index('benchmark')
        benchmark_score = scores[benchmark_idx]
        better_than_benchmark = [model for i, model in enumerate(models) if scores[i] > benchmark_score and model != 'benchmark']
        
        if better_than_benchmark:
            analysis += f"\n- Models outperforming the benchmark: {', '.join(better_than_benchmark)}\n"
        else:
            analysis += "\n- No models outperformed the benchmark on this metric.\n"
    
    return analysis

def generate_bar_chart(model_summaries, metric_name, std_devs=None, conf_intervals=None):
    """Generate a bar chart for a specific metric across all models."""
    # Set up the figure with gridspec for analysis
    plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
    
    ax = plt.subplot(gs[0])
    
    models = []
    scores = []
    errors = []
    
    # Collect data for the chart
    for model, metrics in model_summaries.items():
        if metric_name in metrics:
            models.append(model)
            scores.append(metrics[metric_name])
            
            # Add error bars if available
            if std_devs and conf_intervals and model in std_devs and model in conf_intervals:
                errors.append(conf_intervals[model])
            else:
                errors.append(0)
    
    # Sort by score
    if scores:
        sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
        models = [models[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        errors = [errors[i] for i in sorted_indices]
    
    # Set colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    
    # Plot bars
    bars = ax.bar(models, scores, yerr=errors, capsize=5, color=colors)
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Formatting
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison')
    
    # Add analysis text
    text_ax = plt.subplot(gs[1])
    text_ax.axis('off')
    
    analysis = generate_bar_analysis(model_summaries, metric_name)
    text_ax.text(0.02, 0.95, analysis, fontsize=10, va='top', ha='left', multialignment='left')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{metric_name}_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Bar chart for {metric_name} saved to visualizations/{metric_name}_comparison.png")

async def main():
    # Ask user whether to use custom question or benchmark
    use_benchmark = input("Use benchmark questions? (y/n): ").lower() == 'y'
    
    if use_benchmark:
        # Ask which category of benchmark questions to use
        print("\nBenchmark categories:")
        categories = list(set(q["category"] for q in BENCHMARK_QUESTIONS))
        for i, cat in enumerate(categories):
            print(f"{i+1}. {cat}")
        print(f"{len(categories)+1}. All categories (random selection)")
        
        while True:
            try:
                choice = int(input("\nSelect category (number): "))
                if 1 <= choice <= len(categories) + 1:
                    break
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")
        
        if choice <= len(categories):
            selected_category = categories[choice-1]
            possible_questions = [q for q in BENCHMARK_QUESTIONS if q["category"] == selected_category]
        else:
            # Random selection from all categories
            possible_questions = BENCHMARK_QUESTIONS
            
        # Select the number of questions to use
        num_questions = min(5, len(possible_questions))
        selected_questions = random.sample(possible_questions, num_questions)
        
        # Use selected benchmark questions
        questions = selected_questions
        num_requests = len(questions)
    else:
        # Use custom question
        custom_question = input("Enter your question: ")
        questions = [{"question": custom_question, "reference_answer": None, "key_points": None}]
        num_requests = 5  # Default for custom questions
    
    # Ask if user wants to use multiple evaluators
    use_multi_eval = input("Use multiple evaluator models? (y/n): ").lower() == 'y'
    
    # Models that are actually available on the system and 8B or less
    models = ["deepseek-r1:8b", "llama3.1:8b", "gemma2:2b", "command-r7b:latest", "llama3.2:3b"]
    default_evaluator = "deepseek-r1:8b"  # Define default evaluator
    
    if use_multi_eval:
        # Select evaluator models
        print("\nAvailable evaluator models:")
        for i, model in enumerate(models):
            print(f"{i+1}. {model}")
        
        evaluator_indices = input("\nEnter numbers of models to use as evaluators (comma-separated, e.g. 1,3): ")
        try:
            indices = [int(idx.strip()) - 1 for idx in evaluator_indices.split(",")]
            evaluator_models = [models[idx] for idx in indices if 0 <= idx < len(models)]
            if not evaluator_models:
                print(f"No valid evaluators selected. Using {default_evaluator} as default.")
                evaluator_models = [default_evaluator]
        except (ValueError, IndexError):
            print(f"Invalid input. Using {default_evaluator} as default.")
            evaluator_models = [default_evaluator]
    else:
        evaluator_models = [default_evaluator]  # Default evaluator
    
    client = AsyncClient()

    # Generate responses
    tasks = []
    performance_metrics = []
    for model in models:
        for i in range(num_requests):
            # For benchmark, use the question at position i
            question_text = questions[i]["question"] if use_benchmark and i < len(questions) else questions[0]["question"]
            task = ask_question(client, question_text, model, i)
            tasks.append(task)
    
    # Gather metrics from all tasks
    performance_results = await asyncio.gather(*tasks)
    for metric in performance_results:
        if metric:  # Filter out None results
            performance_metrics.append(metric)

    # Analyze responses
    responses = read_responses("responses")
    full_analysis = {}
    
    for model, responses_list in responses.items():
        print(f"Analyzing responses for model: {model}")
        if use_multi_eval:
            # Use multiple evaluators
            analysis_result = await multi_evaluator_analysis(
                client, evaluator_models, responses_list,
                questions if use_benchmark else None
            )
            # Use the aggregated results for summary
            full_analysis[model] = analysis_result["aggregated"]
            
            # Save detailed per-evaluator results
            with open(f"analysis_results_{model}_by_evaluator.json", "w") as f:
                json.dump(analysis_result["by_evaluator"], f, indent=4)
        else:
            # Single evaluator
            if use_benchmark:
                analysis = await analyze_response_with_llm(client, evaluator_models[0], responses_list, questions)
            else:
                analysis = await analyze_response_with_llm(client, evaluator_models[0], responses_list)
            full_analysis[model] = analysis

    # Write analysis results
    with open("analysis_results.json", "w") as f:
        json.dump(full_analysis, f, indent=4)

    # Prepare data for statistics and visualization
    metric_names = ["accuracy", "completeness", "clarity", "relevance", 
                   "reasoning", "factual_correctness", "hallucination_score", 
                   "bias_score", "creativity"]
    
    if use_benchmark and any(r.get("key_points_coverage") for model in full_analysis.values() 
                              for r in model.values()):
        metric_names.append("key_points_coverage")
    
    model_summaries = {}
    model_stats = {}
    
    print("\nAnalysis Summary:")
    
    for model, analysis in full_analysis.items():
        model_summaries[model] = {}
        model_stats[model] = {}
        
        print(f"\n{model}:")
        
        # For each metric, calculate statistics
        for metric in metric_names:
            values = [r[metric] for r in analysis.values() if r[metric] is not None]
            
            if values:
                mean, std_dev, ci_95 = calculate_statistics(values)
                model_summaries[model][metric] = mean
                model_stats[model][metric] = {
                    "mean": mean,
                    "std_dev": std_dev,
                    "ci_95_low": ci_95[0],
                    "ci_95_high": ci_95[1]
                }
                
                # Print summary statistics
                ci_str = f"95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]" if ci_95[0] is not None else "N/A"
                print(f"  {metric.replace('_', ' ').title()}: {mean:.2f} (±{std_dev:.2f}, {ci_str})")
    
    # Generate comparison table
    table_data = []
    headers = ["Metric"] + list(model_summaries.keys())
    
    for metric in metric_names:
        row = [metric.replace('_', ' ').title()]
        for model in model_summaries.keys():
            value = model_summaries[model].get(metric)
            row.append(f"{value:.2f}" if value is not None else "N/A")
        table_data.append(row)
    
    print("\nModel Comparison Table:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save detailed statistics to a file
    with open("analysis_statistics.json", "w") as f:
        json.dump(model_stats, f, indent=4)
    
    # Generate radar chart
    os.makedirs("visualizations", exist_ok=True)
    generate_radar_chart(model_summaries, metric_names)
    
    # Generate individual bar charts for each metric
    for metric in metric_names:
        generate_bar_chart(model_summaries, metric)
    
    print("\nVisualization files have been saved to the 'visualizations' directory.")

    # Calculate and display performance metrics
    performance_data = read_metrics("metrics")
    
    print("\nPerformance Metrics Summary:")
    performance_table = []
    perf_headers = ["Model", "Avg Time (s)", "Tokens/s", "Total Tokens"]
    
    for model, metrics_list in performance_data.items():
        avg_time = sum(m['processing_time_seconds'] for m in metrics_list) / len(metrics_list)
        total_tokens = sum(m['tokens_generated'] for m in metrics_list)
        tokens_per_sec = sum(m['tokens_per_second'] for m in metrics_list) / len(metrics_list)
        
        performance_table.append([
            model,
            f"{avg_time:.2f}",
            f"{tokens_per_sec:.2f}",
            total_tokens
        ])
    
    print(tabulate(performance_table, headers=perf_headers, tablefmt="grid"))
    
    # Generate performance visualization
    if performance_data:
        # Response time visualization with analysis
        plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
        
        ax = plt.subplot(gs[0])
        models = list(performance_data.keys())
        avg_times = [sum(m['processing_time_seconds'] for m in data) / len(data) for model, data in performance_data.items()]
        
        bars = ax.bar(models, avg_times, color='lightblue')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}s', ha='center', va='bottom')
                    
        ax.set_title('Average Response Time by Model')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Model')
        
        # Generate performance analysis text
        fastest_model = models[avg_times.index(min(avg_times))]
        slowest_model = models[avg_times.index(max(avg_times))]
        avg_response_time = sum(avg_times) / len(avg_times)
        
        text_ax = plt.subplot(gs[1])
        text_ax.axis('off')  # Hide axes for the text subplot
        
        analysis = f"Response Time Analysis:\n\n"
        analysis += f"- Fastest model: {fastest_model} ({min(avg_times):.2f}s)\n"
        analysis += f"- Slowest model: {slowest_model} ({max(avg_times):.2f}s)\n"
        analysis += f"- Average response time across all models: {avg_response_time:.2f}s\n"
        analysis += f"- Performance difference: {fastest_model} is {(max(avg_times)/min(avg_times)):.1f}x faster than {slowest_model}\n\n"
        
        if max(avg_times) > 10:
            analysis += "Some models show significant response delays. This could impact user experience in real-time applications. "
            analysis += f"Consider optimizing {slowest_model} or prioritizing {fastest_model} for time-sensitive tasks.\n"
        else:
            analysis += "All models demonstrate reasonable response times for most applications. "
        
        # Add size comparison if available in the performance data
        if any('tokens_generated' in m for data in performance_data.values() for m in data):
            total_tokens = {model: sum(m.get('tokens_generated', 0) for m in data) for model, data in performance_data.items()}
            analysis += f"\nResponse size comparison: {max(total_tokens.items(), key=lambda x: x[1])[0]} generated the most tokens "
            analysis += f"while {min(total_tokens.items(), key=lambda x: x[1])[0]} generated the fewest."
        
        text_ax.text(0.02, 0.8, analysis, fontsize=10, wrap=True, va='top', ha='left')
        
        plt.tight_layout()
        plt.savefig('visualizations/response_time_comparison.png', dpi=300, bbox_inches='tight')
        
        # Token generation speed comparison with analysis
        plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
        
        ax = plt.subplot(gs[0])
        tokens_per_sec = [sum(m.get('tokens_per_second', 0) for m in data) / len(data) for model, data in performance_data.items()]
        
        bars = ax.bar(models, tokens_per_sec, color='lightgreen')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        ax.set_title('Token Generation Speed by Model')
        ax.set_ylabel('Tokens per Second')
        ax.set_xlabel('Model')
        
        # Generate token speed analysis
        fastest_token_model = models[tokens_per_sec.index(max(tokens_per_sec))]
        slowest_token_model = models[tokens_per_sec.index(min(tokens_per_sec))]
        
        text_ax = plt.subplot(gs[1])
        text_ax.axis('off')
        
        analysis = f"Token Generation Speed Analysis:\n\n"
        analysis += f"- Fastest token generation: {fastest_token_model} ({max(tokens_per_sec):.2f} tokens/sec)\n"
        analysis += f"- Slowest token generation: {slowest_token_model} ({min(tokens_per_sec):.2f} tokens/sec)\n"
        analysis += f"- Speed ratio: {fastest_token_model} generates tokens {(max(tokens_per_sec)/min(tokens_per_sec)):.1f}x faster than {slowest_token_model}\n\n"
        
        # Total tokens generated by each model
        if any('tokens_generated' in m for data in performance_data.values() for m in data):
            total_tokens = {model: sum(m.get('tokens_generated', 0) for m in data) for model, data in performance_data.items()}
            analysis += f"- Most verbose model: {max(total_tokens.items(), key=lambda x: x[1])[0]} with {max(total_tokens.values())} tokens total\n"
            analysis += f"- Most concise model: {min(total_tokens.items(), key=lambda x: x[1])[0]} with {min(total_tokens.values())} tokens total\n\n"
        
        analysis += "Implications:\n"
        analysis += f"- For real-time applications requiring quick responses, {fastest_token_model} offers the best performance.\n"
        
        if fastest_token_model != fastest_model:
            analysis += f"- While {fastest_model} has the shortest overall response time, {fastest_token_model} generates tokens most efficiently once started.\n"
        
        if 'hallucination_score' in metric_names:
            fastest_hallucination = None
            for model, data in model_summaries.items():
                if model == fastest_token_model and data.get('hallucination_score') is not None:
                    fastest_hallucination = data.get('hallucination_score')
                    
            if fastest_hallucination is not None:
                if fastest_hallucination > 0.3:
                    analysis += f"- Note: While {fastest_token_model} is fast, it has a higher hallucination score ({fastest_hallucination:.2f}), suggesting a speed vs. accuracy tradeoff.\n"
                else:
                    analysis += f"- {fastest_token_model} balances speed with accuracy well, with a low hallucination score of {fastest_hallucination:.2f}.\n"
        
        text_ax.text(0.02, 0.8, analysis, fontsize=10, wrap=True, va='top', ha='left')
        
        plt.tight_layout()
        plt.savefig('visualizations/token_speed_comparison.png', dpi=300, bbox_inches='tight')
        
        # Create a combined performance graph
        plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1])
        
        # Normalized scores for easier comparison
        max_time = max(avg_times)
        norm_times = [1 - (t / max_time) for t in avg_times]  # Invert so lower time = higher score
        
        max_token_speed = max(tokens_per_sec)
        norm_token_speeds = [t / max_token_speed for t in tokens_per_sec]
        
        # Plot normalized metrics
        ax1 = plt.subplot(gs[0])
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, norm_times, width, label='Response Speed (normalized)', color='lightblue')
        ax1.bar(x + width/2, norm_token_speeds, width, label='Token Generation Speed (normalized)', color='lightgreen')
        
        ax1.set_ylabel('Normalized Score (higher is better)')
        ax1.set_title('Combined Performance Metrics (Normalized)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Add a combined score (average of normalized metrics)
        combined_scores = [(norm_times[i] + norm_token_speeds[i])/2 for i in range(len(models))]
        
        ax2 = plt.subplot(gs[1])
        bars = ax2.bar(models, combined_scores, color='purple', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        ax2.set_ylabel('Combined Performance Score')
        ax2.set_title('Overall Performance Rating')
        ax2.set_ylim(0, 1.1)
        
        # Add analysis text
        best_performer = models[combined_scores.index(max(combined_scores))]
        worst_performer = models[combined_scores.index(min(combined_scores))]
        
        text_ax = plt.subplot(gs[2])
        text_ax.axis('off')
        
        analysis = f"Overall Performance Analysis:\n\n"
        analysis += f"- Best overall performer: {best_performer} with a combined score of {max(combined_scores):.2f}\n"
        analysis += f"- Lowest overall performer: {worst_performer} with a combined score of {min(combined_scores):.2f}\n\n"
        
        analysis += "This combined score balances response speed with token generation efficiency, providing a holistic view of each model's performance characteristics.\n\n"
        
        # Add recommendations
        analysis += "Recommendations:\n"
        analysis += f"- For applications requiring rapid responses: Consider {fastest_model}\n"
        analysis += f"- For most efficient token generation: Use {fastest_token_model}\n"
        analysis += f"- For best balance of speed metrics: {best_performer} offers the optimal combination\n"
        
        text_ax.text(0.02, 0.8, analysis, fontsize=10, wrap=True, va='top', ha='left')
        
        plt.tight_layout()
        plt.savefig('visualizations/combined_performance.png', dpi=300, bbox_inches='tight')
        
        print("Enhanced performance visualizations have been saved to the 'visualizations' directory.")

if __name__ == '__main__':
    asyncio.run(main())
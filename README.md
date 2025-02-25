# Ollama Model Evaluation Framework

A comprehensive framework for evaluating and comparing different language models using Ollama.

## Features

- **Multi-Model Evaluation**: Compare responses from different models (gemma2, llama3, phi3, mistral, etc.)
- **Benchmark Questions**: Built-in benchmark questions with reference answers across different categories
- **Multi-Evaluator Support**: Cross-validate evaluations using multiple LLM evaluators
- **Comprehensive Metrics**: Accuracy, completeness, clarity, relevance, reasoning, factual correctness, hallucination detection, and more
- **Key Points Coverage**: For benchmark questions, measures how well responses cover key information points
- **Performance Metrics**: Track response time, token generation speed, and token usage
- **Statistical Analysis**: Includes mean, standard deviation, and confidence intervals
- **Rich Visualizations**: Generate radar charts, bar charts, and comparison tables
- **Detailed Reporting**: JSON output of all metrics for further analysis

## Installation

1. Ensure you have Python 3.8+ and [Ollama](https://ollama.ai/) installed.

2. Clone this repository:
```bash
git clone https://github.com/yourusername/ollama-model-evaluation.git
cd ollama-model-evaluation
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file (optional) for any environment variables.

## Usage

Run the evaluation script:

```bash
python test_and_validate.py
```

You'll be guided through the evaluation process:

1. Choose between custom questions or benchmark questions
2. If using benchmark questions, select a category
3. Choose whether to use multiple evaluator models
4. View detailed results and visualizations

## Output

The script generates several outputs:

- `responses/` - Directory containing raw model responses
- `metrics/` - Directory containing performance metrics for each model
- `visualizations/` - Directory containing charts and visualizations
- `analysis_results.json` - Detailed evaluation metrics for all models
- `analysis_statistics.json` - Statistical summary of evaluation results
- Console output with tabulated results and summaries

## Customization

- Add your own benchmark questions by modifying the `BENCHMARK_QUESTIONS` list in the script
- Adjust the models being tested by changing the `models` list
- Modify evaluation criteria by editing the JSON prompts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

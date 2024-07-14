import asyncio
import os
import json
import re
from ollama import AsyncClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

{
    "accuracy": float,  # Accuracy of the response (0.0 to 1.0)
    "completeness": float,  # Completeness of the response (0.0 to 1.0)
    "clarity": float,  # Clarity of the response (0.0 to 1.0)
    "feedback": str  # Detailed feedback on the response
}

Ensure your response contains ONLY the JSON object, with no additional text before or after. Here is the response to analyze:
"""

async def ask_question(client, question, model, idx):
    os.makedirs('responses', exist_ok=True)
    file_path = f'responses/{model.replace(":", "_")}_response_{idx}.txt'
    with open(file_path, 'w') as f:
        async for part in (await client.chat(model=model, messages=[{'role': 'user', 'content': question}], stream=True)):
            f.write(part['message']['content'])

async def analyze_with_retry(client, model, response, max_retries=3):
    for _ in range(max_retries):
        result = await client.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are an AI model analyzing responses.'},
                {'role': 'user', 'content': f"{json_prompt}\nResponse: {response}"}
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
        "feedback": "Failed to parse JSON after multiple attempts"
    }

async def analyze_response_with_llm(client, model, responses):
    analysis = {}
    tasks = [analyze_with_retry(client, model, response) for response in responses]
    results = await asyncio.gather(*tasks)
    for idx, result in enumerate(results):
        analysis[f"response_{idx}"] = result
    return analysis

async def main():
    question = input("Enter your question: ")
    num_requests = 5
    models = ["gemma2:9b", "llama3:8b", "phi3:latest", "mistral:latest"]
    analysis_model = "gemma2:9b"

    client = AsyncClient()

    # Generate responses
    tasks = []
    for model in models:
        for i in range(num_requests):
            tasks.append(ask_question(client, question, model, i))
    await asyncio.gather(*tasks)

    # Analyze responses
    responses = read_responses("responses")
    full_analysis = {}
    for model, responses_list in responses.items():
        print(f"Analyzing responses for model: {model}")
        analysis = await analyze_response_with_llm(client, analysis_model, responses_list)
        full_analysis[model] = analysis

    # Write analysis results
    with open("analysis_results.json", "w") as f:
        json.dump(full_analysis, f, indent=4)

    # Print summary
    print("\nAnalysis Summary:")
    for model, analysis in full_analysis.items():
        print(f"\n{model}:")
        avg_accuracy = sum(r['accuracy'] for r in analysis.values() if r['accuracy'] is not None) / len(analysis)
        avg_completeness = sum(r['completeness'] for r in analysis.values() if r['completeness'] is not None) / len(analysis)
        avg_clarity = sum(r['clarity'] for r in analysis.values() if r['clarity'] is not None) / len(analysis)
        print(f"  Average Accuracy: {avg_accuracy:.2f}")
        print(f"  Average Completeness: {avg_completeness:.2f}")
        print(f"  Average Clarity: {avg_clarity:.2f}")

if __name__ == '__main__':
    asyncio.run(main())
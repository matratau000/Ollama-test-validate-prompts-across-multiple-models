# Ollama-test-validate-prompts-across-multiple-models

Send out one prompt and have the responses be delivered to multiple models and quickly validate using another model of choice to analyze your results.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project allows you to send a single prompt to multiple AI models and validate their responses using another model. The validation results are then analyzed and summarized.

## Features

- Send prompts to multiple AI models.
- Validate responses using a specified model.
- Analyze and summarize the validation results.

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` (Python package installer)

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/Ollama-test-validate-prompts-across-multiple-models.git
    cd Ollama-test-validate-prompts-across-multiple-models
    ```

2. Create and activate a virtual environment:

    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory of the project and add your environment variables. For example:

    ```env
    API_KEY=your_api_key_here
    ```

## Usage

1. Run the main script:

    ```sh
    python test_and_validate.py
    ```

2. Enter your question when prompted.

3. The script will generate responses from multiple models, validate them, and save the analysis results in `analysis_results.json`.

## Configuration

- **Models**: You can configure the models to be used in the `main` function of `test_and_validate.py`.
- **Number of Requests**: Adjust the `num_requests` variable to change the number of requests sent to each model.
- **Analysis Model**: Set the `analysis_model` variable to specify which model to use for validation.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

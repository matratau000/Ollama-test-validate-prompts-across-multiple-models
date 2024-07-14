# Ollama-test-validate-prompts-across-multiple-models

Send a single prompt to multiple AI models, validate their responses using a specified model, and analyze the results efficiently.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Expected Output](#expected-output)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project enables users to send a single prompt to multiple AI models, validate their responses using a specified model, and analyze the validation results comprehensively.

## Features

- Dispatch prompts to multiple AI models.
- Validate responses using a designated model.
- Analyze and summarize validation results.

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
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=4
OLLAMA_SERVE_COMMAND=ollama serve
```

## Usage

1. Execute the main script:

```sh
python test_and_validate.py
```

2. Enter your question when prompted.

3. The script will generate responses from multiple models, validate them, and save the analysis results in `analysis_results.json`.

## Expected Output

After running the script with the initial question "What colors are in the rainbow", you can expect an output similar to the following in the `analysis_results.json` file:

```json
{
    "mistral_latest": {
        "response_0": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately describes a rainbow, including its cause, appearance, and the order of colors. It is also complete, providing all the necessary information, and written clearly and concisely."
        },
        "response_1": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately defines a rainbow, explains the scientific principles behind it, and lists the colors in the correct order. It is also well-written and easy to understand."
        },
        "response_2": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately and comprehensively defines a rainbow, explaining the scientific principles behind its formation and listing the seven colors with their corresponding wavelengths and frequencies. The language is clear and easy to understand."
        },
        "response_3": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately defines a rainbow, explains the causes, describes its appearance, lists the colors in the correct order, provides a mnemonic device, and mentions the wavelength relationship. It is comprehensive, easy to understand, and well-written."
        },
        "response_4": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately and comprehensively describes a rainbow, explaining its formation and the order of colors. The language is clear and easy to understand."
        }
    },
    "gemma2_9b": {
        "response_0": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the colors of the rainbow in the correct order and provides a helpful mnemonic device (ROYGBIV)."
        },
        "response_1": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the colors of the rainbow in order and provides a helpful mnemonic device. It is complete, clear, and well-formatted."
        },
        "response_2": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the colors of the rainbow and provides a helpful mnemonic device (ROYGBIV). It is complete, clear, and well-organized."
        },
        "response_3": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the colors of the rainbow and provides the acronym ROYGBIV for memorization. The response is complete and easy to understand."
        },
        "response_4": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the colors of the rainbow and provides a helpful acronym for remembering the order."
        }
    },
    "phi3_latest": {
        "response_0": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately defines the colors of a rainbow using the ROYGBIV acronym. It provides a complete list of the seven colors and acknowledges alternative models of color perception. The language is clear and easy to understand."
        },
        "response_1": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately defines the colors of a traditional rainbow using the acronym ROYGBIV and explains the phenomenon behind its formation. It is also complete, providing all the relevant information and acknowledges potential variations."
        },
        "response_2": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately defines the colors of a rainbow using the acronym ROYGBIV, provides a clear explanation of the process behind rainbow formation, and is well-organized and easy to understand."
        },
        "response_3": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the colors of a rainbow, explains the scientific reason behind their order, and provides the common acronym ROYGBIV. It is also clear and well-organized."
        },
        "response_4": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the colors of the rainbow in the correct order, explains the visible spectrum, provides the acronym ROYGBIV, and acknowledges variations in the inclusion of indigo."
        }
    },
    "llama3_8b": {
        "response_0": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately describes the colors of a rainbow and uses the mnemonic ROY G BIV. It also explains the reason for the specific color order. The response is complete, clear, and well-organized."
        },
        "response_1": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the traditional colors of the rainbow, uses the acronym ROYGBIV, and provides a concise explanation of the scientific basis for the rainbow's appearance. It is well-structured and easy to understand."
        },
        "response_2": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the classic rainbow colors in order and provides a brief explanation for their appearance."
        },
        "response_3": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the traditional colors of the rainbow in the correct order and explains the scientific phenomenon behind their appearance."
        },
        "response_4": {
            "accuracy": 1.0,
            "completeness": 1.0,
            "clarity": 1.0,
            "feedback": "The response accurately lists the colors of the rainbow in the correct order. It also provides a concise and clear explanation of the scientific phenomenon behind the rainbow."
        }
    }
}
```

## Configuration

- **Models**: Configure the models to be used in the `main` function of `test_and_validate.py`.
- **Number of Requests**: Adjust the `num_requests` variable to change the number of requests sent to each model.
- **Analysis Model**: Set the `analysis_model` variable to specify which model to use for validation.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

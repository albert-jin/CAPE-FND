# CAPE-FND: Context-Aware Prompt Engineering for Fake News Detection

## Overview
CAPE-FND is a novel framework designed to enhance the performance of Large Language Models (LLMs) in detecting fake news by incorporating veracity-oriented context-aware strategies. The framework integrates **veracity context-aware constraints**, **background information**, and **analogical reasoning** to mitigate LLM hallucinations and improve the accuracy of fake news detection. Additionally, it employs a **self-adaptive bootstrap prompting optimization** method to refine LLM prompts iteratively, ensuring robust and reliable predictions.

This repository contains the implementation of the CAPE-FND framework, including modules for context-aware knowledge acquisition, fake news detection, and prompt optimization. The code is designed to be modular and extensible, allowing researchers and practitioners to adapt it for various fake news detection tasks.

---

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Key Features](#key-features)
5. [Experiments](#experiments)
6. [Contributing](#contributing)
7. [Citation](#citation)
8. [License](#license)

---

## Installation
To set up the CAPE-FND framework, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/albert-jin/CAPE-FND.git
   cd CAPE-FND
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key in config.py:
   ```bash
    OPENAI_API_KEY = "your_openai_api_key"
    OPENAI_API_BASE = "https://api.openai.com/v1"
   ```
   
## Project Structure
   The project is organized as follows:

    ```
    CAPE-FND/
    │
    ├── main.py                  # Main entry point for the CAPE-FND framework
    ├── config.py                # Configuration file (API keys, paths, etc.)
    ├── cape_fnd/                # Core modules for CAPE-FND
    │   ├── __init__.py          # Initialization file
    │   ├── constraints.py       # Veracity context-aware constraints
    │   ├── background.py        # Veracity context-aware backgrounds
    │   ├── analogies.py         # Veracity context-aware analogies
    │   ├── detection.py         # Core fake news detection logic
    │   └── optimization.py      # Self-adaptive bootstrap prompting optimization
    ├── data/                    # Data folder
    │   ├── train.json           # Training data
    │   └── test.json            # Testing data
    └── utils/                   # Utility functions
        ├── __init__.py
        ├── logger.py            # Logging utilities
        └── file_utils.py        # File I/O utilities
   ```


---

## Usage
To run the CAPE-FND framework, execute the following command:
```bash
python main.py
```
### Key Steps:
1. **Veracity Context-Aware Knowledge Acquisition**:
   - The framework queries the LLM to generate veracity-oriented constraints, background information, and analogies for each news article. This step ensures that the LLM is provided with rich contextual information to improve its reasoning capabilities.

2. **Fake News Detection**:
   - The detection module integrates the acquired context-aware knowledge to predict the veracity of news articles. By combining constraints, backgrounds, and analogies, the framework enhances the LLM's ability to discern subtle cues in deceptive content.

3. **Self-Adaptive Bootstrap Prompting Optimization**:
   - The optimization module refines the prompts iteratively using a random search bootstrap algorithm. This process ensures that the LLM generates more accurate and reliable predictions by dynamically adjusting the prompts based on feedback.

---

## Key Features
- **Veracity Context-Aware Constraints**: Constructs linguistic variations of news articles and evaluates LLM responses for consistency. This reduces ambiguity and enhances the reliability of the model's predictions.
- **Veracity Context-Aware Backgrounds**: Provides concise background information about key entities or events in the news. This additional context helps the LLM make more informed judgments.
- **Veracity Context-Aware Analogies**: Generates relevant analogies to enhance LLM reasoning and understanding. By drawing parallels from past scenarios, the framework improves the model's ability to detect misinformation patterns.
- **Self-Adaptive Bootstrap Prompting Optimization**: Iteratively refines prompts to maximize LLM performance. This adaptive approach ensures robustness and reliability across diverse datasets.
- **Modular Design**: The framework is designed to be modular and extensible, allowing researchers and practitioners to adapt it for various fake news detection tasks.

---

## Experiments
The CAPE-FND framework has been extensively evaluated on multiple public datasets using GPT-3.5-turbo. The results demonstrate its effectiveness and robustness, outperforming advanced models like GPT-4.0 and human performance in certain scenarios.

### Datasets:
- **Train Data**: `data/FakeNewsNet/GossipCop/train.json, data/FakeNewsNet/PolitiFact/train.json, data/FANG/train.json, `
- **Test Data**: `data/FakeNewsNet/GossipCop/test.json, data/FakeNewsNet/PolitiFact/test.json, data/FANG/test.json`

### Results:
- Outputs are saved in the `results/` directory, including:
  - `output.json`: Initial detection results.
  - `optimized_output.json`: Results after prompt optimization.

---

## License

This project is licensed under the GNU General Public License v3.0 License. See the [LICENSE](LICENSE) file for more details.

---

## Citation
If you use CAPE-FND in your research, please cite our work:
```bibtex
@article{jin2023capefnd,
author = {Jin, Weiqiang and Gao, Yang and Tao, Tao and Wang, Xiujun and Wang, Ningwei and Wu, Baohai and Zhao, Biao},
title = {Veracity-Oriented Context-Aware Large Language Models–Based Prompting Optimization for Fake News Detection},
journal = {International Journal of Intelligent Systems},
volume = {2025},
number = {1},
pages = {5920142},
keywords = {analogical reasoning, bootstrap optimization, chain-of-thought, fake news detection, in-context learning, large language models, prompt engineering},
doi = {https://doi.org/10.1155/int/5920142},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1155/int/5920142},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1155/int/5920142},
year = {2025}
}
```
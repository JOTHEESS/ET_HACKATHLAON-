# FailureSensorIQ Evaluation & Benchmark Layer

This directory serves as the evaluation and benchmark layer utilizing the **FailureSensorIQ** framework developed by IBM Research. This layer is designed to assess the reasoning capabilities of Large Language Models (LLMs) on complex, domain-specific relationships between industrial sensors and failure modes across various industrial assets (Industry 4.0).

---

## 📖 About FailureSensorIQ
**FailureSensorIQ** is a Multi-Choice Question-Answering (MCQA) benchmarking system derived from official ISO standards. It contains **8,296 questions** across **10 different industrial assets**.

### Key Question Types
1. **Failure Mode to Sensors (FM2Sensor)**: Answers "What sensors should be monitored to detect a specific failure early?"
2. **Sensor to Failure Mode (Sensor2FM)**: Answers "What potential failure is predicted given anomalous behavior on specific sensors?"

### Covered Assets
*   Electric Motor
*   Steam Turbine
*   Aero Gas Turbine
*   Industrial Gas Turbine
*   Pump
*   Compressor
*   Reciprocating IC Engine
*   Electric Generator
*   Fan
*   Power Transformer

---

## 🛠️ Installation & Setup

1. **Python Dependencies**
   Make sure you have python (3.10+ recommended) and the required packages installed:
   ```bash
   pip install datasets huggingface_hub scikit-learn transformers accelerate
   ```

2. **Authenticate with Hugging Face**
   Authenticate to ensure access to the dataset repository:
   ```bash
   huggingface-cli login
   # Paste your user access token from huggingface.co/settings/tokens
   ```

3. **Download the Datasets**
   We have provided a automated python downloader script [download_failuresensoriq.py](file:///Users/josh23/Desktop/projects/ET/download_failuresensoriq.py). Run it from the `ET/` directory:
   ```bash
   python download_failuresensoriq.py
   ```
   This will download the standard, multi-choice, and perturbed subsets and save them as formatted `.json` files inside:
   *   `ET/data/eval/`

---

## 📊 Dataset Subsets & Files
Once downloaded, the files inside [data/eval](file:///Users/josh23/Desktop/projects/ET/data/eval) are:

| Local File | Description | Records |
| :--- | :--- | :--- |
| **`failuresensoriq_single.json`** | Single-correct standard MCQA dataset (Config: `single_true_multi_choice_qa`) | 2,667 |
| **`failuresensoriq_multi.json`** | Multi-correct MCQA dataset (Config: `multi_true_multi_choice_qa`) | 5,629 |
| **`failuresensoriq_standard_all.json`** | Full standard single-correct dataset | 2,667 |
| **`failuresensoriq_standard_all_multi_answers.json`** | Full standard multi-correct dataset | 5,629 |
| **`failuresensoriq_standard_all_10_options.json`** | MCQA questions expanded to 10 options (reduces guessing likelihood) | 2,667 |
| **`failuresensoriq_standard_sample_50.json`** | Subset of 50 questions for quick evaluation debugging | 50 |
| **`failuresensoriq_perturbed_simple.json`** | Simple Perturbations (shuffled options, modified option labels P, Q, R...) | 2,667 |
| **`failuresensoriq_perturbed_complex.json`** | Complex Perturbations (reordered options + LLM-rephrased questions) | 2,667 |
| **`failuresensoriq_perturbed_10_options_simple.json`** | Simple Perturbations applied to the 10-option dataset | 2,667 |
| **`failuresensoriq_perturbed_10_options_complex.json`** | Complex Perturbations applied to the 10-option dataset | 2,667 |

---

## 🔍 Perturbation-Uncertainty-Complexity (PUC) Framework
FailureSensorIQ evaluates LLMs across three dimensions of robustness:
1. **Perturbation Robustness**: Compares performance on standard questions vs. `SimplePert` and `ComplexPert` datasets to measure fragility under question rephrasing and formatting changes.
2. **Uncertainty Calibration**: Prompts LLMs to output probability distributions over options, using conformal prediction scores calculated on a calibration split to generate set-valued predictions.
3. **Complexity Ambiguity**: Gauges LLM capabilities under higher option density (e.g., 10 options) to minimize guessing artifacts.

---

## 🤖 LLMFeatureSelector (scikit-learn Pipeline)
The benchmark also includes `LLMFeatureSelector`—a custom scikit-learn feature selector estimator. Instead of purely statistical criteria (like ANOVA or correlation), it queries an LLM to rank and select features based on domain-specific understanding of sensor-failure relationships.

### Implementation Reference
You can use the following scikit-learn compatible estimator pattern based on IBM's pipeline code:

```python
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from accelerate import Accelerator

class LLMFeatureSelector(BaseEstimator):
    def __init__(self, model_name, feature_names, target_variable, prompt_template=None, topk=None):
        self.model_name = model_name
        self.feature_names = feature_names
        self.prompt_template = prompt_template
        self.target_variable = target_variable
        self.topk = topk
        if not self.prompt_template:
            self.prompt_template = (
                "Select the variables from the list that are most relevant for predicting <target_variable>. "
                "Provide the variables sorted starting with the one with the highest priority. "
                "All variables: <all_variables>\n"
                '```json\n{"selected_variables": ["variable 1", "variable 2", ..., "variable n"]}\n```'
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generation_config = transformers.GenerationConfig(max_length=512, stop_strings=['}'])
        
    def fit(self, X, y=None):
        X = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_all_finite="allow-nan",
        )
        prompt = self.prompt_template.replace('<all_variables>', ', '.join(self.feature_names))
        prompt = prompt.replace('<target_variable>', self.target_variable)
        
        # Format the chat prompt
        messages = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': '{"selected_variables": ["'}
        ]
        
        accelerator = Accelerator()
        tokens = self.tokenizer.apply_chat_template(messages, continue_final_message=True, return_tensors='pt').to(accelerator.device)
        output = self.model.generate(tokens, generation_config=self.generation_config, tokenizer=self.tokenizer)
        output_text = self.tokenizer.decode(output[0][tokens.shape[1]:], skip_special_tokens=True)
        output_text = '{"selected_variables": ["' + output_text
        
        try:
            parsed = eval(output_text)
        except Exception as e:
            print(f"Failed to parse LLM output: {e}")
            return X
            
        valid_idxs = []
        for col in parsed.get('selected_variables', [])[:self.topk]:
            if col in self.feature_names:
                valid_idxs.append(self.feature_names.index(col))
                
        return X[:, valid_idxs]

    def transform(self, X):
        # Implementation depends on stored indices after fit
        pass
```

---

## 📝 Citation & Paper Details
To cite the dataset and research findings:

**Dataset & Benchmark Paper (NeurIPS 2025 Datasets & Benchmarks):**
```bibtex
@inproceedings{
  constantinides2025failuresensoriq,
  title={FailureSensor{IQ}: A Multi-Choice {QA} Dataset for Understanding Sensor Relationships and Failure Modes},
  author={Christodoulos Constantinides and Dhaval C Patel and Shuxin Lin and Claudio Guerrero and SUNIL DAGAJIRAO PATIL and Jayant Kalagnanam},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=9KfkMAy2ut}
}
```

*   **Paper Link:** [arXiv:2506.03278](https://arxiv.org/abs/2506.03278)
*   **Official Code Repository:** [IBM/FailureSensorIQ GitHub](https://github.com/IBM/FailureSensorIQ)
*   **Leaderboard:** [Hugging Face Space](https://huggingface.co/spaces/cc4718/FailureSensorIQ) or [Kaggle Benchmark](https://www.kaggle.com/benchmarks/ibm-research/asset-ops-bench/)

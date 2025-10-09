# 🧮 V-Session: Structured Reasoning Framework for Mathematical Problem Solving

This repository provides the implementation, evaluation scripts, and fine-tuning configuration for **V-Session**,  
a structured reasoning framework designed to enhance mathematical reasoning in large language models (LLMs).

---

## 📂 Repository Structure

```bash
├── data/                # Datasets used for experiments
│   ├── GSM8K.json/           # GSM8K dataset for elementary-level math reasoning
│   └── MATH.jsonl/         # MATH500 dataset for advanced mathematical reasoning
│
├── prompt/              # Few-shot prompts for different reasoning styles 
│
├── log/                 # Experiment log files for analysis and reproducibility
│   ├── Qwen2.5-3B_V-Session_GSM8K1000.log
│   └── Qwen2.5-3B_V-Session_MATH500.log
│
├── eval/                # Evaluation utilities
│   └── compute_accuracy.py  # Script for calculating accuracy from log files
│
├── test/                # Testing and experiment scripts
│   ├── math_eval.py         # Core script for running reasoning evaluations
│   └── run_math_eval.sh     # Shell script for parameterized experiments
│
├── fine-tuning/         # Fine-tuning configuration and scripts
│   └── run.sh   # Training parameters for LLaMA-Factory
│
└── README.md
```
🚀 Overview

The V-Session framework introduces a five-stage reasoning process that improves the structure,
interpretability, and numerical stability of LLM-based mathematical reasoning.

This repository supports:

Evaluation of reasoning styles (Base, CoT, ToT, PoT, PAS, and V-Session)

Accuracy reproduction on GSM8K and MATH500

Fine-tuning with the LLaMA-Factory framework

📊 Evaluation

To compute accuracy from experiment logs, run:

cd eval
python compute_accuracy.py

The script automatically scans ./log/ for .log files and reports:

Total number of samples

Number of acc:True entries

Overall accuracy rate

🧠 Datasets

GSM8K — Grade-school math word problems (8-shot setting)

MATH500 — Advanced-level math benchmark (5-shot setting)

Datasets should be placed in the following structure:

⚙️ Fine-tuning

Fine-tuning scripts and configurations are available in fine-tuning/.
Training is conducted using the open-source LLaMA-Factory
 framework.

Example configuration:

Batch size: 72

Learning rate: 8e-6

Epochs: 3

Warmup ratio: 0.08

See train_config.json for detailed settings.

🤝 Acknowledgements

This work builds upon the open-source LLaMA-Factory(https://github.com/hiyouga/LLaMA-Factory) framework.
We also acknowledge the use of GSM8K and MATH500 datasets for benchmarking mathematical reasoning.

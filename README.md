Experiment
The experiment is conducted on GSM8K. Our experiments use the LLAMA and Qwen series of models as base models.


dependencies
Depends on the tool LLama-Factory for training and fine-tuning.

git clone https://github.com/hiyouga/LLaMA-Factory.git

Train the model
clone llama
The training models are all open source models

Train
change the corresponding settings in run.sh.

sh run.sh

evaluate
run EvaluateReasoningStyle.ipynb or EvaluateQwenStyle.ipynb or evaluate_gsm8k_V_session.py

some notes
The instructions above should guide you to reproduce our results exactly.

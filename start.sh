#!/bin/bash
pip install -r requirements.txt

# Preload the model before starting FastAPI
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM;
AutoTokenizer.from_pretrained('distilgpt2');
AutoModelForCausalLM.from_pretrained('distilgpt2');
print('Model downloaded successfully.')
"
uvicorn scene_enhancer:app --host 0.0.0.0 --port ${PORT:-8000}
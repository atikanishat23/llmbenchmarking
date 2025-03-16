#pip install torch transformers bitsandbytes

import torch
import time
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a small pre-trained LLM with quantization
model_name = "facebook/opt-1.3b"  # Change to a smaller model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32, device_map="cpu")

# Define a sample prompt
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# Measure inference time
start_time = time.time()
output = model.generate(**inputs, max_new_tokens=50)
inference_time = time.time() - start_time

# Calculate throughput (tokens per second)
tokens_generated = output.shape[1] - inputs.input_ids.shape[1]
throughput = tokens_generated / inference_time if inference_time > 0 else 0

# Measure memory usage
process = psutil.Process()
memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Print results
print("Inference Time:", inference_time, "seconds")
print("Throughput:", throughput, "tokens/sec")
print("Memory Usage:", memory_usage, "MB")

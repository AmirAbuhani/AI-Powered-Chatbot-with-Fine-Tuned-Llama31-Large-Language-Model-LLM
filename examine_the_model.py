
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:56:49 2024

@author: AmirAbuhani
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)

model_name = "C:\\codes\\llama31\\Meta-Llama-3.1-8B-Instruct"

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fb4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
bnb_4bit_use_double_quant = True

# Load the entire model on the GPU 0
device_map = {"": 0}  # Ensure this is correctly defined

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.config.use_cache = False
model.config.pretraining_tp = 1  # Ensure this is correct for your use case

# Define the input sentence
input_sentence = "hey there i have had cold 'symptoms' for over a week and had a low grade fever last week. for the past two days i have been feeling dizzy. should i contact my dr? should i see a dr"
# Tokenize the input
inputs = tokenizer(input_sentence, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=90, num_return_sequences=1)
    

# Decode the output
output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {input_sentence}")
print(f"Output: {output_sentence}")






# print(f"Model device: {model.device}")
# print(f"Input tensor device: {inputs['input_ids'].device}")








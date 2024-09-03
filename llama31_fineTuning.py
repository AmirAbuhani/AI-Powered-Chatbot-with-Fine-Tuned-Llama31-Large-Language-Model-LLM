# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:56:49 2024

@author: Amir Abu Hani
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
#from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Model and dataset paths
model_name = "C:\\codes\\llama31\\Meta-Llama-3.1-8B-Instruct"
dataset_name = "knowrohit07/know_medical_dialogues"
new_model = "llama31-finetuned-model"


###############################################################################

# QLoRA parameters
###############################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layares
lora_dropout = 0.1

# BitsAndBytes parameters
###############################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type(fb4 or nf4)
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = True

# Training arguments
# Output directory where the model predections and checkpoints will be stored
output_dir = "./results"
# Number of training epochs
num_train_epochs = 5
# Enable fb16/bf16 training
fb16 = True
bfl16 = False
# Batch size per GPU for training
per_device_train_batch_size = 1
# Batch size per GPU for evaluation
per_device_eval_batch_size = 1
max_grad_norm = 0.3
# Number of updates steps to accumulate the gradients
gradient_accumulation_steps = 1
# Enable gradient checkpoint
gradient_checkpointing = True
# Intial learning rate(adamW optimizer)
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weight
weight_decay = 0.01
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine"
# Maximum number of training steps (set to -1 for unlimited)
max_steps = -1
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds uo training considerbly
group_by_length = True
# Save chexkpoint every X updates steps
save_steps = 0
logging_steps = 25
# Maximum sequence length for the inputs
max_seq_length = 128
# Pack multiple short examples in the same input sequene to increase efficiency
packing = False
# Load the entire model on the GPU 0
device_map = {"": 0}

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Split the dataset into train, validation, and test sets
split_dataset = dataset.train_test_split(test_size=0.2)
train_test_valid = split_dataset["train"].train_test_split(test_size=0.25)

train_dataset = train_test_valid["train"]
val_dataset = train_test_valid["test"]
test_dataset = split_dataset["test"]

# Load tokenizer and model with QLoRA configuration
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

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load Llama tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Tokenize and preprocess dataset for training
# Defines a function that will process examples from the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["instruction"], truncation=True, padding="max_length", max_length=max_seq_length)
    outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=max_seq_length)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Attach LoRA adapters to the quantized model
model = PeftModel(model, peft_config)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fb16,
    bf16=bfl16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Use the SFTTrainer for supervised fine-tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)


# Train the model
trainer.train()


trainer.model.save_pretrained(new_model)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)


# -----------------------------------------------
# Evaluation and Visualization on Validation/Test Datasets
# -----------------------------------------------


# Function to decode and generate predictions
def generate_predictions(dataset, max_new_tokens=50):
    predictions = []
    true_labels = []
    for example in dataset:
        inputs = tokenizer(example['instruction'], return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)
        input_ids = inputs['input_ids'].to(model.device)
        
        # Generate predictions with max_new_tokens
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        predictions.append(pred)
        true_labels.append(example['output'])
        
    return predictions, true_labels

# Get predictions and true labels for validation and test datasets
val_predictions, val_labels = generate_predictions(tokenized_val)
test_predictions, test_labels = generate_predictions(tokenized_test)


# Cosine Similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize the sentence transformer model
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Calculating Similarity Between a Prediction and a Reference:
def calculate_similarity(prediction, reference):
    pred_embedding = similarity_model.encode(prediction, convert_to_tensor=True)
    ref_embedding = similarity_model.encode(reference, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(pred_embedding, ref_embedding)
    return similarity.item()
# Evaluating the Entire Dataset
def evaluate_dataset(predictions, references):
    similarities = []
    for pred, ref in zip(predictions, references):
        similarity = calculate_similarity(pred, ref)
        similarities.append(similarity)
    avg_similarity = np.mean(similarities)
    return avg_similarity

val_similarity = evaluate_dataset(val_predictions, val_labels)
test_similarity = evaluate_dataset(test_predictions, test_labels)

print(f"Validation Average Similarity Score: {val_similarity}")
print(f"Test Average Similarity Score: {test_similarity}")


# Cosine Similarity
# Semantic Similarity: Cosine Similarity focuses on the meaning of the text.
# Even if two answers use different words, they can still be considered similar if they convey the same meaning.
# This makes Cosine Similarity a powerful metric for evaluating generated text against reference text, especially in tasks where the exact wording is less important than the conveyed message.

#Exact Match Accuracy
# Exact match accuracy requires the generated text to be identical to the reference text.
# This is often too strict for natural language generation tasks where exact wording can vary, 
# even if the meaning is similar.


# -----------------------------------------------
# Visualization of Cosine Similarity Scores
# -----------------------------------------------

# Plotting distribution of cosine similarity scores for validation and test datasets
def plot_similarity_distribution(val_similarities, test_similarities):
    plt.figure(figsize=(12, 6))

    # Plot validation similarities
    sns.histplot(val_similarities, kde=True, color="blue", label="Validation Similarity")
    
    # Plot test similarities
    sns.histplot(test_similarities, kde=True, color="green", label="Test Similarity")
    
    plt.title("Distribution of Cosine Similarity Scores")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    plt.savefig('plot_similarity_distribution.png')

# Plotting average similarity scores for validation and test datasets
def plot_average_similarity(val_similarity, test_similarity):
    plt.figure(figsize=(8, 6))
    
    # Bar plot
    plt.bar(["Validation", "Test"], [val_similarity, test_similarity], color=["blue", "green"])
    
    plt.title("Average Cosine Similarity Scores")
    plt.ylabel("Average Cosine Similarity")
    plt.ylim(0, 1)  # Similarity ranges from 0 to 1
    plt.show()
    plt.savefig('plot_average_similarity.png')

# Compute similarities for visualization
val_similarities = [calculate_similarity(pred, ref) for pred, ref in zip(val_predictions, val_labels)]
test_similarities = [calculate_similarity(pred, ref) for pred, ref in zip(test_predictions, test_labels)]

# Plot the similarity distribution
plot_similarity_distribution(val_similarities, test_similarities)

# Plot the average similarity scores
plot_average_similarity(val_similarity, test_similarity)







# Define the text generation pipeline
text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

# Define a function to generate text from instructions
def generate_text_from_instruction(instruction):
    # Adding special tokens to help the model understand the task
    prompt = "[INSTRUCTION] " + instruction + " [ANSWER]"
    result = text_generator(prompt, max_length=200, num_return_sequences=1)
    # Extract the answer part from the generated text
    generated_text = result[0]['generated_text']
    answer_start = generated_text.find("[ANSWER]") + len("[ANSWER]")
    answer = generated_text[answer_start:].strip()
    return answer

# Example instructions from your dataset
example_instructions = [
    "i have a flu that is getting worse and i have returned from the uk at the end of feb and i have been in direct contact with british and brazilian tourists in sa since then. should i get tested for covid-19 or just remain home with my family?"
    # Add more example instructions based on your dataset
]

# Generate and print text for example instructions
for instruction in example_instructions:
    generated_text = generate_text_from_instruction(instruction)
    print(f"Instruction: {instruction}\nGenerated Text (Answer): {generated_text}\n")


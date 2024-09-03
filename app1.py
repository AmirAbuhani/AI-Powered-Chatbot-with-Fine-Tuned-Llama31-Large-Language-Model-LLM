from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from torch.cuda import is_available as is_gpu_available

# Initialize FastAPI app
app = FastAPI()

# Load Model and Tokenizer
model_name = "C:\\codes\\llama31\\Meta-Llama-3.1-8B-Instruct"
fine_tuned_model_dir = "C:\\codes\\llama31\\Meta-Llama-3.1-8B-Instruct\\final_project\\llama31-finetuned-model"

# BitsAndBytesConfig setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load the base model with bnb_config
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)

# Load the fine-tuned model with LoRA
model = PeftModel.from_pretrained(base_model, fine_tuned_model_dir)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Initialize the text generation pipeline
text_generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200
)

# Define the request body
class Instruction(BaseModel):
    instruction: str

# Endpoint for generating text based on instruction
@app.post("/generate-text/")
async def generate_text(instruction: Instruction):
    try:
        # Format the prompt
        prompt = f"[INSTRUCTION] {instruction.instruction} [ANSWER]"
        
        # Generate the text
        result = text_generator(prompt, max_length=200, num_return_sequences=1)
        
        # Extract the generated text
        generated_text = result[0]['generated_text']
        answer_start = generated_text.find("[ANSWER]") + len("[ANSWER]")
        answer = generated_text[answer_start:].strip()
        
        return {"instruction": instruction.instruction, "generated_text": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    device_name = torch.cuda.get_device_name(0) if is_gpu_available() else "CPU"
    return {"status": "ok", "gpu_available": is_gpu_available(), "device": device_name}

# Unsloth Fine-Tuning Pipeline

This directory contains scripts for fine-tuning Llama3 models using Unsloth's FastLanguageModel with your specified LoRA configuration and combined datasets.

## Files Created

1. **`unsloth_fine_tuning_pipeline.py`** - Main pipeline using Unsloth's FastLanguageModel
2. **`train_unsloth_model.py`** - Complete training script with actual model training
3. **`requirements-unsloth.txt`** - Dependencies for Unsloth fine-tuning

## Key Configuration

The pipeline uses your specified parameters:

### LoRA Configuration
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
)
```

### SFTTrainer Configuration
```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        report_to="comet_ml",
        seed=0,
    ),
)
trainer.train()
```

### Alpaca Template Formatting
```python
alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:

{}
### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token
dataset = dataset.map(format_samples, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.05)
```

### Inference and Model Saving
```python
# Enable inference mode
FastLanguageModel.for_inference(model)

# Test inference
message = alpaca_prompt.format("Write a paragraph to introduce supervised fine-tuning.", "")
inputs = tokenizer([message], return_tensors="pt").to("cuda")
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True)

# Save merged model
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# Push to hub
model.push_to_hub_merged("mlabonne/TwinLlama-3.1-8B", tokenizer, save_method="merged_16bit")
```

### Combined Datasets
```python
dataset1 = load_dataset("mlabonne/llmtwin")
dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
dataset = concatenate_datasets([dataset1, dataset2])
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements-unsloth.txt
```

2. For Apple Silicon (M1/M2/M3), install PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio
```

3. For NVIDIA GPUs, install CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Option 1: Combined Datasets (Recommended)
Use the combined Hugging Face datasets with optional local data:

```bash
# Just combined datasets (no local data)
python scripts/unsloth_fine_tuning_pipeline.py combined

# Combined datasets + your local articles
python scripts/unsloth_fine_tuning_pipeline.py combined data/articles.json
```

### Option 2: Local Data Only
Use only your local JSON data:

```bash
python scripts/unsloth_fine_tuning_pipeline.py local data/articles.json
```

### Option 3: Complete Training
Actually train the model:

```bash
# Train with combined datasets
python scripts/train_unsloth_model.py combined

# Train with combined + local data
python scripts/train_unsloth_model.py combined data/articles.json my_trained_model

# Train with local data only
python scripts/train_unsloth_model.py local data/articles.json my_trained_model
```

### Option 4: Test Inference
Test your trained model:

```bash
# Test with default instruction
python scripts/test_inference.py llama3_lora_model

# Test with custom instruction
python scripts/test_inference.py llama3_lora_model "Explain quantum computing"

# Interactive chat mode
python scripts/test_inference.py llama3_lora_model --interactive
```

### Option 5: Save and Push Models
Save merged models and push to hub:

```bash
# Data preparation with automatic inference test and push to hub
python scripts/unsloth_fine_tuning_pipeline.py combined --push

# Push to custom repository
python scripts/unsloth_fine_tuning_pipeline.py combined --push your-username/your-model
```

## Model Configuration

The pipeline uses these default settings:
- **Model**: `meta-llama/Meta-Llama-3.1-8B`
- **Max Sequence Length**: `2048`
- **Quantization**: `False` (LoRA, not QLoRA)
- **LoRA Rank**: `32`
- **LoRA Alpha**: `32`
- **LoRA Dropout**: `0`
- **Target Modules**: `["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"]`
- **Template**: Alpaca instruction-response format
- **Test Split**: `5%` (0.05) for evaluation
- **Batch Processing**: `batched=True` for efficient formatting

### SFTTrainer Parameters
- **Learning Rate**: `3e-4`
- **Batch Size**: `2` (with gradient accumulation of `8`)
- **Epochs**: `3`
- **Warmup Steps**: `10`
- **Optimizer**: `adamw_8bit`
- **Scheduler**: `linear`
- **Data Processing**: `packing=True`, `dataset_num_proc=2`
- **Precision**: Automatic `fp16`/`bf16` detection
- **Logging**: `comet_ml` integration

## Dataset Sources

When using `combined` mode:
- **mlabonne/llmtwin**: Full dataset
- **mlabonne/FineTome-Alpaca-100k**: First 10,000 samples
- **Optional**: Your local JSON data (if provided)

## Output

The scripts will create:
- `{model_name}_lora_model/` - LoRA model files
- `{model_name}_lora_model_gguf/` - GGUF model for Ollama
- Training logs and metrics

## Using the Trained Model

After training, you can use your model:

```python
from unsloth import FastLanguageModel

# Load your trained model
model, tokenizer = FastLanguageModel.from_pretrained("my_trained_model")

# Generate text
messages = [{"role": "user", "content": "Your prompt here"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## For Ollama Integration

If you want to use the trained model with Ollama:
1. Use the GGUF model created in `{output_dir}_gguf/`
2. Create a new Ollama model: `ollama create my-llama3-twin -f Modelfile`
3. Run: `ollama run my-llama3-twin`

## Memory Requirements

- **LoRA (default)**: ~16GB VRAM for 8B model
- **QLoRA**: ~8GB VRAM for 8B model (set `load_in_4bit=True`)
- **CPU training**: Slower but possible with sufficient RAM

## Examples

```bash
# Quick test with combined datasets
python scripts/unsloth_fine_tuning_pipeline.py combined

# Full training with your data
python scripts/train_unsloth_model.py combined data/articles.json my_llm_twin

# Local data only training
python scripts/train_unsloth_model.py local data/articles.json local_model
```

## Next Steps

1. Try the combined datasets first to test your setup
2. Adjust training parameters in `train_unsloth_model.py` as needed
3. Increase `max_steps` for better training results
4. Experiment with different `max_seq_length` values
5. Try QLoRA if you have VRAM constraints

# DPO Fine-Tuning Guide for LLM Twin

This guide explains how to use the DPO (Direct Preference Optimization) fine-tuning script to create a preference-aligned version of your LLM twin model.

## Overview

DPO fine-tuning uses preference data (chosen vs rejected responses) to align the model better with desired behaviors. This pipeline:

1. Loads your pre-trained LLM twin model
2. Applies LoRA PEFT for efficient training
3. Uses preference dataset created from your instruction pairs
4. Trains with DPO to improve response quality
5. Saves the fine-tuned model for inference

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements-dpo.txt
```

### 2. Create Preference Dataset

Before running DPO training, you need to create a preference dataset:

```bash
# Create preference dataset from instruction pairs
python scripts/create_preference_dataset.py data/instruction_pairs.json
```

This will generate `data/preference_dataset.json` with preference triples.

### 3. GPU Requirements

- Recommended: GPU with 16GB+ VRAM
- Minimum: GPU with 8GB VRAM (use `load_in_4bit=True`)

## Usage

### Basic Usage

```bash
# Run complete DPO pipeline
python scripts/dpo_fine_tuning.py
```

### Custom Configuration

```python
from scripts.dpo_fine_tuning import DPOTrainerPipeline

# Initialize with custom paths
pipeline = DPOTrainerPipeline(
    model_path="path/to/your/model",  # Your fine-tuned LLM twin
    preference_dataset_path="data/preference_dataset.json"
)

# Run pipeline
model_path = pipeline.run_pipeline()
```

## Configuration Options

### Model Parameters

- `max_seq_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Enable QLoRA for memory efficiency (default: False)
- `r`: LoRA rank (default: 64, increased for DPO)
- `lora_alpha`: LoRA alpha parameter (default: 64)

### Training Parameters

- `per_device_train_batch_size`: Batch size per device (default: 2)
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `learning_rate`: Learning rate (default: 5e-5)
- `num_train_epochs`: Number of training epochs (default: 3)
- `beta`: DPO temperature parameter (default: 0.1)

## Data Format

The preference dataset should have this structure:

```json
{
  "preference_triples": [
    {
      "instruction": "What are your main areas of expertise?",
      "generated_answer": "I specialize in machine learning and software development...",
      "extracted_answer": "My expertise includes AI/ML, full-stack development..."
    }
  ]
}
```

## Output

The pipeline saves the DPO fine-tuned model to:

- `./dpo_llm_twin/` - Main model directory
- `./dpo_results/` - Training checkpoints and logs

## Testing the Model

After training, you can test the model:

```python
from scripts.dpo_fine_tuning import DPOTrainerPipeline

pipeline = DPOTrainerPipeline()
response = pipeline.test_model("./dpo_llm_twin", "What are your core skills?")
print(response)
```

## Monitoring Training

To monitor training progress:

1. **TensorBoard** (default):
   ```bash
   tensorboard --logdir ./dpo_results/runs
   ```

2. **Weights & Biases** (optional):
   - Set `report_to="wandb"` in training arguments
   - Configure your WANDB API key

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size`
   - Enable `load_in_4bit=True`
   - Reduce `max_seq_length`

2. **Preference Dataset Not Found**:
   - Run the preference dataset creation script first
   - Check file paths are correct

3. **Model Loading Issues**:
   - Verify model path is correct
   - Check internet connection for downloading models
   - Ensure sufficient disk space

### Performance Tips

1. **For Faster Training**:
   - Use GPU with more VRAM
   - Enable mixed precision (`fp16=True` or `bf16=True`)
   - Use gradient checkpointing

2. **For Better Quality**:
   - Increase `num_train_epochs`
   - Use larger preference dataset
   - Tune `beta` parameter

## Advanced Usage

### Custom DPO Configuration

```python
# Custom DPO config
dpo_config = DPOConfig(
    beta=0.2,  # Higher beta = more conservative updates
    max_length=4096,
    max_prompt_length=2048,
    max_target_length=2048,
)

# Custom training args
training_args = TrainingArguments(
    output_dir="./custom_dpo_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_train_epochs=5,
    warmup_ratio=0.1,
)
```

### Using Different Base Models

```python
# Use a different base model
pipeline = DPOTrainerPipeline(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    preference_dataset_path="data/preference_dataset.json"
)
```

## Next Steps

After DPO training:

1. **Evaluate the model** on test prompts
2. **Compare with original model** using side-by-side tests
3. **Deploy the model** for inference
4. **Create evaluation metrics** to measure improvement

## References

- [TRL DPO Documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [Unsloth Documentation](https://unsloth.ai/)
- [DPO Paper](https://arxiv.org/abs/2305.18290)

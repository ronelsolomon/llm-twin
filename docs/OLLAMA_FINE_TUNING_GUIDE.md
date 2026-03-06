# Ollama Fine-Tuning Pipeline Guide

This guide explains how to use the Ollama Fine-Tuning Pipeline to create a custom Llama3 model trained on your articles dataset.

## Overview

The `ollama_fine_tuning_pipeline.py` script replaces OpenAI with Ollama's Llama3:latest model for fine-tuning. It creates a personalized AI model trained on your content, articles, and repositories.

## Prerequisites

### 1. Install Ollama

First, install Ollama on your system:

```bash
# macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows (WSL)
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull Llama3 Model

```bash
ollama pull llama3:latest
```

### 3. Start Ollama Server

```bash
ollama serve
```

The server will run on `http://localhost:11434` by default.

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python scripts/ollama_fine_tuning_pipeline.py data/articles.json
```

### With Custom Model Name

```bash
python scripts/ollama_fine_tuning_pipeline.py data/articles.json llama3-ronel-custom
```

## Pipeline Stages

The fine-tuning pipeline consists of 5 stages:

### Stage 1: Ollama Connection Check
- Verifies Ollama is running and accessible
- Checks if Llama3:latest model is available

### Stage 2: Data Loading
- Loads articles from the specified JSON file
- Creates a Hugging Face Dataset with the following fields:
  - `id`: Article ID
  - `content`: Article content
  - `platform`: Source platform (GitHub, Medium, etc.)
  - `author_id`: Author's ID
  - `author_full_name`: Author's full name
  - `link`: Article link

### Stage 3: Training Example Generation
- Converts articles into training examples
- Creates varied example types:
  - **Summarization**: Summarize articles
  - **Question Answering**: Generate Q&A pairs
  - **Content Analysis**: Analyze themes and topics
  - **Title Generation**: Create engaging titles

### Stage 4: Model Creation
- Creates a `Modelfile` for Ollama
- Builds a fine-tuned model using the training examples
- Names the model (default: `llama3-ronel-twin`)

### Stage 5: Model Testing
- Tests the fine-tuned model with sample questions
- Verifies the model responds appropriately

## Data Format

The input JSON file should have the following structure:

```json
{
  "artifact_data": [
    {
      "id": "article_001",
      "content": "Full article content here...",
      "platform": "medium",
      "author_id": "ronel_solomon",
      "author_full_name": "Ronel Solomon",
      "link": "https://medium.com/@ronel/article-001"
    }
  ]
}
```

## Training Example Types

### 1. Summarization Examples
```
Instruction: Summarize the following medium article by Ronel Solomon:
Input: [Article content]
Output: [Generated summary]
```

### 2. Question Answering Examples
```
Instruction: What is 'Article Title' about?
Input: 
Output: [Summary-based answer]
```

### 3. Content Analysis Examples
```
Instruction: Analyze the key themes and topics in this medium content:
Input: [Article content]
Output: Key themes: Technology, Programming, AI. This content provides insights...
```

### 4. Title Generation Examples
```
Instruction: Generate an engaging title for this content:
Input: [Article content]
Output: [Generated title]
```

## Configuration

### OllamaConfig

You can customize the Ollama connection settings:

```python
from scripts.ollama_fine_tuning_pipeline import OllamaFineTuningPipeline, OllamaConfig

config = OllamaConfig(
    base_url="http://localhost:11434",  # Ollama server URL
    model="llama3:latest",              # Base model
    timeout=300                         # Request timeout in seconds
)

pipeline = OllamaFineTuningPipeline(config)
```

## API Usage

### Programmatic Usage

```python
from scripts.ollama_fine_tuning_pipeline import OllamaFineTuningPipeline

# Initialize pipeline
pipeline = OllamaFineTuningPipeline()

# Load data
dataset = pipeline.load_articles_from_json("data/articles.json")

# Generate training examples
examples = pipeline.generate_training_examples()

# Create fine-tuned model
success = pipeline.create_fine_tuned_model("my-custom-model")

# Test the model
pipeline.test_fine_tuned_model("my-custom-model")
```

## Using the Fine-Tuned Model

Once the model is created, you can use it with Ollama:

```bash
# Interactive chat
ollama run llama3-ronel-twin

# API usage
curl http://localhost:11434/api/generate \
  -d '{
    "model": "llama3-ronel-twin",
    "prompt": "What kind of content do you specialize in?",
    "stream": false
  }'
```

## Integration with Existing Pipeline

The fine-tuned model can be integrated with the existing inference pipeline by modifying the `_initialize_llm()` method in `scripts/inference_pipeline.py`:

```python
def _initialize_llm(self):
    """Initialize LLM client for answer generation."""
    # Try Ollama first
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            self.llm_client = "ollama"
            self.llm_model = "llama3-ronel-twin"
            logger.info("✅ Ollama fine-tuned model initialized")
            return
    except Exception as e:
        logger.warning(f"Failed to initialize Ollama: {e}")
    
    # Fallback to OpenAI or other options...
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama is running: `ollama serve`
   - Check if server is accessible: `curl http://localhost:11434/api/tags`

2. **Model Not Found**
   - Pull Llama3: `ollama pull llama3:latest`
   - Check available models: `ollama list`

3. **Memory Issues**
   - Reduce the number of training examples
   - Use a smaller base model if available

4. **Slow Training**
   - The process can take several minutes depending on data size
   - Monitor logs for progress

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## File Outputs

The pipeline generates several files:

- `Modelfile`: Ollama model configuration file
- `training_data.json`: All generated training examples
- Fine-tuned model: Stored in Ollama's model registry

## Best Practices

1. **Data Quality**: Ensure articles are clean and well-formatted
2. **Diverse Examples**: The pipeline automatically creates varied training examples
3. **Regular Updates**: Re-train the model when you have new content
4. **Testing**: Always test the fine-tuned model before deployment

## Performance Tips

1. **Batch Processing**: Process articles in batches for large datasets
2. **Example Selection**: Focus on high-quality content for training
3. **Model Size**: Llama3:latest requires significant RAM (8GB+ recommended)

## Next Steps

1. Integrate the fine-tuned model with your RAG pipeline
2. Create specialized models for different content types
3. Implement continuous learning with new articles
4. Add evaluation metrics to measure model performance

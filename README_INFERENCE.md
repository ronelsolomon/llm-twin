# LLM Twin Inference Pipeline

A comprehensive inference system for the DPO fine-tuned LLM Twin models, providing multiple interfaces for text generation including programmatic API, REST server, and interactive chat.

## 🚀 Features

- **Multiple Generation Modes**: Single, batch, and streaming generation
- **Flexible Configuration**: Customizable generation parameters
- **Performance Optimized**: Batch processing and efficient memory management
- **REST API Server**: FastAPI-based server for production use
- **Interactive Chat**: Command-line chat interface with context awareness
- **Health Monitoring**: Built-in health checks and performance metrics
- **Easy Integration**: Simple functions for quick usage

## 📦 Installation

The inference pipeline is included in the main LLM Twin project. Ensure you have the required dependencies:

```bash
pip install torch transformers fastapi uvicorn loguru
```

## 🎯 Quick Start

### Basic Usage

```python
from src.inference_pipeline import quick_generate

# Quick generation
response = quick_generate("What are your main areas of expertise?")
print(response)
```

### Advanced Usage

```python
from src.inference_pipeline import (
    LLMTwinInferencePipeline,
    InferenceConfig,
    GenerationRequest
)

# Create pipeline with custom config
config = InferenceConfig(
    temperature=0.7,
    max_new_tokens=512,
    device="auto"
)

pipeline = LLMTwinInferencePipeline(config)
pipeline.load_model()

# Generate response
request = GenerationRequest(
    prompt="Tell me about your experience with AI development.",
    temperature=0.8
)

response = pipeline.generate(request)
print(f"Response: {response.generated_text}")
print(f"Time: {response.generation_time:.2f}s")

pipeline.unload_model()
```

## 🛠️ Components

### 1. Inference Pipeline (`src/inference_pipeline.py`)

The core inference system with the following classes:

- **`LLMTwinInferencePipeline`**: Main pipeline class
- **`InferenceConfig`**: Configuration for generation parameters
- **`GenerationRequest`**: Request structure for single generation
- **`GenerationResponse`**: Response structure with metrics

#### Key Methods

```python
# Load model
pipeline.load_model()

# Single generation
response = pipeline.generate(request)

# Batch generation
responses = pipeline.generate_batch(requests)

# Streaming generation
async for token in pipeline.generate_stream(request):
    print(token, end="")

# Model information
info = pipeline.get_model_info()

# Health check
health = pipeline.health_check()

# Cleanup
pipeline.unload_model()
```

### 2. REST API Server (`tools/inference_server.py`)

FastAPI-based server for production deployment:

#### Start Server

```bash
python tools/inference_server.py
```

#### API Endpoints

- `GET /` - Server information
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /generate` - Single generation
- `POST /generate/batch` - Batch generation
- `GET /generate/stream` - Streaming generation
- `POST /reload` - Reload model
- `GET /stats` - Server statistics

#### Example API Usage

```python
import requests

# Single generation
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "What are your main areas of expertise?",
    "temperature": 0.7,
    "max_new_tokens": 256
})

result = response.json()
print(result["generated_text"])
```

### 3. Interactive Chat (`tools/interactive_chat.py`)

Command-line chat interface:

```bash
python tools/interactive_chat.py
```

Features:
- Context-aware conversations
- Chat history management
- Command system (/help, /clear, /history, /save)
- Real-time generation statistics

## ⚙️ Configuration

### InferenceConfig Parameters

```python
@dataclass
class InferenceConfig:
    model_path: str = "./dpo_llm_twin_merged"  # Path to model
    max_new_tokens: int = 512                   # Max tokens to generate
    temperature: float = 0.7                     # Sampling temperature
    top_p: float = 0.9                          # Top-p sampling
    top_k: int = 50                             # Top-k sampling
    do_sample: bool = True                       # Use sampling
    repetition_penalty: float = 1.1              # Repetition penalty
    device: str = "auto"                         # Device selection
    use_cache: bool = True                       # Use KV cache
    batch_size: int = 4                         # Batch processing size
```

### Device Selection

- `"auto"` - Automatically choose best available device
- `"cuda"` - NVIDIA GPU
- `"mps"` - Apple Metal Performance Shaders
- `"cpu"` - CPU (fallback)

## 📊 Examples

### 1. Basic Generation

```python
# See examples/inference_example.py
python examples/inference_example.py
```

### 2. Batch Processing

```python
requests = [
    GenerationRequest(prompt="What is your expertise?"),
    GenerationRequest(prompt="Describe your approach."),
    GenerationRequest(prompt="What technologies do you use?")
]

responses = pipeline.generate_batch(requests)
for response in responses:
    print(f"Prompt: {response.prompt}")
    print(f"Response: {response.generated_text}")
```

### 3. Streaming Generation

```python
import asyncio

async def stream_example():
    request = GenerationRequest(
        prompt="Tell me about your experience with AI.",
        max_new_tokens=300
    )
    
    async for token in pipeline.generate_stream(request):
        print(token, end="", flush=True)

asyncio.run(stream_example())
```

### 4. Performance Benchmarking

```python
# Benchmark with multiple prompts
test_prompts = [
    "What are your main areas of expertise?",
    "Describe your software development approach.",
    "What technologies do you work with?"
]

start_time = time.time()
total_tokens = 0

for prompt in test_prompts:
    request = GenerationRequest(prompt=prompt)
    response = pipeline.generate(request)
    total_tokens += response.token_count

total_time = time.time() - start_time
tokens_per_second = total_tokens / total_time

print(f"Generated {total_tokens} tokens in {total_time:.2f}s")
print(f"Speed: {tokens_per_second:.2f} tokens/second")
```

## 🔧 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "tools/inference_server.py"]
```

### Environment Variables

```bash
export MODEL_PATH="/path/to/model"
export INFERENCE_DEVICE="cuda"
export BATCH_SIZE=4
export LOG_LEVEL="INFO"
```

### Monitoring

The inference pipeline includes built-in monitoring:

```python
# Health check
health = pipeline.health_check()
print(f"Status: {health['status']}")
print(f"Memory usage: {health['memory_usage']}")

# Model info
info = pipeline.get_model_info()
print(f"Parameters: {info['num_parameters']:,}")
print(f"Device: {info['device']}")
```

## 📈 Performance Tips

1. **Batch Processing**: Use `generate_batch()` for multiple prompts
2. **Device Selection**: Use GPU when available for better performance
3. **Token Limits**: Set appropriate `max_new_tokens` to balance quality and speed
4. **Temperature**: Lower values (0.3-0.5) for focused responses, higher (0.8-1.0) for creativity
5. **Memory Management**: Call `unload_model()` when done to free memory

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**:
   ```
   FileNotFoundError: Model not found at: ./dpo_llm_twin_merged
   ```
   Solution: Ensure the model path is correct and model files exist

2. **CUDA Out of Memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   Solution: Reduce batch_size, use CPU, or get more GPU memory

3. **Slow Generation**:
   - Check if using GPU: `torch.cuda.is_available()`
   - Reduce `max_new_tokens` for testing
   - Use batch processing for multiple requests

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
pipeline = LLMTwinInferencePipeline(config)
pipeline.load_model()
```

## 🤝 Contributing

To contribute to the inference pipeline:

1. Add new features to `src/inference_pipeline.py`
2. Update examples in `examples/inference_example.py`
3. Add tests for new functionality
4. Update documentation

## 📄 License

This inference pipeline is part of the LLM Twin project. See the main project license for details.

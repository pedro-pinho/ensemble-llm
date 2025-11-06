# Model Guide for Ensemble LLM

## Recommended Models by RAM

### 16GB RAM Configuration
```yaml
Models: 3-4 small models
Total Memory: ~12GB for models + 4GB system

Recommended Setup:
- llama3.2:3b (3GB)
- phi3.5 (3.5GB)  
- gemma2:2b (2GB)
- qwen2.5:3b (3GB)
```

### 24GB RAM Configuration (Optimal)
```yaml
Models: 4-5 medium models
Total Memory: ~18GB for models + 6GB system

Recommended Setup:
- llama3.2:3b (3GB)
- phi3.5 (3.5GB)
- qwen2.5:7b (5GB)
- mistral:7b-instruct-q4_K_M (4.5GB)
- gemma2:2b (2GB)
```

### 32GB+ RAM Configuration
```yaml
Models: 3-4 large models or 6+ small models
Total Memory: ~28GB for models + 4GB system

Recommended Setup:
- mixtral:8x7b-instruct-q3_K_M (15GB)
- llama3.1:13b (8GB)
- qwen2.5:7b (5GB)
- phi3.5 (3.5GB)
```

## Model Specializations

### Best for Coding
- **qwen2.5:7b** - Excellent for code generation and debugging
- **deepseek-coder:6.7b** - Specialized for programming tasks
- **codellama:7b** - Meta's code-specific model

### Best for Creative Writing
- **mistral:7b** - Great for stories and creative content
- **llama3.2:3b** - Good balance of creativity and coherence
- **gemma2:9b** - Google's model with creative capabilities

### Best for Analysis & Reasoning
- **phi3.5** - Microsoft's model optimized for reasoning
- **qwen2.5:7b** - Strong analytical capabilities
- **llama3.1:8b** - Good for complex reasoning

### Best for Quick Responses
- **gemma2:2b** - Fastest, good for simple queries
- **llama3.2:1b** - Extremely fast, basic capabilities
- **phi3:mini** - Quick and reasonably capable

## Quantization Levels

### Understanding Quantization
Quantization reduces model size by using fewer bits for weights:

- **Q8_0**: 8-bit (best quality, largest size)
- **Q6_K**: 6-bit (excellent quality, moderately large)
- **Q5_K_M**: 5-bit medium (great balance)
- **Q4_K_M**: 4-bit medium (recommended - best size/quality ratio)
- **Q3_K_M**: 3-bit medium (smaller, slight quality loss)
- **Q2_K**: 2-bit (smallest, noticeable quality loss)

### Choosing Quantization
```bash
# For maximum quality (more RAM needed)
ollama pull model:q6_k

# Balanced (recommended)
ollama pull model:q4_k_m

# For limited RAM
ollama pull model:q3_k_m
```

## Performance Tips

### Memory Usage Formula
```
Total RAM needed = (Sum of model sizes) + 2GB (Ollama overhead) + 2GB (System)
```

### Optimal Configurations

**For Speed Priority:**
- Use 2-3 small models (2-3B parameters)
- Use Q3 or Q4 quantization
- Set lower token limits

**For Quality Priority:**
- Use 3-4 diverse models (mix of sizes)
- Use Q4 or Q5 quantization
- Include at least one specialized model

**For Balanced Performance:**
- Use 3 models: 1 small (fast), 1 medium (balanced), 1 specialized
- Use Q4_K_M quantization
- Enable web search for current events only

## Adding Custom Models

To add a new model to the ensemble:

1. Pull the model:
```bash
ollama pull newmodel:tag
```

2. Add to config.py:
```python
MODEL_CONFIGS['newmodel:tag'] = {
    'memory_gb': 5,
    'specialties': ['domain1', 'domain2'],
    'timeout': 30
}
```

3. Use in ensemble:
```bash
python -m ensemble_llm.main --models newmodel:tag llama3.2:3b "Query"
```

## Troubleshooting Model Issues

### Model Not Loading
```bash
# Check if model exists
ollama list

# Re-pull if corrupted
ollama rm model:tag
ollama pull model:tag
```

### Model Too Slow
- Use smaller quantization (Q3_K_M instead of Q5_K_M)
- Reduce token generation limit
- Use fewer models in ensemble

### Out of Memory
- Use Q2_K quantization for large models
- Reduce number of concurrent models
- Close other applications
- Set OLLAMA_MAX_LOADED_MODELS=2
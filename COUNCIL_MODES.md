# Council & Synthesis Modes - Quick Reference

## Overview

Your Ensemble LLM now supports three operational modes for how models interact and respond:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Standard** | Models work independently, unaware of each other | Fast, simple queries |
| **Council** | Models know they're in a council | Better role differentiation |
| **Council + Synthesis** | Models discuss, winner synthesizes | Best quality, comprehensive answers |

---

## Mode Configuration

Edit `ensemble_llm/config.py` and modify `COUNCIL_CONFIG`:

```python
COUNCIL_CONFIG = {
    "enabled": False,        # True = Council mode ON
    "synthesis_mode": True,  # True = Add synthesis step
    # ... other settings ...
}
```

### Configuration Options

| Setting | Values | Description |
|---------|--------|-------------|
| `enabled` | `True`/`False` | Enable council awareness |
| `synthesis_mode` | `True`/`False` | Enable final synthesis step |
| `system_prompt_template` | string | How models are briefed about council |
| `synthesis_prompt_template` | string | Instructions for synthesis |

---

## How Each Mode Works

### 1. Standard Mode (Default)
```python
COUNCIL_CONFIG = {
    "enabled": False,
    "synthesis_mode": False,
}
```

**Flow:**
```
User Question
    ↓
[Model A] [Model B] [Model C]  ← Query all models independently
    ↓         ↓         ↓
Response  Response  Response
    ↓
Voting Algorithm selects best
    ↓
Return winning response
```

**Pros:** Fast, simple
**Cons:** Models unaware of collaborative context

---

### 2. Council Mode
```python
COUNCIL_CONFIG = {
    "enabled": True,
    "synthesis_mode": False,
}
```

**Flow:**
```
User Question
    ↓
Enhanced prompts telling each model:
- "You are part of a council"
- "Your role: [specialty]"
- "Council members: [list]"
    ↓
[Model A] [Model B] [Model C]  ← Models provide perspectives
    ↓         ↓         ↓
Response  Response  Response
    ↓
Voting selects best
    ↓
Return winning response (may contain council meta-talk)
```

**Pros:** Better role awareness, more thorough responses
**Cons:** May include meta-discussion about being in council

**Example response:**
> "As a member of this council focusing on technical analysis, I would say that Python is great for beginners because... My fellow council members might emphasize different aspects..."

---

### 3. Council + Synthesis Mode (Recommended)
```python
COUNCIL_CONFIG = {
    "enabled": True,
    "synthesis_mode": True,
}
```

**Flow:**
```
User Question
    ↓
Enhanced prompts with council context
    ↓
[Model A] [Model B] [Model C]  ← Models discuss from perspectives
    ↓         ↓         ↓
Response  Response  Response
    ↓
Voting selects best model → Model B wins!
    ↓
Model B receives ALL responses:
"You are spokesperson. Here's what everyone said:
- Model A: [response]
- Model B: [response]
- Model C: [response]

Synthesize into one answer. No meta-discussion."
    ↓
Model B synthesizes unified answer
    ↓
Return clean, comprehensive response to user
```

**Pros:**
- Best insights from all models combined
- No meta-discussion in final answer
- More comprehensive coverage
- Clean, professional output

**Cons:**
- Slower (extra synthesis query)
- Uses more tokens

**Example response:**
> "Python is an excellent choice for beginners for several reasons: the syntax is clean and readable, there's extensive documentation and community support, and it's versatile across domains from web development to data science. Key advantages include..."

---

## Comparison Example

**Question:** "What is quantum computing?"

### Standard Mode Output:
> Quantum computing uses quantum bits or qubits which can exist in multiple states simultaneously through superposition. This allows quantum computers to process information differently than classical computers...

### Council Mode Output:
> As a council member specializing in technical explanations, I'll provide my perspective on quantum computing. My fellow models might focus on other aspects, but from my viewpoint, quantum computing leverages quantum mechanical phenomena...

### Council + Synthesis Output:
> Quantum computing represents a fundamentally different approach to computation. Unlike classical computers that use binary bits, quantum computers use qubits that leverage quantum mechanical phenomena including superposition and entanglement. This enables them to solve certain classes of problems exponentially faster than classical computers. Key applications include cryptography, drug discovery, and optimization problems. However, quantum computers face challenges such as maintaining quantum coherence and error correction...

Notice: Synthesis combines technical depth, applications, and challenges into one coherent answer without mentioning the council.

---

## When to Use Each Mode

### Use Standard Mode when:
- Speed is critical
- Simple, straightforward questions
- Testing or debugging
- Limited computational resources

### Use Council Mode when:
- Want models to leverage their specialties
- Complex topics benefiting from multiple angles
- Don't mind meta-discussion in responses
- Debugging to see how models think about council

### Use Council + Synthesis when:
- Quality is priority over speed
- Complex questions needing comprehensive answers
- Want clean, professional output
- Production use cases
- User-facing applications

---

## Performance Considerations

### Standard Mode:
- **Time:** Fastest (one round of queries + voting)
- **Queries:** N models = N API calls
- **Tokens:** Standard

### Council Mode:
- **Time:** Slightly slower (longer prompts)
- **Queries:** N models = N API calls
- **Tokens:** Higher (longer prompts with council context)

### Council + Synthesis:
- **Time:** Slowest (extra synthesis step)
- **Queries:** N models + 1 synthesis = (N+1) API calls
- **Tokens:** Highest (council context + synthesis with all responses)

**Typical timings (3 models):**
- Standard: 5-10 seconds
- Council: 6-11 seconds
- Council + Synthesis: 8-15 seconds

---

## Customization

### Customize Council Prompt

Edit `COUNCIL_CONFIG["system_prompt_template"]` in config.py:

```python
"system_prompt_template": """You are {model_name}, an expert AI.

You're collaborating with: {council_members}

Your specialty: {model_specialty}

Provide your best analysis. A consensus algorithm will select the
final answer from all perspectives.

Be thorough and accurate."""
```

Variables available:
- `{model_name}` - Current model's name
- `{total_models}` - Number of models in council
- `{model_specialty}` - Model's specialty from config
- `{council_members}` - Comma-separated list of all models

### Customize Synthesis Prompt

Edit `COUNCIL_CONFIG["synthesis_prompt_template"]` in config.py:

```python
"synthesis_prompt_template": """The question was: {question}

Here are perspectives from the council:
{all_responses}

Your task: Create one comprehensive, unified answer.
- Combine the best insights
- Resolve any contradictions
- Write as a single authoritative response
- DO NOT mention the council process

Final answer:"""
```

---

## Testing & Debugging

### Test the synthesis feature:
```bash
python examples/synthesis_demo.py
```

### Compare modes:
```bash
# The demo script includes comparisons
python examples/synthesis_demo.py
```

### Check if synthesis is working:
```python
response, metadata = await ensemble.ensemble_query(question, verbose=True)

print(f"Synthesized: {metadata.get('synthesized', False)}")
print(f"Spokesperson: {metadata.get('synthesis_model', 'N/A')}")
```

---

## Troubleshooting

### Synthesis not working?

**Check 1:** Is it enabled?
```python
# In config.py
COUNCIL_CONFIG["enabled"] = True
COUNCIL_CONFIG["synthesis_mode"] = True
```

**Check 2:** Check metadata
```python
if not metadata.get("synthesized"):
    print("Synthesis was skipped or failed")
    # Check logs for errors
```

**Check 3:** Not in turbo mode?
Synthesis is disabled in turbo mode for performance.

### Synthesis returning empty?

- Check model timeout settings
- Check if model has enough context window (needs 4096+)
- Check logs for errors
- Fallback to original response is automatic

### Response still has council meta-talk?

- Synthesis may not be enabled
- Check `metadata["synthesized"]` - should be `True`
- Try adjusting synthesis prompt template

---

## Best Practices

1. **Production:** Use Council + Synthesis for user-facing apps
2. **Development:** Use Standard mode for fast iteration
3. **Testing:** Use Council mode to see individual model perspectives
4. **Custom prompts:** Tailor to your domain/use case
5. **Monitor:** Check `metadata["synthesized"]` to ensure synthesis is working
6. **Fallback:** System automatically falls back to original response if synthesis fails

---

## Examples

See `examples/` directory:
- `council_mode_demo.py` - Basic council mode demonstration
- `synthesis_demo.py` - Full synthesis demonstration with comparisons
- `iterative_council.py` - Advanced multi-round discussions

Run any example:
```bash
python examples/synthesis_demo.py
```

---

## Summary

**Simple flow:**
```
Standard:   Models → Vote → Winner
Council:    Models(aware) → Vote → Winner
Synthesis:  Models(aware) → Vote → Winner synthesizes all → Final
```

**Enable synthesis in 2 steps:**
1. Edit `ensemble_llm/config.py`
2. Set `COUNCIL_CONFIG["enabled"] = True` and `synthesis_mode = True`

**Result:** Better, more comprehensive answers with no meta-discussion! 
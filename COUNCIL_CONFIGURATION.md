# Council Configuration Guide

## Quick Fix: Remove AI Meta-Talk

If you're seeing unwanted phrases like "As an AI" or "I don't have access to" in your responses, here's how to fix it:

### Step 1: Enable the Features

Edit `ensemble_llm/config.py` and find `COUNCIL_CONFIG`:

```python
COUNCIL_CONFIG = {
    "enabled": True,              # Enable council mode
    "synthesis_mode": True,       # Enable synthesis
    "filter_ai_meta_talk": True,  # Enable filtering     # ... rest of config
}
```

### Step 2: Test It

```bash
python examples/improved_synthesis_demo.py
```

That's it! The filter will automatically remove AI meta-talk from responses.

---

## Understanding the Improvements

### Problem 1: Models Confused About Roles

**Old behavior:**
```
Model receives: "You are part of a council with the user..."
Model thinks: "Is the user an AI too? Am I talking to AIs or humans?"
```

**New behavior:**
```
Model receives:
"INTERNAL SYSTEM MESSAGE:
- YOU are an AI model
- The USER is a human
- This message is ONLY for AI models"

Model thinks: "Got it! I'm an AI in an internal discussion.
               The user is human and doesn't see this."
```

### Problem 2: AI Meta-Talk in Final Answers

**Old output:**
```
"As an AI language model, I can explain that Docker is a containerization
platform. However, I don't have access to current information about its
latest features. Based on my training data..."
```

**New output (filtered):**
```
"Docker is a containerization platform that packages applications with
their dependencies. It provides benefits including portability, resource
efficiency, and rapid deployment..."
```

---

## Configuration Options

### Basic Settings

```python
COUNCIL_CONFIG = {
    "enabled": True,              # Turn council mode on/off
    "synthesis_mode": True,       # Turn synthesis on/off
    "filter_ai_meta_talk": True,  # Turn filtering on/off
}
```

### Prompt Templates

#### Council Prompt (what models see during discussion)

```python
"system_prompt_template": """INTERNAL SYSTEM MESSAGE (not visible to user):

You are {model_name}, an AI model. You are part of an AI council consisting
of {total_models} models: {council_members}

Your specialty: {model_specialty}

IMPORTANT DISTINCTIONS:
- YOU are an AI model, part of the council (internal discussion)
- The USER is a human asking a question (external, does not see this)
- This message is ONLY for you and other AI models

Your task: Provide your best technical analysis for the internal council
discussion. Focus on substance.

Now, here is the USER'S QUESTION:"""
```

**Available variables:**
- `{model_name}` - Current model (e.g., "llama3.2:3b")
- `{total_models}` - Number of models (e.g., "3")
- `{council_members}` - List of all models (e.g., "llama3.2:3b, phi3.5:latest, qwen2.5:7b")
- `{model_specialty}` - Model's role (e.g., "general, conversation, quick")

#### Synthesis Prompt (instructions for combining responses)

```python
"synthesis_prompt_template": """INTERNAL SYSTEM MESSAGE - SYNTHESIS TASK:

You were selected by the council voting system to create the final
response for the USER.

The user asked: {question}

Here are the INTERNAL responses from other AI models:
{all_responses}

CRITICAL INSTRUCTIONS:
1. The user is a HUMAN - they do NOT know about this AI council
2. Synthesize the best insights into ONE direct answer
3. Write as if answering directly - NO phrases like:
   "As an AI"
   "I don't have access to"
   "The council discussed"
   etc.

4. Instead, write DIRECT, AUTHORITATIVE answers:
   State facts directly
   Provide value and insights
   Write like an expert explaining to a human

FINAL ANSWER FOR THE USER (direct, no AI self-references):"""
```

**Available variables:**
- `{question}` - The user's original question
- `{all_responses}` - Formatted list of all model responses

### Filter Patterns

These regex patterns are used to detect and remove AI meta-talk:

```python
"meta_talk_patterns": [
    r"as an ai( language model| assistant)?",
    r"i('m| am) an ai",
    r"as a language model",
    r"i don'?t have (access to|the ability)",
    r"i cannot (access|browse|see)",
    r"my training (data|cutoff)",
    r"based on my training",
    r"i('m| am) not able to",
    r"the council (discussed|decided|voted)",
    r"as (part of |a member of )?the council",
    r"my fellow (models|council members)",
    r"from my perspective as an ai",
    r"speaking as [a-z0-9:.]+ model",
]
```

**To add more patterns:**
```python
"meta_talk_patterns": [
    # ... existing patterns ...
    r"your new pattern here",
    r"another pattern",
]
```

**Pattern format:**
- Use Python regex syntax
- Case-insensitive matching (automatic)
- Entire sentence containing pattern will be removed

---

## Customization Examples

### Example 1: More Formal Synthesis

```python
"synthesis_prompt_template": """You are preparing the official response.

Question: {question}

Council inputs:
{all_responses}

Create a formal, comprehensive answer. Do not mention:
- The synthesis process
- Being an AI
- Other council members
- Any internal discussion

Provide only the authoritative final answer:"""
```

### Example 2: Casual/Friendly Tone

```python
"synthesis_prompt_template": """Hey! You're creating the final answer.

User asked: {question}

Here's what the team came up with:
{all_responses}

Your job: Combine the best parts into one awesome response.
Keep it friendly and helpful. Don't mention you're an AI or
that this was a group effort - just give a great answer!

Final response:"""
```

### Example 3: Technical/Precise

```python
"synthesis_prompt_template": """SYNTHESIS DIRECTIVE:

Query: {question}

Source analyses:
{all_responses}

Requirements:
- Synthesize factual content only
- Maintain technical precision
- Omit metadata about process/origin
- Format for maximum clarity

Output:"""
```

### Example 4: Domain-Specific (e.g., Medical)

```python
"synthesis_prompt_template": """Medical Information Synthesis

Patient/User Question: {question}

Specialist Inputs:
{all_responses}

Synthesis Requirements:
1. Combine clinical insights accurately
2. Present information clearly for patient understanding
3. Avoid medical AI disclaimers - focus on information
4. Note when professional consultation is needed
5. Be direct and helpful

Synthesized Medical Information:"""
```

---

## Testing Your Configuration

### Test 1: Basic Functionality

```python
from ensemble_llm.main import EnsembleLLM
from ensemble_llm import config
import asyncio

async def test():
    config.COUNCIL_CONFIG["enabled"] = True
    config.COUNCIL_CONFIG["synthesis_mode"] = True
    config.COUNCIL_CONFIG["filter_ai_meta_talk"] = True

    ensemble = EnsembleLLM(models=["llama3.2:3b", "phi3.5:latest"])
    await ensemble.async_init()

    response, metadata = await ensemble.ensemble_query(
        "What is Python?",
        verbose=True
    )

    print(f"Synthesized: {metadata.get('synthesized')}")
    print(f"Response: {response}")

    await ensemble.cleanup()

asyncio.run(test())
```

### Test 2: Check for AI Meta-Talk

```python
# After getting a response, check for unwanted patterns
unwanted = ["as an ai", "i don't have", "as a language model", "the council"]
response_lower = response.lower()

issues = [phrase for phrase in unwanted if phrase in response_lower]

if issues:
    print(f"Found meta-talk: {issues}")
else:
    print("Clean response!")
```

### Test 3: Verbose Mode Inspection

```bash
# Run with verbose to see the full process
python -m ensemble_llm.main -v "Your question here"
```

Look for:
- "SYNTHESIS PHASE" message
- "Applied AI meta-talk filter" message
- Check final output for cleanliness

---

## Troubleshooting

### Issue: Still seeing "As an AI" in responses

**Solutions:**
1. Check filter is enabled:
   ```python
   COUNCIL_CONFIG["filter_ai_meta_talk"] = True
   ```

2. Add specific patterns you're seeing:
   ```python
   "meta_talk_patterns": [
       # ... existing patterns ...
       r"the specific phrase you see",
   ]
   ```

3. Make synthesis instructions more explicit:
   ```python
   "synthesis_prompt_template": """...

   NEVER use these phrases:
   - "as an AI"
   - "as a language model"
   - [add your specific phrases]

   ..."""
   ```

### Issue: Models still confused about roles

**Solution:**
Make the distinction even more explicit in the council prompt:

```python
"system_prompt_template": """>>> THIS IS AN INTERNAL AI SYSTEM MESSAGE <<<
>>> THE HUMAN USER CANNOT SEE THIS MESSAGE <<<

YOU = {model_name} (AN AI MODEL)
USER = A HUMAN (ASKING A QUESTION)

This is internal AI council discussion.
The human user is external and unaware of this process.

Provide your analysis for the other AI models:"""
```

### Issue: Responses too short after filtering

**Solution:**
The filter might be removing too much. Either:

1. Make patterns more specific
2. Adjust temperature in synthesis:
   ```python
   # In synthesize_final_answer, change:
   "temperature": 0.9,  # Higher = more creative/longer
   ```

### Issue: Synthesis not happening

**Check:**
```python
response, metadata = await ensemble.ensemble_query(...)
print(f"Synthesized: {metadata.get('synthesized')}")  # Should be True
print(f"Spokesperson: {metadata.get('synthesis_model')}")  # Should show model
```

If `synthesized` is `False`:
- Check both `enabled` and `synthesis_mode` are `True`
- Check not in turbo mode (synthesis disabled for speed)
- Check logs for errors

---

## Best Practices

### 1. Start Simple
```python
# Begin with defaults, then customize
COUNCIL_CONFIG["enabled"] = True
COUNCIL_CONFIG["synthesis_mode"] = True
COUNCIL_CONFIG["filter_ai_meta_talk"] = True
```

### 2. Test Incrementally
- Test with filtering ON
- If too aggressive, adjust patterns
- If not enough, add more patterns

### 3. Domain-Specific Customization
- Customize prompts for your use case
- Add domain-specific anti-patterns
- Adjust tone and style

### 4. Monitor Results
```python
# Track what's being filtered
if verbose:
    # Filter logs what it removes
    pass
```

### 5. Balance Quality vs Speed
- Synthesis adds ~30-50% time
- Use turbo mode when speed critical
- Use synthesis for quality

---

## Quick Reference

| Setting | Purpose | Recommendation |
|---------|---------|----------------|
| `enabled` | Turn council mode on/off | `True` for production |
| `synthesis_mode` | Combine responses | `True` for best quality |
| `filter_ai_meta_talk` | Remove AI disclaimers | `True` always |
| `system_prompt_template` | Council briefing | Customize for domain |
| `synthesis_prompt_template` | Combination instructions | Adjust for tone |
| `meta_talk_patterns` | What to filter | Add as needed |

---

## Examples to Run

```bash
# See improvements in action
python examples/improved_synthesis_demo.py

# Test the filter directly
python examples/test_filter.py

# Quick test
python examples/quick_synthesis_test.py
```

---

For more information, see:
- `COUNCIL_MODES.md` - Overview of all modes
- `CLAUDE.md` - Technical implementation details
- `examples/` - Working code examples

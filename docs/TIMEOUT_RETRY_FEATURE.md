# Timeout Retry Feature

## Overview

The timeout retry feature allows users to quickly retry failed queries with different speed modes when a timeout occurs. This improves user experience by eliminating the need to manually retype queries and providing intelligent speed mode suggestions.

## Features

### 1. Automatic Timeout Detection

When a timeout error occurs, the system automatically detects it and offers retry options:

```
❌ Error: Request timed out after 25 seconds.

💡 Timeout detected! Quick retry options:
   Current mode: balanced

   ⚡ Suggested: 'fast' mode
      (3 models, 15s timeout)

Retry with 'fast' mode? (y/n/other):
```

### 2. Smart Speed Mode Suggestions

The system suggests the next faster speed mode based on your current setting:

| Current Mode | Suggested Mode | Models | Timeout |
|--------------|----------------|--------|---------|
| quality | balanced | 4 | 25s |
| balanced | fast | 3 | 15s |
| fast | turbo | 2 | 10s |
| turbo | (already fastest) | 2 | 10s |

### 3. Manual Speed Mode Control

Users can manually change speed modes at any time:

#### Using the 'speed' Command

```
Q: speed

Current speed mode: balanced

Available speed modes:
   1. turbo   - Ultra-fast (2 models, 10s timeout)
   2. fast    - Fast (3 models, 15s timeout)
   3. balanced - Balanced (4 models, 25s timeout)
   4. quality  - Quality (5 models, 40s timeout)

Enter number or name: 2
✓ Speed mode changed to: fast
```

### 4. Retry Previous Query

Use the `retry` command to manually retry the last query:

```
Q: retry

Retrying last query: "What is quantum computing?"
Current speed mode: balanced

Options:
   1. Retry with current speed (balanced)
   2. Change speed mode first
   3. Cancel

Choice: 2

Available speed modes:
   1. turbo   - Ultra-fast (2 models, 10s timeout)
   2. fast    - Fast (3 models, 15s timeout)
   3. balanced - Balanced (4 models, 25s timeout)
   4. quality  - Quality (5 models, 40s timeout)

Enter number or name: turbo
✓ Speed mode changed to: turbo
```

## Usage

### Starting Interactive Mode

```bash
python -m ensemble_llm.main -i
```

or

```bash
python -m ensemble_llm.main --interactive
```

### Available Commands

- `exit` / `quit` - Exit the program
- `status` - Show model performance statistics
- `models` - List current active models
- **`speed`** - Change speed mode (NEW)
- **`retry`** - Retry last query with different speed (NEW)
- `help` - Show help message

### Speed Modes Explained

#### Turbo Mode
- **Models**: 2 fastest models
- **Timeout**: 10 seconds
- **Use case**: Quick answers, simple queries
- **Strategy**: Race (returns first response)

#### Fast Mode
- **Models**: 3 fast models
- **Timeout**: 15 seconds
- **Use case**: Good balance for most queries
- **Strategy**: Cascade (starts fast models first)

#### Balanced Mode (Default)
- **Models**: 4 models
- **Timeout**: 25 seconds
- **Use case**: Standard queries needing accuracy
- **Strategy**: Parallel (all models run together)

#### Quality Mode
- **Models**: 5 models
- **Timeout**: 40 seconds
- **Use case**: Complex queries requiring comprehensive answers
- **Strategy**: Parallel with full ensemble voting

## Example Workflow

### Scenario: Timeout While Processing Complex Query

```
Q: Explain the complete history of quantum mechanics with all major discoveries

Processing query...

❌ Error: Request timed out after 25 seconds.

💡 Timeout detected! Quick retry options:
   Current mode: balanced

   ⚡ Suggested: 'fast' mode
      (3 models, 15s timeout)

Retry with 'fast' mode? (y/n/other): y
✓ Switched to fast mode. Retrying...

Answer (via llama3.2:3b):
----------------------------------------
Quantum mechanics is a fundamental theory in physics...
[Response continues]
----------------------------------------
Response time: 12.3s
```

### Scenario: Manual Speed Change

```
Q: What is 2+2?

Answer (via gemma2:2b):
----------------------------------------
4
----------------------------------------
Response time: 1.2s

Q: speed
Current speed mode: balanced

Enter number or name: turbo
✓ Speed mode changed to: turbo

Q: What is 5+5?

Answer (via gemma2:2b):
----------------------------------------
10
----------------------------------------
Response time: 0.8s
```

## Implementation Details

### Code Location

The retry feature is implemented in `ensemble_llm/main.py` (lines 1842-2058):

1. **Last Query Tracking**: Lines 1854-1856
2. **Speed Command**: Lines 1896-1919
3. **Retry Command**: Lines 1921-1963
4. **Automatic Timeout Detection**: Lines 2000-2058

### Key Variables

- `last_query`: Stores the most recent user query
- `last_error`: Stores the most recent error message
- `ensemble.speed_mode`: Current speed mode setting
- `ensemble.speed_profile`: Speed profile configuration

### Timeout Detection Logic

```python
is_timeout = "timeout" in error_lower or "timed out" in error_lower
```

The system checks if the error message contains "timeout" or "timed out" keywords.

### Speed Mode Progression

```python
speed_order = ["quality", "balanced", "fast", "turbo"]
current_idx = speed_order.index(ensemble.speed_mode)
faster_mode = speed_order[current_idx + 1]  # Next faster mode
```

## Configuration

Speed profiles are defined in `ensemble_llm/config.py`:

```python
SPEED_PROFILES = {
    "turbo": {
        "max_models": 2,
        "timeout": 10,
        "num_predict": 200,
        "temperature": 0.5,
        "strategy": "race",
    },
    "fast": {
        "max_models": 3,
        "timeout": 15,
        "num_predict": 300,
        "temperature": 0.6,
        "strategy": "cascade",
    },
    "balanced": {
        "max_models": 4,
        "timeout": 25,
        "num_predict": 512,
        "temperature": 0.7,
        "strategy": "parallel",
    },
    "quality": {
        "max_models": 5,
        "timeout": 40,
        "num_predict": 768,
        "temperature": 0.8,
        "strategy": "parallel",
    },
}
```

You can customize these values to adjust behavior.

## Tips

1. **Start with balanced mode** for most queries (default)
2. **Use turbo mode** for simple factual queries
3. **Use quality mode** for complex analysis or creative writing
4. **Switch to fast/turbo** if you experience frequent timeouts
5. **Use retry command** to experiment with different speed modes on the same query

## Troubleshooting

### Still Getting Timeouts on Turbo Mode

- Check if Ollama service is running: `ollama list`
- Check system resources (CPU/GPU/RAM)
- Try using smaller models
- Consider increasing timeout values in config.py

### Speed Command Not Working

Make sure you're in interactive mode:
```bash
python -m ensemble_llm.main -i
```

### Retry Doesn't Remember My Query

The retry feature only works in interactive mode and tracks the last successfully submitted query. If an error occurred before query submission, there's nothing to retry.

## Future Enhancements

Potential future improvements:

1. **Auto-adjust speed mode**: Automatically downgrade speed on timeout
2. **Query complexity detection**: Suggest speed mode based on query type
3. **Timeout prediction**: Estimate required time before starting
4. **Per-model timeouts**: Allow individual model timeout settings
5. **Retry history**: Track and replay multiple previous queries

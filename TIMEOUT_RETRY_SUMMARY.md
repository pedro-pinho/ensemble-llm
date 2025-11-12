# Timeout Retry Feature - Implementation Summary

## Overview

A new intelligent timeout retry system has been added to the interactive mode, allowing users to quickly retry failed queries with different speed modes without having to retype their questions.

## What Was Added

### 1. New Interactive Commands

#### `speed` Command
Change speed mode at any time during your session:
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

#### `retry` Command
Manually retry the last query with optional speed mode change:
```
Q: retry

Retrying last query: "Explain quantum computing"
Current speed mode: balanced

Options:
   1. Retry with current speed (balanced)
   2. Change speed mode first
   3. Cancel

Choice:
```

### 2. Automatic Timeout Detection and Retry

When a timeout occurs, the system automatically:
1. Detects the timeout error
2. Shows the current speed mode
3. Suggests the next faster mode
4. Offers instant retry

Example:
```
❌ Error: Request timed out after 25 seconds.

💡 Timeout detected! Quick retry options:
   Current mode: balanced

   ⚡ Suggested: 'fast' mode
      (3 models, 15s timeout)

Retry with 'fast' mode? (y/n/other): y
✓ Switched to fast mode. Retrying...
```

### 3. Query History Tracking

The system now automatically tracks:
- `last_query`: Your most recent question
- `last_error`: The most recent error (if any)

This enables retry functionality without requiring you to retype.

## How It Works

### Speed Mode Hierarchy

The system suggests faster modes in this order:

```
quality (5 models, 40s)
    ↓ timeout? suggest:
balanced (4 models, 25s)
    ↓ timeout? suggest:
fast (3 models, 15s)
    ↓ timeout? suggest:
turbo (2 models, 10s)
    ↓ already fastest!
```

### Retry Flow

```
User Query → Timeout Error
      ↓
Detect Timeout
      ↓
Suggest Faster Mode
      ↓
User Accepts (y)
      ↓
Switch Mode & Auto-Retry
      ↓
Success or Show Error
```

## Usage Examples

### Example 1: Automatic Timeout Retry

**Before (without feature):**
```
Q: Write a detailed analysis of quantum mechanics

❌ Error: Request timed out after 25 seconds.

Q: [User has to retype entire question]
```

**After (with feature):**
```
Q: Write a detailed analysis of quantum mechanics

❌ Error: Request timed out after 25 seconds.

💡 Timeout detected! Quick retry options:
   Current mode: balanced
   ⚡ Suggested: 'fast' mode (3 models, 15s timeout)

Retry with 'fast' mode? (y/n/other): y
✓ Switched to fast mode. Retrying...

[Answer appears]
```

### Example 2: Manual Speed Control

```
Q: speed
Enter number or name: turbo
✓ Speed mode changed to: turbo

Q: What is 2+2?
[Fast response]

Q: speed
Enter number or name: quality
✓ Speed mode changed to: quality

Q: Explain the theory of relativity
[Comprehensive response from 5 models]
```

### Example 3: Manual Retry

```
Q: Explain machine learning

[Gets answer]

Q: retry

Options:
   1. Retry with current speed (balanced)
   2. Change speed mode first
   3. Cancel

Choice: 2
Enter number or name: quality

[Gets more comprehensive answer with 5 models]
```

## Implementation Details

### Modified Files

1. **ensemble_llm/main.py** (lines 1842-2058)
   - Added `last_query` and `last_error` tracking
   - Added `speed` command handler
   - Added `retry` command handler
   - Added automatic timeout detection and retry logic
   - Updated help text with new commands

### Key Code Sections

**Query Tracking:**
```python
# Store query for potential retry
last_query = prompt
last_error = None
```

**Timeout Detection:**
```python
is_timeout = "timeout" in error_lower or "timed out" in error_lower
```

**Speed Mode Suggestion:**
```python
speed_order = ["quality", "balanced", "fast", "turbo"]
current_idx = speed_order.index(ensemble.speed_mode)
faster_mode = speed_order[current_idx + 1]
```

**Automatic Retry:**
```python
if retry_now == "y":
    ensemble.speed_mode = faster_mode
    ensemble.speed_profile = SPEED_PROFILES[faster_mode]
    response, metadata = await ensemble.ensemble_query(last_query, args.verbose)
```

## Benefits

### 1. Improved User Experience
- No need to retype queries after timeouts
- One-click retry with intelligent suggestions
- Smooth speed mode switching

### 2. Time Savings
- Instant retry without retyping (saves 10-30 seconds)
- Smart mode suggestions (no guessing)
- Quick speed adjustments

### 3. Better Error Recovery
- Automatic timeout detection
- Guided recovery path
- Multiple retry options

### 4. Flexibility
- Manual or automatic retry
- Change speed mode anytime
- Experiment with different modes easily

## Configuration

Speed profiles are configured in `config.py`:

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

You can adjust these values to customize behavior.

## Documentation

- **User Guide**: `docs/TIMEOUT_RETRY_FEATURE.md`
- **Demo Examples**: `examples/demo_timeout_retry.md`
- **Code Changes**: `ensemble_llm/main.py` (lines 1842-2058)

## Testing

To test the feature:

1. **Start interactive mode:**
   ```bash
   python -m ensemble_llm.main -i
   ```

2. **Test speed command:**
   ```
   Q: speed
   ```

3. **Test retry command:**
   ```
   Q: What is AI?
   Q: retry
   ```

4. **Test timeout retry:**
   - Use quality mode and ask a very complex question
   - Wait for timeout
   - Accept the automatic retry suggestion

## Future Enhancements

Potential improvements for future versions:

1. **Auto-mode switching**: Automatically downgrade speed on repeated timeouts
2. **Query complexity analysis**: Suggest initial speed mode based on query
3. **Retry history**: Track multiple previous queries for replay
4. **Custom timeout values**: Per-query timeout overrides
5. **Timeout prediction**: Estimate time before starting
6. **Smart caching**: Cache successful retries to avoid duplicates

## Summary

The timeout retry feature provides:

✅ **New Commands**: `speed`, `retry`
✅ **Automatic Timeout Detection**: Smart suggestions on failure
✅ **Query History**: Last query automatically saved
✅ **One-Click Retry**: No retyping needed
✅ **Flexible Speed Control**: Change modes anytime
✅ **Better UX**: Smoother error recovery

**Result**: Faster, more convenient interaction with the ensemble system!

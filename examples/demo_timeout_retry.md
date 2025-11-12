# Timeout Retry Feature Demo

## Quick Start

1. Start interactive mode:
```bash
python -m ensemble_llm.main -i
```

2. The system will show available commands including the new features:
```
Commands:
   'exit' or 'quit' - Exit the program
   'status' - Show model performance statistics
   'models' - List current active models
   'speed' - Change speed mode (turbo/fast/balanced/quality)
   'retry' - Retry last query with different speed
   'help' - Show this help message

Current speed mode: balanced
```

## Demo Scenarios

### Scenario 1: Automatic Timeout Retry

When a query times out, you'll see:

```
Q: Write a comprehensive essay about the history of artificial intelligence including all major breakthroughs from 1950 to 2025

Processing query...

❌ Error: Request timed out after 25 seconds.

💡 Timeout detected! Quick retry options:
   Current mode: balanced

   ⚡ Suggested: 'fast' mode
      (3 models, 15s timeout)

Retry with 'fast' mode? (y/n/other):
```

**Options:**
- Type `y` - Immediately retry with faster mode
- Type `n` - Don't retry now
- Type `other` - See manual retry options

### Scenario 2: Manual Speed Change

```
Q: speed

Current speed mode: balanced

Available speed modes:
   1. turbo   - Ultra-fast (2 models, 10s timeout)
   2. fast    - Fast (3 models, 15s timeout)
   3. balanced - Balanced (4 models, 25s timeout)
   4. quality  - Quality (5 models, 40s timeout)

Enter number or name: 1
✓ Speed mode changed to: turbo

Q: What is the capital of France?

Answer (via gemma2:2b):
----------------------------------------
Paris
----------------------------------------
Response time: 0.9s
```

### Scenario 3: Manual Retry with Speed Change

```
Q: Explain quantum entanglement in detail

Answer (via qwen2.5:7b):
----------------------------------------
[Long detailed answer about quantum entanglement...]
----------------------------------------
Response time: 18.4s

Q: retry

Retrying last query: "Explain quantum entanglement in detail"
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

Enter number or name: quality
✓ Speed mode changed to: quality

[Processes with 5 models for a more comprehensive answer...]
```

## Comparison of Speed Modes

### Speed vs Quality Tradeoff

| Mode | Speed | Quality | Best For |
|------|-------|---------|----------|
| Turbo | ⚡⚡⚡⚡ | ⭐⭐ | Quick facts, simple math |
| Fast | ⚡⚡⚡ | ⭐⭐⭐ | General queries, coding help |
| Balanced | ⚡⚡ | ⭐⭐⭐⭐ | Complex questions, analysis |
| Quality | ⚡ | ⭐⭐⭐⭐⭐ | Essays, creative writing |

### Example Response Times

Based on typical queries:

**Simple Query: "What is 2+2?"**
- Turbo: 0.5-1s
- Fast: 1-2s
- Balanced: 2-3s
- Quality: 3-5s

**Medium Query: "Explain how a car engine works"**
- Turbo: 3-5s
- Fast: 5-8s
- Balanced: 8-12s
- Quality: 12-18s

**Complex Query: "Write a detailed essay about climate change"**
- Turbo: May timeout
- Fast: 12-15s
- Balanced: 18-25s
- Quality: 25-35s

## Best Practices

### 1. Start with Balanced Mode
The default balanced mode works well for most queries.

### 2. Use Turbo for Rapid-Fire Questions
```
Q: speed
Enter number or name: turbo

Q: What is 5+5?
Q: Who wrote Hamlet?
Q: What year did WWII end?
```

### 3. Use Quality for Important Work
```
Q: speed
Enter number or name: quality

Q: Review this code and suggest improvements:
[Paste code here]
```

### 4. Retry Failed Queries Intelligently
If a query times out on balanced mode:
1. Try fast mode first
2. If still timeout, try turbo mode
3. If still failing, the query may be too complex - try simplifying it

### 5. Monitor Performance with 'status'
```
Q: status

Model Performance Statistics:
================================
llama3.2:3b
  Success Rate: 98.5%
  Avg Response Time: 3.2s
  Times Selected: 45

phi3.5:latest
  Success Rate: 96.2%
  Avg Response Time: 4.1s
  Times Selected: 38
...
```

## Interactive Commands Summary

| Command | Description | Example |
|---------|-------------|---------|
| `help` | Show all commands | `Q: help` |
| `status` | Model performance stats | `Q: status` |
| `models` | List active models | `Q: models` |
| `speed` | Change speed mode | `Q: speed` → `1` (turbo) |
| `retry` | Retry last query | `Q: retry` → `1` |
| `exit` | Quit program | `Q: exit` |

## Tips for Success

1. **Timeouts are normal** for very complex queries - use the retry feature!
2. **Speed mode persists** throughout your session
3. **Last query is always saved** - you can retry anytime
4. **Experiment with modes** to find what works best for your use case
5. **Use 'status' regularly** to see which models perform best

## Common Workflows

### Research Workflow
```
1. Start with balanced mode (default)
2. Ask initial question
3. If answer is good, continue with balanced
4. If too slow, switch to fast mode
5. Use retry to get faster responses on similar questions
```

### Quick Facts Workflow
```
1. Switch to turbo mode immediately
2. Rapid-fire multiple questions
3. Get quick, concise answers
4. Switch back to balanced for detailed questions
```

### Deep Analysis Workflow
```
1. Switch to quality mode
2. Ask complex, multi-part questions
3. Get comprehensive answers from 5 models
4. Use retry with different modes to compare quality
```

## Troubleshooting

### Q: Still getting timeouts on turbo?
**A**: Check if Ollama is running properly:
```bash
ollama list
ollama ps
```

### Q: Responses are low quality on turbo mode?
**A**: This is expected! Turbo mode prioritizes speed over quality. Switch to balanced or quality mode for better answers.

### Q: Want to change default speed mode?
**A**: Use the `--speed` flag when starting:
```bash
python -m ensemble_llm.main -i --speed fast
```

### Q: How do I see current speed mode?
**A**: Type `speed` command or check the startup message.

---

**Happy querying!** 🚀

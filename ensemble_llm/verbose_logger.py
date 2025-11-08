"""Verbose logging system for detailed tracking"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import textwrap
from collections import deque


class VerboseFileLogger:
    """Detailed file logging for ensemble operations"""

    def __init__(self, log_dir: str = "logs", max_response_length: int = 500):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.verbose_log_file = self.log_dir / f"ensemble_verbose_{timestamp}.log"
        self.session_log_file = self.log_dir / "ensemble_session.log"
        self.rotation_log_file = self.log_dir / "model_rotation.log"

        self.max_response_length = max_response_length

        # Setup formatters
        self.separator = "=" * 80
        self.sub_separator = "-" * 60

        # Write session header
        self._write_session_header()

    def _write_session_header(self):
        """Write session start header"""
        header = f"""
{self.separator}
ENSEMBLE LLM SESSION STARTED
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log File: {self.verbose_log_file}
{self.separator}
"""
        self._write_to_file(self.verbose_log_file, header)
        self._write_to_file(self.session_log_file, header)

    def _write_to_file(self, file_path: Path, content: str):
        """Write content to file"""
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
            f.flush()

    def _truncate_response(self, response: str) -> str:
        """Truncate long responses for logging"""
        if len(response) <= self.max_response_length:
            return response

        return f"{response[:self.max_response_length]}... [TRUNCATED - {len(response)} total chars]"

    def log_query_start(
        self,
        query_id: int,
        prompt: str,
        models: List[str],
        speed_mode: str = None,
        web_search: bool = False,
    ):
        """Log the start of a query"""

        content = f"""
{self.separator}
QUERY #{query_id} STARTED
Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}
Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}
Models: {', '.join(models)}
Speed Mode: {speed_mode or 'standard'}
Web Search: {'Yes' if web_search else 'No'}
{self.separator}
"""
        self._write_to_file(self.verbose_log_file, content)

    def log_model_response(self, model: str, response: Dict, query_time: float):
        """Log individual model response"""

        success_indicator = "SUCCESS" if response.get("success") else "[X] FAILED"

        content = f"""
{self.sub_separator}
MODEL: {model}
Status: {success_indicator}
Response Time: {response.get('response_time', query_time):.3f}s
"""

        if response.get("success"):
            truncated_response = self._truncate_response(response.get("response", ""))
            content += (
                f"Response Preview:\n{textwrap.indent(truncated_response, '  ')}\n"
            )

            # Add metadata if available
            if "metadata" in response:
                content += f"Metadata: {json.dumps(response['metadata'], indent=2)}\n"
        else:
            content += f"Error: {response.get('response', 'Unknown error')}\n"

        # Add cache info if present
        if response.get("from_cache"):
            content += f"Cache: HIT (similarity: {response.get('cache_similarity', 1.0):.2f})\n"

        content += self.sub_separator + "\n"

        self._write_to_file(self.verbose_log_file, content)

    def log_voting_details(self, voting_data: Dict):
        """Log detailed voting information"""

        content = f"""
{self.sub_separator}
VOTING PROCESS
{self.sub_separator}
Total Models: {voting_data.get('total_models', 0)}
Successful Models: {voting_data.get('successful_models', 0)}

CONSENSUS MATRIX:
"""

        # Log consensus scores
        if "all_scores" in voting_data:
            content += "\nModel Scores (sorted by final score):\n"
            sorted_scores = sorted(
                voting_data["all_scores"].items(),
                key=lambda x: x[1].get("final", 0),
                reverse=True,
            )

            for i, (model, scores) in enumerate(sorted_scores, 1):
                selected = (
                    "**SELECTED**" if model == voting_data.get("selected_model") else ""
                )
                content += f"""
  {i}. {model} {selected}
     - Consensus Score: {scores.get('consensus', 0):.3f}
     - Quality Score: {scores.get('quality', 0):.3f}
     - Final Score: {scores.get('final', 0):.3f}
     - Response Time: {scores.get('response_time', 0):.2f}s
     - Used Web Search: {scores.get('used_web', False)}
"""

        # Log winner
        content += f"""
{self.sub_separator}
WINNER: {voting_data.get('selected_model', 'None')}
Final Score: {voting_data.get('final_score', 0):.3f}
Total Query Time: {voting_data.get('total_ensemble_time', 0):.2f}s
{self.sub_separator}
"""

        self._write_to_file(self.verbose_log_file, content)

    def log_model_rotation(
        self,
        removed_models: List[str],
        added_models: List[str],
        reasons: Dict[str, str],
        performance_data: Dict = None,
    ):
        """Log model rotation events"""

        content = f"""
{self.separator}
MODEL ROTATION EVENT
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{self.separator}

REMOVED MODELS:
"""

        for model in removed_models:
            reason = reasons.get(model, "Unknown")
            content += f"  [X] {model}\n"
            content += f"    Reason: {reason}\n"

            if performance_data and model in performance_data:
                perf = performance_data[model]
                content += f"    Final Stats:\n"
                content += f"      - Success Rate: {perf.get('success_rate', 0):.1%}\n"
                content += f"      - Avg Response Time: {perf.get('avg_response_time', 0):.2f}s\n"
                content += (
                    f"      - Selection Rate: {perf.get('selection_rate', 0):.1%}\n"
                )
                content += f"      - Consecutive Failures: {perf.get('consecutive_failures', 0)}\n"

        content += f"\nADDED MODELS:\n"
        for model in added_models:
            content += f"  {model}\n"
            if performance_data and model in performance_data:
                perf = performance_data[model]
                content += f"    Previous Performance:\n"
                content += f"      - Success Rate: {perf.get('success_rate', 0):.1%}\n"
                content += f"      - Last Used: {perf.get('last_used', 'Never')}\n"

        content += f"{self.separator}\n"

        # Write to both rotation log and verbose log
        self._write_to_file(self.rotation_log_file, content)
        self._write_to_file(self.verbose_log_file, content)

    def log_cache_event(self, event_type: str, query: str, details: Dict = None):
        """Log cache-related events"""

        content = f"""
{self.sub_separator}
CACHE EVENT: {event_type}
Query: {query[:100]}{'...' if len(query) > 100 else ''}
"""

        if details:
            content += f"Details: {json.dumps(details, indent=2)}\n"

        content += self.sub_separator + "\n"

        self._write_to_file(self.verbose_log_file, content)

    def log_performance_update(self, model: str, metrics: Dict):
        """Log model performance updates"""

        content = f"""
{self.sub_separator}
PERFORMANCE UPDATE: {model}
New Metrics:
  - Success Rate: {metrics.get('success_rate', 0):.1%}
  - Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s
  - Quality Score: {metrics.get('quality_score', 0):.3f}
  - Status: {metrics.get('status', 'unknown')}
{self.sub_separator}
"""

        self._write_to_file(self.verbose_log_file, content)

    def log_error(self, error_type: str, error_msg: str, traceback: str = None):
        """Log errors with full traceback"""

        content = f"""
{self.sub_separator}
ERROR: {error_type}
Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}
Message: {error_msg}
"""

        if traceback:
            content += f"Traceback:\n{traceback}\n"

        content += self.sub_separator + "\n"

        self._write_to_file(self.verbose_log_file, content)

    def log_session_end(self, stats: Dict = None):
        """Log session end with statistics"""

        content = f"""
{self.separator}
SESSION ENDED
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        if stats:
            content += f"""
Session Statistics:
  - Total Queries: {stats.get('total_queries', 0)}
  - Cache Hits: {stats.get('cache_hits', 0)}
  - Avg Response Time: {stats.get('avg_response_time', 0):.2f}s
  - Model Rotations: {stats.get('rotations', 0)}
"""

        content += f"{self.separator}\n"

        self._write_to_file(self.verbose_log_file, content)
        self._write_to_file(self.session_log_file, content)


class ModelPerformanceLogger:
    """Specialized logger for model performance tracking"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.performance_log = self.log_dir / "model_performance_detailed.jsonl"
        self.daily_summary_log = (
            self.log_dir / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.json"
        )

        # In-memory buffer for performance data
        self.performance_buffer = deque(maxlen=1000)

    def log_model_query(
        self,
        model: str,
        success: bool,
        response_time: float,
        query_type: str = None,
        was_selected: bool = False,
        quality_score: float = 0,
        consensus_score: float = 0,
    ):
        """Log individual model query performance"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "success": success,
            "response_time": response_time,
            "query_type": query_type,
            "was_selected": was_selected,
            "quality_score": quality_score,
            "consensus_score": consensus_score,
        }

        # Add to buffer
        self.performance_buffer.append(entry)

        # Write to JSONL file
        with open(self.performance_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def generate_daily_summary(self):
        """Generate daily performance summary"""

        summary = {"date": datetime.now().strftime("%Y-%m-%d"), "models": {}}

        # Process performance log
        if self.performance_log.exists():
            with open(self.performance_log, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        model = entry["model"]

                        if model not in summary["models"]:
                            summary["models"][model] = {
                                "total_queries": 0,
                                "successful_queries": 0,
                                "total_response_time": 0,
                                "times_selected": 0,
                                "total_quality_score": 0,
                                "query_types": {},
                            }

                        model_stats = summary["models"][model]
                        model_stats["total_queries"] += 1

                        if entry["success"]:
                            model_stats["successful_queries"] += 1

                        model_stats["total_response_time"] += entry["response_time"]

                        if entry["was_selected"]:
                            model_stats["times_selected"] += 1

                        model_stats["total_quality_score"] += entry.get(
                            "quality_score", 0
                        )

                        # Track by query type
                        query_type = entry.get("query_type", "general")
                        if query_type not in model_stats["query_types"]:
                            model_stats["query_types"][query_type] = 0
                        model_stats["query_types"][query_type] += 1

                    except:
                        pass

        # Calculate averages
        for model, stats in summary["models"].items():
            if stats["total_queries"] > 0:
                stats["success_rate"] = (
                    stats["successful_queries"] / stats["total_queries"]
                )
                stats["avg_response_time"] = (
                    stats["total_response_time"] / stats["total_queries"]
                )
                stats["selection_rate"] = (
                    stats["times_selected"] / stats["total_queries"]
                )
                stats["avg_quality_score"] = (
                    stats["total_quality_score"] / stats["total_queries"]
                )

        # Save summary
        with open(self.daily_summary_log, "w") as f:
            json.dump(summary, f, indent=2)

        return summary


class LiveTailLogger:
    """Logger that can be tailed in real-time"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.live_log = self.log_dir / "ensemble_live.log"

    def log_live(self, message: str, level: str = "INFO"):
        """Log message for live tailing"""

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_msg = f"[{timestamp}] [{level:7}] {message}\n"

        with open(self.live_log, "a") as f:
            f.write(formatted_msg)
            f.flush()

    def clear_live_log(self):
        """Clear the live log file"""
        if self.live_log.exists():
            self.live_log.unlink()

#!/usr/bin/env python3
"""Log viewer utility for ensemble logs"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import subprocess


def tail_live_log(log_file: str = "logs/ensemble_live.log"):
    """Tail the live log file"""
    try:
        subprocess.run(["tail", "-f", log_file])
    except KeyboardInterrupt:
        print("\nStopped tailing log")


def view_verbose_log(date: str = None, lines: int = 100):
    """View verbose log for a specific date"""
    log_dir = Path("logs")

    if date:
        # Find log for specific date
        pattern = f"ensemble_verbose_{date}*.log"
    else:
        # Find most recent log
        pattern = "ensemble_verbose_*.log"

    logs = sorted(log_dir.glob(pattern), reverse=True)

    if not logs:
        print(f"No verbose logs found matching pattern: {pattern}")
        return

    log_file = logs[0]
    print(f"Viewing: {log_file}")
    print("=" * 80)

    with open(log_file, "r") as f:
        all_lines = f.readlines()

        if lines == -1:
            # Show all
            for line in all_lines:
                print(line, end="")
        else:
            # Show last N lines
            for line in all_lines[-lines:]:
                print(line, end="")


def view_rotation_log():
    """View model rotation events"""
    rotation_log = Path("logs/model_rotation.log")

    if not rotation_log.exists():
        print("No rotation log found")
        return

    print("Model Rotation History:")
    print("=" * 80)

    with open(rotation_log, "r") as f:
        print(f.read())


def view_performance_summary():
    """View daily performance summary"""
    log_dir = Path("logs")
    today = datetime.now().strftime("%Y%m%d")

    summary_file = log_dir / f"daily_summary_{today}.json"

    if not summary_file.exists():
        print(f"No summary found for today ({today})")
        return

    with open(summary_file, "r") as f:
        summary = json.load(f)

    print(f"Performance Summary for {summary['date']}")
    print("=" * 80)

    for model, stats in summary["models"].items():
        print(f"\n{model}:")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
        print(f"  Selection Rate: {stats.get('selection_rate', 0):.1%}")
        print(f"  Avg Response Time: {stats.get('avg_response_time', 0):.2f}s")
        print(f"  Avg Quality Score: {stats.get('avg_quality_score', 0):.3f}")

        if stats.get("query_types"):
            print(f"  Query Types: {stats['query_types']}")


def search_logs(keyword: str, log_type: str = "verbose"):
    """Search logs for specific keyword"""
    log_dir = Path("logs")

    if log_type == "verbose":
        pattern = "ensemble_verbose_*.log"
    elif log_type == "live":
        pattern = "ensemble_live.log"
    else:
        pattern = "*.log"

    print(f"Searching for '{keyword}' in {pattern}")
    print("=" * 80)

    for log_file in log_dir.glob(pattern):
        with open(log_file, "r") as f:
            for line_no, line in enumerate(f, 1):
                if keyword.lower() in line.lower():
                    print(f"{log_file.name}:{line_no}: {line.strip()}")


def main():
    parser = argparse.ArgumentParser(description="View ensemble logs")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Tail live log
    tail_parser = subparsers.add_parser("tail", help="Tail live log")

    # View verbose log
    verbose_parser = subparsers.add_parser("verbose", help="View verbose log")
    verbose_parser.add_argument("--date", help="Date (YYYYMMDD)")
    verbose_parser.add_argument(
        "--lines", type=int, default=100, help="Number of lines to show (-1 for all)"
    )

    # View rotations
    rotation_parser = subparsers.add_parser("rotations", help="View model rotations")

    # View performance
    perf_parser = subparsers.add_parser("performance", help="View performance summary")

    # Search logs
    search_parser = subparsers.add_parser("search", help="Search logs")
    search_parser.add_argument("keyword", help="Keyword to search")
    search_parser.add_argument(
        "--type",
        default="verbose",
        choices=["verbose", "live", "all"],
        help="Log type to search",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "tail":
        tail_live_log()
    elif args.command == "verbose":
        view_verbose_log(args.date, args.lines)
    elif args.command == "rotations":
        view_rotation_log()
    elif args.command == "performance":
        view_performance_summary()
    elif args.command == "search":
        search_logs(args.keyword, args.type)


if __name__ == "__main__":
    main()

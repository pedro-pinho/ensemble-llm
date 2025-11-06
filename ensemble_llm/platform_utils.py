"""Platform-specific utilities for cross-platform compatibility"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional, List, Dict


class PlatformUtils:
    """Utilities for handling platform-specific operations"""

    @staticmethod
    def get_platform():
        """Get the current platform"""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        elif system == "windows":
            return "windows"
        elif system == "linux":
            return "linux"
        else:
            return "unknown"

    @staticmethod
    def get_ollama_executable():
        """Get the Ollama executable path"""
        system = PlatformUtils.get_platform()

        if system == "windows":
            # Check common Windows paths
            possible_paths = [
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Programs"
                / "Ollama"
                / "ollama.exe",
                Path("C:/Program Files/Ollama/ollama.exe"),
                Path("C:/Program Files (x86)/Ollama/ollama.exe"),
                Path(os.environ.get("USERPROFILE", ""))
                / "AppData"
                / "Local"
                / "Programs"
                / "Ollama"
                / "ollama.exe",
            ]

            for path in possible_paths:
                if path.exists():
                    return str(path)

            # Try to find in PATH
            result = subprocess.run(
                ["where", "ollama"], capture_output=True, text=True, shell=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]

        else:
            # Unix-like systems
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()

        return "ollama"  # Fallback to hoping it's in PATH

    @staticmethod
    def kill_process(process_name: str):
        """Kill a process by name (cross-platform)"""
        system = PlatformUtils.get_platform()

        if system == "windows":
            subprocess.run(
                ["taskkill", "/F", "/IM", f"{process_name}.exe"],
                capture_output=True,
                shell=True,
            )
        else:
            subprocess.run(["killall", process_name], capture_output=True)

    @staticmethod
    def check_process_running(process_name: str) -> bool:
        """Check if a process is running"""
        system = PlatformUtils.get_platform()

        if system == "windows":
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {process_name}.exe"],
                capture_output=True,
                text=True,
                shell=True,
            )
            return process_name in result.stdout
        else:
            result = subprocess.run(["pgrep", "-x", process_name], capture_output=True)
            return result.returncode == 0

    @staticmethod
    def get_memory_usage(process_name: str) -> Optional[float]:
        """Get memory usage of a process in MB"""
        system = PlatformUtils.get_platform()

        if system == "windows":
            cmd = f'wmic process where name="{process_name}.exe" get WorkingSetSize'
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        try:
                            bytes_used = int(line.strip())
                            return bytes_used / (1024 * 1024)  # Convert to MB
                        except:
                            pass
        else:
            cmd = f"ps aux | grep {process_name} | awk '{{sum+=$6}} END {{print sum/1024}}'"
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

            if result.returncode == 0:
                try:
                    return float(result.stdout.strip())
                except:
                    pass

        return None

    @staticmethod
    def get_gpu_info() -> Dict:
        """Get GPU information"""
        gpu_info = {
            "available": False,
            "name": "Unknown",
            "memory_total": 0,
            "memory_free": 0,
            "cuda_available": False,
        }

        # Try NVIDIA SMI
        try:
            nvidia_smi = (
                "nvidia-smi.exe"
                if PlatformUtils.get_platform() == "windows"
                else "nvidia-smi"
            )

            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(", ")
                    if len(parts) >= 3:
                        gpu_info["available"] = True
                        gpu_info["name"] = parts[0]
                        gpu_info["memory_total"] = int(parts[1])
                        gpu_info["memory_free"] = int(parts[2])
        except:
            pass

        # Check CUDA availability
        try:
            import torch

            gpu_info["cuda_available"] = torch.cuda.is_available()
            if gpu_info["cuda_available"]:
                gpu_info["cuda_device_count"] = torch.cuda.device_count()
        except ImportError:
            pass

        return gpu_info

    @staticmethod
    def get_system_info() -> Dict:
        """Get system information"""
        import psutil

        return {
            "platform": PlatformUtils.get_platform(),
            "processor": platform.processor(),
            "cpu_cores": os.cpu_count(),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "python_version": sys.version,
            "gpu": PlatformUtils.get_gpu_info(),
        }


class WindowsOptimizer:
    """Windows-specific optimizations"""

    @staticmethod
    def set_process_priority(process_name: str, priority: str = "HIGH"):
        """Set process priority on Windows"""
        priorities = {
            "LOW": "IDLE",
            "BELOW_NORMAL": "BELOW_NORMAL",
            "NORMAL": "NORMAL",
            "ABOVE_NORMAL": "ABOVE_NORMAL",
            "HIGH": "HIGH",
            "REALTIME": "REALTIME",
        }

        priority_class = priorities.get(priority.upper(), "NORMAL")

        cmd = f'wmic process where name="{process_name}.exe" CALL setpriority "{priority_class}"'
        subprocess.run(cmd, shell=True, capture_output=True)

    @staticmethod
    def optimize_ollama_for_gpu():
        """Optimize Ollama settings for GPU on Windows"""
        env_vars = {
            "OLLAMA_NUM_GPU": "999",  # Use all GPU layers
            "CUDA_VISIBLE_DEVICES": "0",  # Use first GPU
            "OLLAMA_GPU_OVERHEAD": "0",  # Minimize overhead
            "OLLAMA_MAX_LOADED_MODELS": "4",  # More models with GPU
            "OLLAMA_KEEP_ALIVE": "10m",  # Keep models loaded longer
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        return env_vars

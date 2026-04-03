#!/usr/bin/env python3
"""Qwen2.5-Omni vLLM server startup script."""

import subprocess
import sys

def main():
    """Start vLLM server with Qwen2.5-Omni-7B model."""
    
    model_name = "Qwen/Qwen2.5-Omni-7B"
    
    cmd = [
        "vllm", "serve", model_name,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.85",
	"--enforce-eager", 
    ]
    
    print(f"Starting vLLM server with {model_name}...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

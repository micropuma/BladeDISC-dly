#!/usr/bin/env python3
import os
import subprocess
import sys
import importlib.util

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.decode().strip()}"

print("=" * 60)
print(f"🐍 Python version: {sys.version}")
print("=" * 60)

# -------- PyTorch --------
try:
    import torch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version (torch): {torch.version.cuda}")
        print(f"   cuDNN version (torch): {torch.backends.cudnn.version()}")
except ImportError:
    print("❌ PyTorch not installed")

# -------- CUDA (system) --------
print("\n⚡ Checking system CUDA:")
print(run_cmd("nvcc --version"))
print(run_cmd("ls -l /usr/local/cuda/lib64/libcudart.so* || true"))

# -------- NVIDIA driver --------
print("\n🧱 Checking NVIDIA driver:")
print(run_cmd("nvidia-smi | head -n 10"))

# -------- TensorRT --------
print("\n🔌 Checking TensorRT:")
if importlib.util.find_spec("tensorrt"):
    import tensorrt as trt
    print(f"TensorRT Python version: {trt.__version__}")
else:
    print("TensorRT Python module not found, checking .so files...")
    print(run_cmd("find /usr/local -name 'libnvinfer.so*' 2>/dev/null | head -n 5"))
    print(run_cmd("find /usr/lib -name 'libnvinfer.so*' 2>/dev/null | head -n 5"))

# -------- LD_LIBRARY_PATH --------
print("\n📦 LD_LIBRARY_PATH content:")
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
for p in ld_path.split(":"):
    print(f"  - {p}")
    if os.path.isdir(p):
        so_files = [f for f in os.listdir(p) if f.endswith(".so") or ".so." in f]
        for so in sorted(so_files):
            print(f"      {so}")

print("=" * 60)
print("✅ Environment check finished.")
print("=" * 60)


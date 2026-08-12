import sys
import torch


def get_device() -> torch.device:
    """
    Returns the optimal execution device following the priority:
    1. NVIDIA CUDA GPU ("cuda")
    2. Apple Silicon GPU ("mps")
    3. CPU ("cpu")
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def is_cuda() -> bool:
    """Returns True if execution device is CUDA."""
    return get_device().type == "cuda"


def is_mps() -> bool:
    """Returns True if execution device is Apple Silicon MPS."""
    return get_device().type == "mps"


def is_cpu() -> bool:
    """Returns True if execution device is CPU."""
    return get_device().type == "cpu"


def is_amp_available() -> bool:
    """Returns True if Automatic Mixed Precision (AMP) is natively supported (CUDA)."""
    return is_cuda()


def get_device_name() -> str:
    """Returns human-readable name of the current execution hardware."""
    device = get_device()
    if device.type == "cuda":
        return f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "Apple Silicon GPU (Metal Performance Shaders - MPS)"
    else:
        return "CPU Execution"


def print_device_info():
    """Prints a detailed hardware diagnostics banner for CUDA, MPS, or CPU."""
    device = get_device()
    print("====================================================")
    print("AIR-NET V1 HARDWARE & DEVICE DIAGNOSTICS")
    print("====================================================")
    print(f"Execution Device:              {device.type.upper()}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Name:                      {gpu_name}")
        print(f"CUDA Version:                  {cuda_ver}")
        print(f"Total GPU Memory:              {total_mem:.2f} GB")
        print(f"Mixed Precision (AMP):         Enabled (CUDA Native)")
    elif device.type == "mps":
        print(f"Apple GPU:                     Apple Silicon (MPS)")
        print(f"Memory Architecture:           Unified System Memory")
        print(f"Mixed Precision (AMP):         Disabled (FP32 Mode for MPS)")
    else:
        print(f"Hardware Acceleration:         CPU Mode")
        print(f"Mixed Precision (AMP):         Disabled (FP32 Mode for CPU)")
    print("====================================================")


def get_gpu_memory_info():
    """
    Returns (allocated_mb, reserved_mb, peak_mb) for CUDA.
    Returns (0.0, 0.0, 0.0) for MPS and CPU without throwing exceptions.
    """
    if is_cuda():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        res = torch.cuda.memory_reserved() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        return alloc, res, peak
    else:
        return 0.0, 0.0, 0.0

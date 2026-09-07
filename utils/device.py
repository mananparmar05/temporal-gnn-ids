"""
Device Management Module
Auto-detects the best available compute device (CUDA GPU, Apple MPS, or CPU)
and provides unified device selection for all model components.
"""

import torch
import platform


def get_device(prefer: str = "auto") -> torch.device:
    """
    Detect and return the optimal compute device.
    
    Args:
        prefer: Device preference.
            - "auto": Automatically select best available (CUDA > MPS > CPU)
            - "cuda": Force CUDA GPU (raises error if unavailable)
            - "mps": Force Apple Metal Performance Shaders
            - "cpu": Force CPU execution
    
    Returns:
        torch.device: Selected compute device.
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"[Device] Using CUDA GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"[Device] Using Apple MPS (Metal) on {platform.processor()}")
        else:
            device = torch.device("cpu")
            print(f"[Device] Using CPU ({platform.processor()})")
    elif prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA GPU detected.")
        device = torch.device("cuda")
    elif prefer == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but Apple Metal is not available.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU (explicitly requested)")
    
    return device


def get_device_summary() -> dict:
    """
    Returns a summary dictionary of all available compute devices.
    Useful for logging and experiment tracking.
    """
    summary = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        summary["cuda_device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        summary["cuda_total_memory_gb"] = round(props.total_mem / (1024**3), 2)
        summary["cuda_capability"] = f"{props.major}.{props.minor}"
    
    return summary

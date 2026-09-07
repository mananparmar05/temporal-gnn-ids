"""
Reproducibility Module
Sets all random seeds (Python, NumPy, PyTorch) for deterministic runs.
Ensures experimental results are fully reproducible across hardware.
"""

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.
    
    Args:
        seed: Integer seed value for all random number generators.
        deterministic: If True, enables PyTorch deterministic algorithms
                       and disables CUDA benchmark mode for exact reproducibility.
                       May slightly reduce training speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # PyTorch >= 1.8 deterministic flag
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
    
    print(f"[Seed] Global seed set to {seed} | Deterministic: {deterministic}")

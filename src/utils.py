"""
Common utility functions for the genre story generation project.

This module contains helper functions that are used across multiple
notebooks and scripts.
"""

import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    return config


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2h 34m 56s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    print("\nModel architecture:")
    print(model)
    
    print("="*60 + "\n")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint


def create_dir_structure(base_dir: str = "."):
    """
    Create standard project directory structure.
    
    Args:
        base_dir: Base directory for the project
    """
    base_path = Path(base_dir)
    
    directories = [
        "data",
        "src",
        "src/model",
        "notebooks",
        "experiments",
        "experiments/configs",
        "experiments/checkpoints",
        "experiments/logs",
        "gui",
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created:")
    for directory in directories:
        print(f"  ✓ {directory}/")

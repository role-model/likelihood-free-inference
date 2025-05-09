---
id: gpu-acceleration
title: GPU Acceleration
sidebar_position: 1
---

# GPU Acceleration

This section outlines how to enable and utilize GPU acceleration in the Simulation-Based Inference (SBI) pipeline. Leveraging CUDA-capable GPUs can significantly improve training performance, especially for large-scale simulations or high-dimensional parameter spaces.


## Requirements

To use GPU acceleration, your system must meet the following conditions:

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** installed (version compatible with PyTorch)
- **PyTorch with CUDA backend** installed

You can install PyTorch with CUDA 11.8 support as follows:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Using the GPU in SBI Workflows

PyTorch automatically places tensors and models on the CPU by default. To utilize a GPU, explicitly transfer relevant objects to the CUDA device:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Example: preparing input data
theta = theta.to(device)
x = x.to(device)
```

Most sbi components are compatible with GPU execution, provided that inputs and models are transferred appropriately.

## Training with GPU

When training the neural posterior estimator, GPU acceleration is automatically used if:
* Input tensors (```theta```, ```x```) are on the GPU
* The neural network model is on the GPU

This process is managed internally by the ```sbi``` API, but it's good practice to confirm that inputs and outputs remain on the same device.

```python
inference = SNPE(prior=prior)
density_estimator = inference.append_simulations(theta.to(device), x.to(device)).train()
```

## Performance Considerations

### Batch Size and Memory

* Increase batch size to better utilize GPU memory, if available.
* Monitor GPU usage with nvidia-smi.

### Data Movement

* Avoid frequent CPU–GPU transfers, which can bottleneck performance.
* Pre-load and pre-transform data on the GPU when possible.

### Parallelism

* Simulation may remain CPU-bound unless the simulator itself is parallelized.
* If training time dominates, GPU offers the most significant benefit.

## Verifying GPU Usage

To confirm that training uses the GPU:

1. Run nvidia-smi in the terminal during training.
2. Use PyTorch utilities to print device assignment:

```python
for param in density_estimator.parameters():
    print(param.device)
```

## Summary

GPU acceleration can dramatically reduce training time for neural density estimators in the SBI pipeline. Ensure all data and models are transferred to the CUDA device, and monitor resource usage to achieve optimal performance.

In the next section, we demonstrate a complete inference workflow using a real example.

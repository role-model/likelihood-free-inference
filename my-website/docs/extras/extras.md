---
id: extras
title: Additional Notes
sidebar_position: 1
---

# Additional Notes

This section provides supplementary information to support simulation-based inference workflows, including performance tips, reproducibility practices, and common pitfalls.

---

## Performance Tips

### Use Vectorized Simulators

Whenever possible, implement the simulator as a vectorized function. This significantly improves data generation speed and makes better use of CPU or GPU parallelism.

```python
# Vectorized simulation (example)
x = simulator(theta_batch)
```

## Batch Training

Use appropriate batch sizes during training to improve throughput:

```python
inference.append_simulations(theta, x, batch_size=512).train()
```

Batch size may need tuning based on available GPU memory.

## Monitor GPU Usage

Use ```nvidia-smi``` to track GPU utilization in real time. If your GPU is underutilized, check for CPU–GPU transfer bottlenecks.

## Reproducibility

To ensure consistent results:
* Set random seeds for both PyTorch and NumPy
* Document the prior distribution and simulator version
* Save trained posterior models using ```torch.save```

```python
torch.manual_seed(42)
np.random.seed(42)
```

## Logging and Debugging

Use logging or TensorBoard to monitor training:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='sbi-logs/')
```

Log loss curves, training metrics, and key hyperparameters for debugging or later review.

## Common Pitfalls

Simulation Hangs or Fails
* Ensure all parameter samples produce valid output
* Avoid invalid or extreme values near prior bounds
* Test simulator independently before using in SBI

Posterior Samples Are Degenerate
* Increase the number of training simulations
* Use more expressive density estimators (e.g., NSF instead of MDN)
* Inspect diagnostics such as loss curves and pair plots

Mismatch in Tensor Shapes
* Always ensure simulator output shape matches the expected observation shape
* Use ```.unsqueeze()``` or ```.reshape()``` if needed

###  Contact

For issues, contributions, or questions, please refer to the project's repository or contact the maintainers.

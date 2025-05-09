---
id: full-example
title: Full Example
sidebar_position: 1
---

# Full Example

This section presents a complete example of Simulation-Based Inference (SBI) applied to a real use case. It covers every stage of the workflow—from parameter sampling and simulation to posterior estimation and result analysis.

This example corresponds to the notebook `004-full-example.ipynb`.

---

## Objective

This full example demonstrates how SBI can be applied to a parameter inference problem using a domain-specific simulator. The goal is to recover posterior estimates for unknown model parameters using observed data and simulation-based training.

Estimate the posterior distribution over model parameters given a single observed dataset, using a validated simulator and a predefined prior distribution.

---

## Workflow Overview

![SBI workflow diagram](/img/flowchart.png)


## 1. Define the Prior

A prior distribution is specified over the input parameters. These parameters are drawn from a domain-specific distribution based on known ranges and physical constraints.

Example:

```python
# Prior over 3 parameters: [param1, param2, param3]
prior = BoxUniform(
    low=torch.tensor([0.0, 0.0, 0.0]),
    high=torch.tensor([1.0, 5.0, 10.0])
)

```

Ensure that all sampled parameters produce valid simulator outputs.

## 2. Simulate Training Data
Simulate 𝑁 parameter–observation pairs using the simulator:

```python
theta = prior.sample((N,))
x = torch.stack([simulator(t) for t in theta])
```
If the simulator is vectorized, this step can be significantly optimized.

## 3. Train the Posterior Estimator
Use SNPE to train a neural density estimator:

```python
inference = SNPE(prior)
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)
```

Ensure data is transferred to GPU if available:

```
theta, x = theta.to(device), x.to(device)
```

## 4. Load and Condition on Observation
Load the reference observation (e.g., from ```output.csv```) and convert to a tensor:

```python
x_o = torch.tensor([...])  # from observed dataset
samples = posterior.sample((1000,), x=x_o)
```

These samples represent the posterior over parameters conditioned on the observation.

## 5. Posterior Validation
Analyze the posterior samples using visualization and statistical diagnostics:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(pd.DataFrame(samples.numpy(), columns=[...]))
plt.show()
```

Also consider posterior predictive checks or comparisons with known ground-truth parameters if available.

## 6. Save/Load Posterior for Reuse
To reuse the trained posterior:

```python
torch.save(posterior, 'posterior.pt')
```

Later, reload with:

```python
posterior = torch.load('posterior.pt')
samples = posterior.sample((1000,), x=x_o)
```

This enables reproducible inference without retraining.

## Files Referenced
This example makes use of the following files located in ```examples/data/```:
* output.csv: Observation vector used for conditioning
* params.csv: True simulation parameters (if known)
* posterior.pt: Trained posterior object
* test-posterior.pt: Optional test posterior object for cross-validation

## Summary
This example demonstrates a complete SBI workflow using a custom simulator, a defined prior, and neural posterior estimation via SNPE. The process is fully automated and reproducible using the sbi library and GPU acceleration.

This end-to-end approach can be adapted to other models by substituting the simulator and modifying the parameter space.
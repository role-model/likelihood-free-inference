---
id: sbi-pipeline
title: SBI Pipeline
sidebar_position: 1
---

# SBI Pipeline

This section provides an overview of the Simulation-Based Inference (SBI) pipeline used to estimate posterior distributions in the absence of tractable likelihood functions. The pipeline leverages forward simulations and neural density estimation to perform efficient, scalable inference.

---

## Overview

SBI is designed to perform inference in cases where traditional likelihood-based methods (e.g., MCMC, variational inference) are infeasible due to the complexity or inaccessibility of the likelihood function. Instead, it operates by:

1. Sampling from a prior distribution
2. Simulating observations using a domain-specific simulator
3. Comparing simulated data to observed data
4. Learning a posterior using neural networks

The pipeline is modular and can be extended with different prior types, simulators, summary statistics, and inference algorithms.

---

## Pipeline Components

The diagram below illustrates the full SBI pipeline, from sampling the prior and simulating data, to generating a dataset and training a posterior model.

![SBI Workflow Diagram](/img/flowchart.png)

### 1. Prior Distribution

The pipeline begins by defining a prior over the parameters of interest. This can be a uniform or normal distribution, often implemented via `sbi.utils.BoxUniform` or `torch.distributions`.

```python
prior = sbi.utils.BoxUniform(low=torch.tensor([...]), high=torch.tensor([...]))
```


### 2. Simulator
A simulator function maps parameters to synthetic observations. It must be deterministic or stochastic and compatible with PyTorch tensors.

```python
def simulator(theta):
    # Run your model and return simulated data
    return simulate_model(theta)
```

Simulators can be CPU- or GPU-based, and should ideally be vectorized for performance.

### 3. Observation
This is the real or reference dataset for which we want to estimate the posterior distribution over parameters.

```python
x_o = torch.tensor([...])  # Observed data
```
The observation must match the output format of the simulator.

### 4. Inference Method
The sbi library provides multiple inference methods, such as:
* SNPE (Sequential Neural Posterior Estimation)
* SNLE (Likelihood Estimation)
* SNRE (Ratio Estimation)

A common pattern is:

```python
inference = sbi.inference.SNPE(prior=prior)
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)
```

## Training
The neural density estimator is trained using simulated data (```theta```, ```x```) pairs. Training typically takes place on GPU and supports checkpointing/logging.

## Posterior Sampling
After training, the posterior can be sampled conditionally on the observed data:

```python
samples = posterior.sample((1000,), x=x_o)
```
These samples represent the posterior distribution over parameters given the observed data.

## Summary
The SBI pipeline enables flexible, modular, and scalable inference workflows. It is particularly useful for problems involving:
* Custom simulators
* Complex, high-dimensional data
* Scientific models with uncertain or implicit likelihoods
* Subsequent sections provide details on neural posterior estimation, performance tuning, and full end-to-end examples.

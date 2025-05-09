---
id: neural-posterior
title: Neural Posterior Estimation
sidebar_position: 1
---

# Neural Posterior Estimation

This section covers the role of neural networks in approximating posterior distributions within the Simulation-Based Inference (SBI) pipeline. Neural density estimators provide a scalable and flexible means of learning complex posterior shapes from simulation data.

---

## Purpose of Neural Posterior Estimators

In likelihood-free settings, we cannot compute or differentiate the likelihood. Neural posterior estimators address this by directly modeling the posterior P(θ), trained on synthetic pairs of parameters θ and observations x.

These models learn a conditional distribution over parameters given observed data and can be queried for fast posterior sampling.

---

## Supported Architectures

The `sbi` library supports several neural density estimators:

### 1. Masked Autoregressive Flow (MAF)
- Suitable for continuous, high-dimensional parameter spaces
- Learns flexible, invertible transformations of a base distribution
- Often used as the default in SNPE

### 2. Mixture Density Network (MDN)
- Models the posterior as a mixture of Gaussians
- Simple and interpretable, but less flexible for complex posteriors
- May struggle with multimodality or high-dimensionality

### 3. Neural Spline Flow (NSF)
- An advanced normalizing flow using rational quadratic splines
- Provides highly expressive modeling power

---

## Training the Estimator

The estimator is trained using parameter–observation pairs produced by simulations. The training process is handled internally by the `sbi` inference API:

```python
inference = SNPE(prior)
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)
```

Internally, the network is optimized using negative log-likelihood loss. Training supports GPU acceleration and can be monitored via logging tools such as TensorBoard.

## Conditioning and Sampling

Once trained, the posterior estimator can be conditioned on observed data to draw samples:

```python
samples = posterior.sample((1000,), x=x_o)
```
The estimator returns samples from the learned posterior \( p(θ) | x<sub>o</sub>), where \( x<sub>o</sub> \) is the observation of interest.

## Evaluation and Diagnostics
It is essential to evaluate the performance of the neural posterior estimator. Recommended practices include:
* Visual inspection of posterior samples
* Posterior predictive checks
* Comparison with known ground truth (if available)
* Coverage tests using simulated data

If diagnostic plots or metrics indicate poor calibration or fit, consider increasing simulation budget, using alternative architectures, or tuning hyperparameters.

## Summary
Neural posterior estimation is the core machine learning component of the SBI pipeline. It enables efficient and expressive inference over complex posterior distributions without requiring access to the likelihood function.

In the next section, we describe how to accelerate training and inference using CUDA-enabled GPUs.
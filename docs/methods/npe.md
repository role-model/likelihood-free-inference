---
layout: default
title: Neural Posterior Estimation
---

# Neural Posterior Estimation (NPE)

Neural Posterior Estimation is a simulation-based inference technique that uses neural networks to directly learn the posterior distribution p(θ|x) of parameters θ given observed data x.

## Theoretical Background

NPE belongs to the family of neural density estimation methods for likelihood-free inference. It works by:

1. Generating simulation data from the prior distribution p(θ)
2. Running the simulator to produce synthetic data x for each parameter set θ
3. Training a neural network to approximate the posterior distribution p(θ|x)

The trained network can then be used to estimate the posterior for real observed data.

## Implementation

We implement NPE using the Sequential Neural Posterior Estimation (SNPE) algorithm through the SBI (Simulation-Based Inference) library in Python.

### Key Components

- **Neural Network Architecture**: We use a mixture density network (MDN) with normalizing flows to represent the posterior distribution
- **Training Procedure**: We employ sequential training to progressively focus on regions of high posterior probability
- **Hyperparameters**: Details on network architecture, learning rate, batch size, etc.

## Code Example

```python
import torch
from sbi import utils as utils
from sbi.inference import SNPE

# Define a simulator function
def simulator(theta):
    # Run the Rules of Life Engine Model with parameters theta
    # Return summary statistics of the simulation
    pass

# Set up prior
prior = utils.BoxUniform(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

# Set up inference
inference = SNPE(prior=prior)

# Generate simulations
theta, x = inference.run_simulation(simulator, num_simulations=1000)

# Train the neural network
estimator = inference.train(theta, x)

# Generate posterior from observed data x_o
posterior = inference.build_posterior(estimator)
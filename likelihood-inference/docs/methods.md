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



# ==========================================================================================


---
layout: default
title: Likelihood-Free Inference Methods
---

# Methods

This page provides an overview of the different likelihood-free inference methods explored in this project. We are investigating various approaches to parameter inference for complex simulation models where traditional likelihood-based methods are intractable.

## Likelihood-Free Inference

Likelihood-free inference (also known as simulation-based inference or approximate Bayesian computation) refers to a class of methods that enable parameter inference for models with intractable likelihoods. These methods are particularly useful for complex simulation models like our Rules of Life Engine Model.

## Available Methods

- [Neural Posterior Estimation (NPE)](npe.html) - A deep learning approach that directly learns the posterior distribution
- [Neural Ratio Estimation (NRE)](nre.html) - A method that learns the ratio between posterior and prior distributions

Each method page includes:
- Theoretical background
- Implementation details
- Code examples
- Performance metrics
- Advantages and limitations

## Method Comparison

For a detailed comparison of these methods applied to our specific problem, see our [Results Comparison](../results/comparisons.html) page.
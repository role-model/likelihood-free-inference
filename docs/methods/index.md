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
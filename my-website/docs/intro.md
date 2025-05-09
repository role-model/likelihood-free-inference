---
id: intro
title: Introduction
sidebar_position: 1
slug: /
---

# Introduction

This documentation provides an overview of a Simulation-Based Inference (SBI) workflow for performing Likelihood-Free Inference (LFI) using modern machine learning techniques.

It covers the architecture, tools, and methodology used to estimate posterior distributions in cases where the likelihood function is unavailable or computationally intractable.

### Project Context

This project was developed as part of the ICS 496 Capstone course at the University of Hawai‘i at Mānoa.

It was **sponsored** by Andy Rominger, with the goal of applying deep learning techniques to ecological and evolutionary modeling. The team designed and implemented an end-to-end SBI pipeline using custom simulation tools, PyTorch, and the `sbi` library.

**Team Members**:
- Frances Uy
- Micah Tilton

**Supervising Instructor**: Mehdi Tarrit Mirakhorli


### Goals

The __Rules of Life Engine Model__—an _agent-based simulation model_ motivating the need for LFI—that predicts multiple aspects of biodiversity from first principles using biological mechanisms including how organisms interact with each other to acquire resources, how they reproduce, how new species arise through speciation, and how traits of populations change through evolution. The multiple aspects of biodiversity predicted by the simulation model include the number of species (species richness), the population sizes of each species, the genetic diversity of populations (number and frequency of unique alleles), and the heterogeneity of traits.

An _Agent-Based Simulation_ model is a computational approach for modeling complex systems by simulating the interactions of autonomous entities, known as agents. Each agent operates based on its own set of rules and behaviors, and the overall system behavior emerges from these local interactions.

This simulation model includes roughly 10 tunable parameters that govern how agents live, die, reproduce, and speciate. Our objective is to infer these parameters from real datasets. Since the model is too complex to express analytically with an equation, we use AI to learn the mapping from data patterns back to parameter values. Unlike standard AI tasks that classify or predict data, our approach trains on simulated data to recover the parameters that produced it. Once trained, the model is applied to real data to estimate the underlying parameters.

### Understanding the Underlying Parameters

The simulation model is governed by a set of adjustable parameters that define how individual agents (e.g., organisms) behave and evolve over time. These parameters influence key ecological and evolutionary processes such as birth, death, dispersal, mutation, and speciation.

In the context of simulation-based inference (SBI), these are referred to as the underlying parameters—the latent variables that we aim to estimate from observed data.

Examples include:

* Ecological parameters: `speciation_local`, `extinction_meta`, `dispersal_prob`, `individuals_local`
* Evolutionary parameters: `mutation_rate`, `trait_sigma`, `env_sigma`
* Simulation controls: `equilib_escape`, `num_basepairs`, `niter`

During inference, we use the SBI pipeline to learn the relationship between these parameters and the patterns they generate in simulated data. Once trained, the model can infer the most likely values of these parameters that explain a real observed dataset.

This approach allows us to study complex systems where the underlying dynamics are too intricate to describe analytically but can be simulated effectively.

### Sources

The simulation model is previously implemented in a combination of C++, R, and python.  You can view work to date [here](https://github.com/role-model/roleR).

You can view a publication describing an earlier version of this model [here](https://onlinelibrary.wiley.com/doi/full/10.1111/1755-0998.13514).

---

## Overview of Likelihood-Free Inference

Likelihood-Free Inference is applicable when the probability of observing data given model parameters (i.e., the likelihood) cannot be explicitly computed. Instead, it relies on generating simulated data from a forward model and comparing it to observed data to estimate posteriors.

This approach is common in complex scientific models where analytical likelihoods are difficult or impossible to derive.

---

## Simulation-Based Inference (SBI)

SBI refers to a class of machine learning methods that perform inference by:
- Sampling parameters from a prior
- Running a simulator to generate synthetic data
- Using neural networks to approximate the posterior

The `sbi` library is used to implement this pipeline, supporting a range of neural density estimators such as MAF, NSF, and MDN, along with GPU acceleration.

---

## Use Cases

This framework is designed for inference in domains such as:
- Neuroscience and cognitive modeling
- Physics-based simulations
- Systems biology
- Any scientific context where simulators are available but likelihoods are not

---

## Documentation Structure

The documentation is organized into the following sections:

- **Getting Started**: Environment setup and dependencies  
- **SBI Pipeline**: Core concepts and components of the inference workflow  
- **Neural Posterior Estimation**: Details on network architecture and training  
- **GPU Acceleration**: Guidance on using CUDA-enabled devices  
- **Full Example**: End-to-end example with real data and outputs  
- **Extras**: Additional notes, performance tips, and data handling

Each section is self-contained and assumes familiarity with Python and PyTorch.
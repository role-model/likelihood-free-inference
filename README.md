# Likelihood-Free Inference
__Experimenting with different likelihood-free inference methods__

---

## Table of Contents
- [Goals](#goals)
- [Parameters](#parameters)
- [Demo](#demo)
- [Results](#results)
- [Conclusion](#conclusion)

---
### Goals
The Rules of Life Engine Model is an agent-based simulation model that predicts multiple aspects of biodiversity from first principles using biological mechanisms including how organisms interact with each other to acquire resources, how they reproduce, how new species arise through speciation, and how traits of populations change through evolution. The multiple aspects of biodiversity predicted by the simulation model include the number of species (species richness), the population sizes of each species, the genetic diversity of populations (number and frequency of unique alleles), and the heterogeneity of traits.

This simulation model has approximately 10 adjustable parameters that govern how agents (individual organisms) live, die, reproduce, and ultimately give rise to new species. Our goal is to be able to fit these parameters to real data sets.  Because the model is too complex to capture with an equation (or set of equations) we instead have to train an AI to predict parameter values from patterns in data.  This is different from a typical AI workflow which classifies data, or predicts patterns in data.  Here we train the AI not on real data, but on data simulated from our agent based model.  The AI does not predict or classify the simulated data, but rather predicts the parameters that generated the simulated data. We then confront that trained AI with real data and ask it to predict the parameter values responsible for generating those real data.

We have seen moderate success using simple random forest regression.  We see a team of students to develop even better approaches using deep learning.

The model is implemented in a combination of C++, R, and python.  You can view our work to date here: https://github.com/role-model/roleR

You can view a publication describing an earlier version of this model here: https://onlinelibrary.wiley.com/doi/full/10.1111/1755-0998.13514 (open access)

---
### Parameters
These set of parameters are known to work with the simulation Other sets of parameters may cause the simluation to hang indefinitely

```
original_param_distribution = {
    "individuals_local": 100,
    "individuals_meta": 1000,
    "species_meta": 50,
    "speciation_local": 0.05,
    "speciation_meta": 0.05,
    "extinction_meta": 0.05,
    "env_sigma": 0.5,
    "trait_sigma": 1,
    "comp_sigma": 0.5,
    "dispersal_prob": 0.1,
    "mutation_rate": 0.01,
    "equilib_escape": 1,
    "num_basepairs": 250,
    "init_type": 'oceanic_island',
    "niter": 10000,
    "niterTimestep": 10
}
```
---
### Demo
_Goals for Demo_

- Wrapper convert R model into Python
- Sample random parameters
- Generate dataset
- Train NPE using SBI
- Hyper parameter tuning (maybe)
- Metrics: Accuracy, AUROC

_Stretch Goals_:

- Train different models
- User interface
- By Demo Day
- Generate Dataset
- Try to train a model
- Update the website
- Create presentation slides (if we need to)
- Create Demo Jupyter Notebook
- Clean up the repo

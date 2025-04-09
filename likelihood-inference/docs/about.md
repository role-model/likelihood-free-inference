# About the Likelihood-Free Inference Project

## Background

The Rules of Life Engine Model is an agent-based simulation model that predicts multiple aspects of biodiversity from first principles using biological mechanisms including how organisms interact with each other to acquire resources, how they reproduce, how new species arise through speciation, and how traits of populations change through evolution.

This simulation model has approximately 10 adjustable parameters that govern how agents (individual organisms) live, die, reproduce, and ultimately give rise to new species. Our goal is to fit these parameters to real data sets.

## The Challenge

Because the model is too complex to capture with an equation (or set of equations), we need to train AI models to predict parameter values from patterns in data. This differs from typical AI workflows in that:

1. We train the AI not on real data, but on data simulated from our agent-based model
2. The AI doesn't predict or classify the simulated data, but rather predicts the parameters that generated the data
3. We then apply the trained AI to real data to predict the parameter values responsible for generating those real data

## Previous Work

We have seen moderate success using simple random forest regression and are now developing better approaches using deep learning and other likelihood-free inference techniques.

The simulation model is previously implemented in a combination of C++, R, and Python. You can view related work at [the roleR GitHub repository](https://github.com/role-model/roleR).

A publication describing an earlier version of this model is available [here](https://onlinelibrary.wiley.com/doi/full/10.1111/1755-0998.13514).
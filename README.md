# Likelihood-Free Inference

This project implements a complete **Simulation-Based Inference (SBI)** pipeline for complex ecological simulations. By using deep neural density estimators (e.g., Masked Autoregressive Flow), we recover the underlying parameters of an agent-based model from observed biodiversity data — without needing an explicit likelihood function.

Developed as part of an undergraduate capstone project (ICS 496 at University of Hawai‘i at Mānoa), this repository showcases an end-to-end example of **Likelihood-Free Inference (LFI)** applied to real-world ecological research.

---

Documentation Site: https://role-model.github.io/likelihood-free-inference/ 

## Features

- Agent-based simulation model for biodiversity
- Neural posterior estimation using PyTorch and the `sbi` library
- GPU acceleration with CUDA
- Full inference workflow and result visualization
- Jupyter notebooks for testing and experiments

---

## Quick Start

```bash
git clone https://github.com/your-org/likelihood-free-inference
cd likelihood-free-inference

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

**Authors**: Micah Tilton & Frances Uy
**Project Sponsor**: Andrew Rominger, PhD


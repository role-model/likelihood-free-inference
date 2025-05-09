---
id: getting-started
title: Getting Started
sidebar_position: 2
---

# Getting Started

This section describes how to prepare your environment to run the simulation-based inference (SBI) pipeline. It covers installation, environment setup, and verified parameters for reliable simulation execution.

---

## System Requirements

Ensure the following tools are installed on your system:

- **Python ≥ 3.8**
- **PyTorch ≥ 1.12** (with CUDA if GPU support is required)
- **Jupyter Notebook** or **JupyterLab**
- Python scientific libraries: `numpy`, `scipy`, `matplotlib`, `seaborn`
- SBI toolkit: `sbi`

---

## Installation (macOS/Linux)

A virtual environment is recommended for dependency isolation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

To run on GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Directory Layout

```
├── examples/ # All example notebooks and related outputs
│ ├── 000-getting-started.ipynb
│ ├── 001-sbi-pipeline.ipynb
│ ├── 002-neural-network.ipynb
│ ├── 003-sbi-gpu-acceleration.ipynb
│ ├── 004-full-example.ipynb
│ ├── scratch.ipynb
│ └── data/
│ ├── output.csv
│ ├── params.csv
│ ├── posterior.pt
│ └── test-posterior.pt
├── my-website/ # Docusaurus documentation site
├── src/ # Python source code (package: roler)
│ └── roler/
├── sbi-logs/ # Training logs and TensorBoard events
├── pyproject.toml # Project metadata and dependencies
├── main.ipynb # Entry-point notebook (optional)
└── README.md
```

## Verified Simulation Parameters

The following parameter set is validated and known to run the simulation successfully. Alternative configurations may result in long runtimes or hanging behavior.

```python
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
    "init_type": "oceanic_island",
    "niter": 10000,
    "niterTimestep": 10
}
```

## Running a Test Notebook

To verify your environment:

1. Activate your virtual environment
2. Run:

```bash
jupyter notebook
```
3. Open and execute the ```000-getting-started.ipynb``` notebook
4. Successful execution should:
* Simulate data using the above parameters
* Visualize intermediate results
* Return an inference-ready dataset

## Next Steps

Continue to the SBI Pipeline section to explore the components used to implement simulation-based inference.
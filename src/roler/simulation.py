import subprocess
import warnings

# Import necessary modules
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
except ImportError as e:
    raise ImportError(
        "The 'rpy2' library is required but not installed. "
        "Install it with 'pip install rpy2'."
    ) from e

# Activate pandas conversion for rpy2
pandas2ri.activate()

# Function to check if R is installed
def check_r_installed():
    try:
        result = subprocess.run(
            ["R", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise EnvironmentError("R is not installed or not available in PATH.")
    except FileNotFoundError:
        raise EnvironmentError("R is not installed or not available in PATH.")
    except Exception as e:
        raise EnvironmentError(f"An error occurred while checking for R: {e}")

# Check if R is installed
check_r_installed()

# Install R packages if they are not already installed
try:
    remotes = importr('remotes')
    print("Installing the 'roleR' R package from GitHub...")
    remotes.install_github("role-model/roleR", dependencies=True)
except Exception as e:
    warnings.warn(
        f"Error installing R packages: {e}\n"
        "Make sure you have R and the 'remotes' package installed correctly.",
        RuntimeWarning
    )

# Import the R package
try:
    roleR = importr('roleR')
except Exception as e:
    raise ImportError(
        f"Error importing the 'roleR' R package: {e}\n"
        "Ensure the package is installed and available in your R environment."
    ) from e

import pandas as pd
import torch
import numpy as np
from dataclasses import asdict
from roler.model import ModelPrior, ModelParams

class Simulator:
    def __init__(self, prior: ModelPrior = None):
        self.prior = prior
    
    def simulate(self, theta: ModelParams) -> pd.DataFrame:
        params = roleR.roleParams(**asdict(theta))
        model = roleR.runRole(roleR.roleModel(params))
        stats = roleR.getSumStats(model)
        
        stats_df = pandas2ri.rpy2py(stats)
        return pd.DataFrame(stats_df)       
        
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        if self.prior is None:
            raise ValueError("ModelPrior must be specified in Simulator constructor.")
        
        params = roleR.roleParams(**asdict(self.prior.get_params(theta)))
        self.simulate(params)
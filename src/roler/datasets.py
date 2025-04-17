from roler.simulation import Simulator
from typing import TypedDict
import pandas as pd
from joblib import Parallel, delayed
import warnings

class Dataset(TypedDict):
    params: pd.DataFrame
    output: pd.DataFrame

class DatasetGenerator:
    def __init__(self, simulator: Simulator):
        self.simulator = simulator

    def _safe_simulate(self, theta):
        try:
            # call your simulator (via __call__ or direct)
            return self.simulator(theta)
        except Exception as e:
            warnings.warn(f"Simulation failed for params {theta}: {e}", RuntimeWarning)
            return None

    def generate_dataset(self, samples: int, n_jobs: int = 1) -> Dataset:
        """
        Generate a dataset of parameters and simulation outputs using joblib for parallelism.
        Any simulation that raises an exception will be skipped.
        """
        if self.simulator.prior is None:
            raise ValueError("Simulator has no prior for sampling.")

        # 1) Sample parameter objects
        param_objs = [self.simulator.prior.sample() for _ in range(samples)]

        # 2) Run simulations (safely) in parallel or serial
        if n_jobs == 1:
            outputs = [self._safe_simulate(theta) for theta in param_objs]
        else:
            outputs = Parallel(
                n_jobs=n_jobs, verbose=10, batch_size=5
            )(delayed(self._safe_simulate)(theta) for theta in param_objs)

        # 3) Pair params with their outputs, drop failures
        successes = [
            (theta, out)
            for theta, out in zip(param_objs, outputs)
            if out is not None
        ]

        if not successes:
            # if everything failed, return empty DataFrames
            return {"params": pd.DataFrame(), "output": pd.DataFrame()}

        # 4) Build params DataFrame from only the successful thetas
        params_df = pd.DataFrame([theta.numpy() for theta, _ in successes])

        # 5) Concatenate only the successful outputs
        output_df = pd.concat([out for _, out in successes], ignore_index=True)

        return {"params": params_df, "output": output_df}

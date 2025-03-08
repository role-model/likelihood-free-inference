import torch
from roler.model import ModelPrior, ModelParams
from roler.simulation import Simulator
import numpy as np
from joblib import Parallel, delayed

DEFAULT_COLUMNS = [
    "richness",
    "hill_abund_1",
    "hill_abund_2",
    "hill_abund_3",
    "hill_abund_4",
    "hill_trait_1",
    "hill_trait_2",
    "hill_trait_3",
    "hill_trait_4",
]


class Dataset:
    def __init__(
        self,
        simulator: Simulator,
        prior: ModelPrior,
        columns: list[str] = DEFAULT_COLUMNS,
    ):
        self.prior = prior
        self.simulator = simulator
        self.columns = columns

    def generate_dataset(
        self,
        samples: int,
        select_ratio: float = None,
        select_last_n: int = None,
        n_jobs: int = 1,
    ):
        """
        Generates a dataset by running simulations in parallel.

        Parameters:
            samples (int): Number of simulations to run.
            select_ratio (float, optional): If provided (and between 0 and 1),
                only keep the last portion (given by the ratio) of timesteps.
            select_last_n (int, optional): If provided, keep only the last n timesteps.
            n_jobs (int): Number of parallel jobs to run.

        Returns:
            list: [theta_samples_transformed, x_samples_transformed] where
                - theta_samples_transformed is a tensor of repeated parameter samples,
                - x_samples_transformed is a tensor of simulation outputs.
        """
        # Get the joint uniform distribution prior.
        tensor_prior = self.prior.get_joint_uniform()

        def simulate_one_sample(_):
            # Sample a parameter vector theta.
            theta = tensor_prior.sample()
            # Convert theta from the tensor into parameters format.
            params = self.prior.get_params_from_tensor(theta)

            # Pass theta as an argument to simulate, as it is required.
            stats_df = self.simulator.simulate(params)
            # Filter the DataFrame to keep only the desired columns.
            stats_df = stats_df[[col for col in stats_df.columns if col in self.columns]]
            stats_df = stats_df.dropna()

            # Convert simulation data to a torch Tensor.
            x = torch.Tensor(np.array(stats_df))
            num_timesteps = x.shape[0]

            # Apply selection logic per simulation.
            if select_ratio is not None and 0 < select_ratio <= 1:
                keep_timesteps = max(1, int(num_timesteps * select_ratio))
                x = x[-keep_timesteps:]
            elif select_last_n is not None:
                x = x[-min(select_last_n, num_timesteps) :]

            # Repeat theta for each timestep in this simulation.
            theta_tile = torch.tile(theta, (x.shape[0], 1))
            return theta_tile, x

        # Run simulations in parallel.
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_one_sample)(i) for i in range(samples)
        )

        # Collect and concatenate the results.
        theta_samples_transformed = []
        x_samples_transformed = []
        for theta_sample, x_sample in results:
            theta_samples_transformed.append(theta_sample)
            x_samples_transformed.append(x_sample)

        if theta_samples_transformed:
            theta_samples_transformed = torch.cat(theta_samples_transformed, dim=0)
        else:
            theta_samples_transformed = torch.tensor([])

        if x_samples_transformed:
            x_samples_transformed = torch.cat(x_samples_transformed, dim=0)
        else:
            x_samples_transformed = torch.tensor([])

        return [theta_samples_transformed, x_samples_transformed]

    def get_params_from_tensor(self, sample: torch.Tensor) -> ModelParams:
        return self.prior.get_params_from_tensor(sample)

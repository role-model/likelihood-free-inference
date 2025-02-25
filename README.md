# likelihood-free-inference
Experimenting with different likelihood-free inference methods

These set of parameters are known to work with the simulation
Other sets of parameters may cause the simluation to hang indefinitely
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
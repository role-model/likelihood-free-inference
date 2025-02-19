library(remotes)
remotes::install_github("role-model/roleR",dependencies=TRUE)

library(roleR)
library(ggplot2)

p <- roleParams(
  individuals_local = 100,
  individuals_meta = 1000,
  species_meta = 50,
  speciation_local = 0.05,
  speciation_meta = 0.05,
  extinction_meta = 0.05,
  env_sigma = 0.5,
  trait_sigma = 1,
  comp_sigma = 0.5,
  dispersal_prob = 0.1,
  mutation_rate = 0.01,
  equilib_escape = 1,
  num_basepairs = 250,
  init_type = 'oceanic_island',
  niter = 10000,
  niterTimestep = 100
)

model <- runRole(roleModel(p))
# How to get all parameters
stats <- getSumStats(model, list(rich = richness,hill_abund=hillAbund))

ggplot(stats, aes(iteration, rich)) +
  geom_line()


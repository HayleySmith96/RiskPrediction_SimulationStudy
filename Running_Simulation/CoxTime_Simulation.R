# CoxTime Simulation Script
library(survival)
library(data.table)
library(reticulate)
library(riskRegression)
library(survivalmodels)

source('../Running_Simulation/Simulation_Script.R') # Get methods from simulation script file
use_python('../env_simulation/bin/python', required=TRUE)
use_virtualenv('~/env_simulation')
source_python('../Running_Simulation/CoxTime_Simulation.py')

random = import('random')

nobs = 2982
reps = 2000
data_types = list('cox', 'rcs', 'mfp', 'rsf', 'NN', 'CoxTime')
method = 'CoxTime' 
eval_times = c(1, 5, 10, 14)

performance_measures = data.frame(n_obs = double(),
                                  data_type = character(),
                                  method = character(),
                                  time = double(),
                                  rep_number = double(),
                                  auc = double(),
                                  brier = double(),
                                  mse = double(),
                                  mae = double())

for(data_type in data_types){
  # Set seed
  seed_R = 12345
  set_seed(seed_R, seed_np = seed_R, seed_torch = seed_R)
  random$seed(seed_R)
  performance_measures = simulation(nobs, reps, method = method, data_type, eval_times, performance_measures)
}

filename = paste('../Results/',
                 nobs, '/CoxTime_Results.csv', sep='')
write.csv(performance_measures, filename)

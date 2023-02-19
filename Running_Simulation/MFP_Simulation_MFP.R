# MFP Simulation Script
library(mfp)
library(survival)
library(data.table)
library(reticulate)
library(riskRegression)

source('../Running_Simulation/Simulation_Script.R') # Get methods from simulation script file
set.seed(12345)

nobs = 2982
reps = 2000
data_types = list('mfp')
method = 'mfp'
eval_times = c(1,5,10,14)

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
  performance_measures = simulation(nobs, reps, method = method, data_type, eval_times, performance_measures)
}

filename = paste('../Results/',
                 nobs, '/MFP_Results_MFP.csv', sep='')
write.csv(performance_measures, filename)

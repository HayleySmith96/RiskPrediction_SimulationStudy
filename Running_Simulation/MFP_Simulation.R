# MFP Simulation Script
library(mfp)
library(survival)
library(data.table)
library(reticulate)
library(riskRegression)

source('../Running_Simulation/Simulation_Script.R') # Get methods from simulation script file

nobs = 2982
reps = 2000
data_types = list('cox', 'rcs', 'mfp', 'rsf', 'NN', 'CoxTime')
method = 'mfp' 
eval_times = c(1,5,10,14)

performance_measures = data.frame(data_type = character(),
                                  method = character(),
                                  time = double(),
                                  rep_number = double(),
                                  auc = double(),
                                  brier = double(),
                                  mse = double(),
                                  mae = double())

for(data_type in data_types){
  set.seed(12345)
  performance_measures = simulation(reps, method = method, data_type, eval_times, performance_measures)
}

filename = paste('../Results/',
                 nobs, '/MFP_Results.csv', sep='')
write.csv(performance_measures, filename)

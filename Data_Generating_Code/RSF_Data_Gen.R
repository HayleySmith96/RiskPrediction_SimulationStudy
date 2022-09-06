# RSF DATA-GENERATING SCRIPT
library(tibble)
library(tidyverse)
library(survival)
library(ranger)

source('../Data_Generating_Code/Data_Generating_Methods.R') # Get the whole data gen script - call RSF only from this loop

############------------ SET THE SEED ------------############
set.seed(12345)
# True model fit is sank to ../Results/true_models/RSF_true_model.txt
# True predictions are saved to ../Results/true_models/RSF_true_predictions.txt

############------------ GET THE REAL DATA ------------############
# Load in the real dataset 
filepath = '../datasets/Real_Datasets/RGBC_data.csv' # HPC
real_data = read_csv(filepath)
real_data = real_data[,-1] # Remove X1 row count column

# Sample size
nobs = 2982

# Number of datasets
r = 2000

# Adminstrative censoring time
admin_cens = 15

# Maximum number of random search iterations
max_iter = 40

# Simulate the datasets
simulate_r_rsf(real_data, nobs, r, admin_cens, max_iter)


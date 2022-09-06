# COX DATA-GENERATING SCRIPT
library(tibble)
library(tidyverse)
library(survival)

source('../Data_Generating_Code/Data_Generating_Methods.R')

############------------ SET THE SEED ------------############
set.seed(12345)
# True model fit is sank to ../Results/true_models/Cox_true_model.txt
# True predictions are saved to ../Results/true_models/Cox_true_predictions.txt 

############------------ GET THE REAL DATA ------------############
# Load in the real dataset
filepath = '../datasets/Real_Datasets/RGBC_data.csv'
real_data = read.csv(filepath)
real_data = real_data[,-1] # Remove X1 row count column

# Sample size 
nobs = 2982

# Number of datasets
r = 2000

# Administrative censoring time
admin_cens = 15

# Simulate the datasets
simulate_r_cox(real_data, nobs, r, admin_cens)

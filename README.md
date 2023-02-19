# RiskPrediction_SimulationStudy
Code for simulation study comparing statistical and machine learning approaches to risk prediction 

# Aim

Compare prognostic predictive performance of several risk prediction methods, in terms of calibration and discrimination, with:
• Different DGMs
• Short, medium, and long follow-up times
• Various training sample sizes


# Data

Real data used to simulate data = Rotterdam German Breast Cancer dataset (sample = 2982, covariates = 9).

Data is generated from:
• Cox Proportional Hazards Model (Cox)
• Flexible Parametric Model (FP)
• Multivariable Fractional Polynomial (MFP)
• Random Survival Forest (RSF)
• Cox-CC Neural Network (CoxCC)
• Cox-Time Neural Network (CoxTime)

Training sample sizes = 500, 1000, 2000, 2982
Testing sample size = 2982

Repetitions = 2000


# Estimand

Survival probability at t years, where t = 1, 5, 10, 14 years. 


# Methods Compared

Cox Proportional Hazards Model (Cox)
Flexible Parametric Model (FP)
Multivariable Fractional Polynomial (MFP)
Random Survival Forest (RSF)
Cox-CC Neural Network (CoxCC)
Cox-Time Neural Network (CoxTime)


# Performance Measures

Brier Score (calibration)
Mean Absolute Error (calibration)
Time-Dependent AUC (discrimination)
Variance, Standard Deviation, and MCSE (variance)


## Generating the Simulated Datasets

Data_Generating_Methods.R contains the code for simulating data from each of the methods. 

To generate data, run each of the Data_Gen files: Cox_Data_Gen.R; FP_Data_Gen.R; MFP_Data_Gen.R, RSF_Data_Gen.R; CoxCC_Data_Gen.py, and CoxTime_Data_Gen.py. 

Default sample size is nobs = 2982. This can be changed by altering the value for 'nobs' in each of the Data_Gen files. 

Default number of simulated datasets is r = 2000. This can be changed by altering the value for 'r' in each of the Data_Gen files. 

Default administrative censoring time is admin_cens  = 15. This can be changed by altering the value for 'admin_cens' in each of the Data_Gen files.

The maximum number of iterations for Random Search for the RSF, CoxCC, and CoxTime methods is max_iter = 40. This can be changed by altering the value for 'max_iter' in each of the Data_Gen files.


## Running the Simulation

Simulation_Script.R contains the methods to fit each of the models to the simulated training datasets, evaluate them on the testing datasets, and get performance measures. 

Performance_Measures.R contains the methods to calculate the AUC, Brier Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE). 

Each method (Cox, FP, MFP, RSF, CoxCC, and CoxTime) can be run for all of the DGMs or for each of the DGMs seperately. The R code is provided for both of these approaches. 

If running the CoxCC and CoxTime methods for multiple DGMs simultaneously, they must be run in seperate folders to avoid issues with saving the weightcheckpoint files.

NOTE: Python virtual environment file paths must be changed to the path of your virtual environment in the CoxCC and CoxTime files. 


## Simulation Results

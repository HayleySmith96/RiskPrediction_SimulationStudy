# Fitting each model to each dataset
library(survival)
library(data.table)
library(reticulate)
library(riskRegression)
source('../Running_Simulation/Performance_Measures.R')
source('../Running_Simulation/Fitting_Methods.R')

use_python('../env_simulation/bin/python', required=TRUE)
use_virtualenv('~/env_simulation')
source_python('../Running_Simulation/CoxCC_Simulation.py')
source_python('../Running_Simulation/CoxTime_Simulation.py')

# Function to turn factor columns of rotterdam into factors
factor_data = function(data){
  data = as_tibble(data) # Make into a tibble
  
  # Make columns factors
  data$size = as.factor(data$size)
  data$grade = as.factor(data$grade)
  
  return(data)
}

# Get the true predictions for true event time closest to eval_time
get_true_predictions = function(data_type){
  # Read in the true predictions
  filename = paste('../Results/true_models/true_predictions_', 
                   data_type, '.csv', sep='')
  
  # Transpose and no header the data frame if its CoxCC method
  if(data_type == 'CoxCC' | data_type == 'CoxTime'){
    true_predictions = read.csv(filename, header=FALSE)
    true_predictions = as.data.frame(t(as.matrix(true_predictions)))
  } else {
    true_predictions = read.csv(filename)
  }

  return(true_predictions)
}

get_true_event_times = function(){
  # Load in the real dataset 
  filepath = '../datasets/Real_Datasets/RGBC_data_sorted.csv' # HPC
  real_data = read_csv(filepath)
  real_data = subset(real_data, select = -c(...1) ) # Remove X1 row count column
  
  events = real_data[real_data$d == 1, ]
  true_event_times = sort(unique(events$t)) # Get sorted unique event times

  return(true_event_times)
}

# Run the simulation - methods list
simulation_one_rep = function(method, nobs, train_dataset, test_dataset, eval_times, true_predictions, true_event_times){
  
  if(method=='cox'){
    model = fit_cox_model(train_dataset)
    predictions = predict_cox_model(model, test_dataset, eval_times)
  }else if(method == 'fp'){
    model = fit_fp_model(train_dataset)
    predictions = predict_fp_model(model, test_dataset, eval_times)
  }else if(method == 'mfp'){
    predictions = predict_mfp_model(train_dataset, test_dataset, eval_times)
  } else if(method == 'rsf'){
    model = fit_rsf_model(train_dataset, nobs)
    predictions = predict_rsf_model(model, test_dataset, eval_times)
  } else if(method == 'CoxCC'){
    model = fit_CoxCC_model(train_dataset, nobs)
    predictions = as.data.frame(predict_CoxCC_model(model, test_dataset, eval_times))
    predictions = data.matrix(predictions, rownames.force = NA)
  } else if(method == 'CoxTime'){
    model = fit_CoxTime_model(train_dataset, nobs)
    predictions = as.data.frame(predict_CoxTime_model(model, test_dataset, eval_times))
    predictions = data.matrix(predictions, rownames.force = NA)
  }
  if(is.null(predictions)== FALSE){
    auc_value = auc(predictions, test_dataset, eval_times)
    brier_value = brier_score(predictions, test_dataset, eval_times)
    #int_brier_score = int_brier_score(full_predictions, test_dataset)
    mse_value = mse(predictions, true_predictions, true_event_times, eval_times)
    mae_value = mae(predictions, true_predictions, true_event_times, eval_times)
  } else {
    auc_value = NA
    brier_value = NA
    mse_value = NA
    mae_value = NA
  }
  pms = list(auc = auc_value, brier = brier_value, mse = mse_value, mae = mae_value)
  return(pms)
}

simulation = function(nobs, reps, method, data_type, eval_times, performance_measures){
  # Get the true predictions and the true event times
  true_predictions = get_true_predictions(data_type)
  true_event_times = get_true_event_times()
  
  rep = 1
  while(rep <= reps){
    dataset_number = rep
    # Training data
    filename = paste('../datasets/', nobs, '/', 
                data_type, '/training/dataset_train_', data_type, '_', dataset_number,
                '.csv', sep='')
    train_dataset = read.csv(filename)
    
    # Testing data
    filename = paste('../datasets/', nobs, '/', 
                     data_type, '/testing/dataset_test_', data_type, '_', dataset_number,
                     '.csv', sep='')
    test_dataset = read.csv(filename)

    # Make sure columns are factors
    train_dataset = factor_data(train_dataset)
    test_dataset = factor_data(test_dataset)
    
    # Do simulation run
    pms = simulation_one_rep(method, nobs, train_dataset, test_dataset, eval_times, true_predictions, true_event_times)
    i = 1
    while(i<=length(eval_times)){
      pms_rep = data.frame(n_obs = nobs,
                          data_type = data_type,
                          method = method,
                          time = eval_times[i],
                          rep_number = rep,
                          auc = as.numeric(pms$auc[i]),
                          brier = as.numeric(pms$brier[i]),
                          mse = as.numeric(pms$mse[i]),
                          mae = as.numeric(pms$mae[i]))
      performance_measures = rbind(performance_measures, pms_rep)
      i = i + 1
    }
    
    rep = rep + 1
  }
  
  return(performance_measures)
}

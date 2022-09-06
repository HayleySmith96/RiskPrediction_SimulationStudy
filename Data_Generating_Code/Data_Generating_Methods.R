###### Simulate data methods for each risk prediction model ######
library(tibble)
library(tidyverse)
library(survival)
library(ranger)
library(rstpm2)
library(mfp)

############------------ SINK TRUE MODELS TO TXT FILES ------------############

# Function to turn factor columns of rotterdam into factors
factor_data = function(real_data){
  real_data = as_tibble(real_data) # Make into a tibble
  
  # Make columns factors
  real_data$size = as.factor(real_data$size)
  real_data$grade = as.factor(real_data$grade)
  
  return(real_data)
}

############------------ GENERAL METHOD TO GET NEW TIMES ------------############
simulate_new_times = function(type, real_data, predictions, event_times, cov_names, admin_cens, nobs){
  n = nrow(real_data) # Number of observations
  if(type == 'test'){
    nobs = n # Sample size is full sample for testing datasets
    data_sample = real_data # data_sample is full dataset
  } else{ # Sample training data to sample size with replacement
    rows = sample(1:n, nobs, replace=TRUE)
    data_sample = real_data[rows, ]
    predictions = predictions[rows, ] # Get predictions for the correct sampled rows
  }
  
  # Draw from uniform, vector of n_obs numbers
  U = runif(nobs,0,1)

  # Find row min of abs(predictions - U)
  abs_values = abs(predictions - U)
  row_mins = apply(abs_values, 1, which.min)

  # Matrix of indexes (rownumber, row_min_index) of minimums
  indexes = cbind(c(1:nobs), row_mins)
  
  # Create new times vector
  newtimes = vector("double", nobs)

  # Get survival_probabilities at minimum indexes
  survival_ts = predictions[indexes]
  
  # For each individual in the dataset, simulate their new survival time
  for(i in 1:nobs){
    # Get corresponding row_min index and s(t)
    index_t = row_mins[i]
    survival_t = survival_ts[i]
    
    # Get the times at which S(ti) < U < S(ti+1) and interpolate
    if(U[i] > survival_t & index_t != 1){
      if(survival_t == predictions[i, index_t-1]){
        time = event_times[index_t]
      } else{
        inter = approx(c(predictions[i, index_t-1], survival_t), 
                       c(event_times[index_t-1], event_times[index_t]),
                       xout=U[i])
        time = inter$y
      }
    } else if(U[i] > survival_t & index_t == 1){
      inter = approx(c(1, survival_t), 
                     c(0, event_times[index_t]),
                     xout=U[i])
      time = inter$y
    } else if(U[i] < survival_t & index_t != length(event_times)) {
      if(survival_t == predictions[i, index_t+1]){
        time = event_times[index_t]
      } else{
        inter = approx(c(survival_t, predictions[i, index_t+1]),
                      c(event_times[index_t], event_times[index_t+1]),
                      xout=U[i])
        time = inter$y
      }
    } else if(U[i] < survival_t & event_times[index_t] == max(event_times)){
      time = runif(1, event_times[index_t], admin_cens + 1)
    } else {
      time = event_times[index_t]
    }
    
    # Save new survival time
    newtimes[i] = time
  }
  
  # Censoring - take minimum of censoring and event times and corresponding indicator
  lam = 0.01
  U = runif(nobs)
  exp_censoring = -log(U)/lam
  event_ind_exp = as.numeric(newtimes < exp_censoring)
  event_ind_admin = as.numeric(newtimes < admin_cens) # Administrative censoring
  event_ind = as.numeric(event_ind_exp == event_ind_admin)
  newtimes = pmin(newtimes, exp_censoring)
  newtimes = pmin(newtimes, admin_cens)
  
  # Create dataset of covariates and survival times
  simulated_data = data_sample[, cov_names]
  simulated_data = simulated_data %>% add_column(t = newtimes, d = event_ind)
  
  return(simulated_data)
}

############------------ SIMULATE FROM COX METHODS ------------############

# Function to simulate r datasets using cox model
simulate_r_cox = function(real_data, nobs, r, admin_cens){
  n = nrow(real_data) # Full number of observations in real dataset

  # Sink true Cox
  true_cox_path = '../Results/true_models/Cox_true_model.txt'
  sink(true_cox_path)
  
  # Get covariate names
  cov_names = names(real_data)[!(names(real_data) %in% c('t', 'd', 'pid'))]
  # Factor the factor columns
  real_data = factor_data(real_data)
  
  # Get the event times
  events = real_data[real_data$d == 1, ]
  event_times = sort(unique(events$t)) # Get sorted unique times
  
  # Fit the cox model - fit once and simulate r times from this model for full sample (n)
  measurevar = 'Surv(t, d)'
  form = as.formula(paste(measurevar, paste(cov_names, collapse=' + '), sep=' ~ '))
  cox = coxph(form, data = real_data, x=TRUE)
  
  print(cox) # sink to real model txt file
  
  # Get predictions for every event time for every individual (n)
  ind_pred = real_data
  ind_pred$id = 1:n # Give individual's IDs
  ind_pred = subset(ind_pred, select = -c(t)) # Remove actual time
  ind_pred = ind_pred[rep(seq_len(n), each = length(event_times)),] # Repeat each individual
  ind_pred['t'] = rep(event_times, n) # Add in event times to each individual
  ind_pred$surv = predict(cox, newdata = ind_pred, type='survival')
  
  # Pivot to wide format (one column per event time)
  predictions = ind_pred %>% pivot_wider(id_cols=id, names_from = t, values_from = surv) %>%
    select(-id)
  
  # Turn into a matrix
  predictions = unname(as.matrix(predictions)) # unname = no column names from tibble
  
  # Save the predictions for performance measures
  filename = '../Results/true_models/true_predictions_cox.csv'
  write.csv(predictions, filename, row.names = FALSE)
  
  # Training Datasets
  count = 1
  while(count <= r){
    # Get simulated Data
    simulated_data = simulate_new_times(type='train', real_data, predictions, event_times, cov_names, admin_cens, nobs)
    filename = paste('../datasets/', nobs, '/cox/training/dataset_train_cox_', count, '.csv', sep='') # HPC
    write.csv(simulated_data, filename, row.names = FALSE)
    count = count + 1
  }
  
  # Testing Datasets
  count = 1
  while(count <= r){
    simulated_data = simulate_new_times(type='test', real_data, predictions, event_times, cov_names, admin_cens, nobs)
    filename = paste('../datasets/', nobs, '/cox/testing/dataset_test_cox_', count, '.csv', sep='') # HPC
    write.csv(simulated_data, filename, row.names = FALSE)
    count = count + 1
  }
  sink()
  return(cox)
}

############------------ SIMULATE FROM FP MODEL METHODS ------------############

# Function to simulate r datasets using Flexible Parametric Model
simulate_r_fp = function(real_data, nobs, r, admin_cens){
  n = nrow(real_data)
  # Sink true fp
  true_fp_path = '../Results/true_models/FP_true_model.txt'
  sink(true_fp_path)
  
  # Get the covariate names for the simulating method
  cov_names = names(real_data)[!(names(real_data) %in% c('t', 'd', 'pid'))]
  real_data = factor_data(real_data)
  
  # Get the event times
  events = real_data[real_data$d == 1, ]
  event_times = sort(unique(events$t)) # Get sorted unique times
  
  # Fit FP model
  measurevar = 'Surv(t, d)'
  form = as.formula(paste(measurevar, paste(cov_names, collapse=' + '), sep=' ~ '))
  fp = stpm2(form, data=real_data, df=3, se.fit=TRUE)
  print(fp) # sink to real model txt file
  print(summary(fp))
  
  # Get predictions for every event time for every individual
  ind_pred = real_data
  ind_pred$id = 1:n # Give individual's IDs
  ind_pred = subset(ind_pred, select = -c(t)) # Remove actual time
  ind_pred = ind_pred[rep(seq_len(n), each = length(event_times)),] # Repeat each individual
  ind_pred['t'] = rep(event_times, n) # Add in event times to each individual
  ind_pred$surv = predict(fp, newdata = ind_pred, type='surv', se.fit=FALSE)
  
  # Pivot to wide format (one column per event time)
  predictions = ind_pred %>% pivot_wider(id_cols=id, names_from = t, values_from = surv) %>%
    select(-id)
  
  # Turn into a matrix
  predictions = unname(as.matrix(predictions)) # unname = no column names from tibble
  
  # Save the predictions for performance measures
  filename = '../Results/true_models/true_predictions_fp.csv'
  write.csv(predictions, filename, row.names = FALSE)
  
  # Training Datasets
  count = 1
  while(count <= r){
    simulated_data = simulate_new_times(type='train', real_data, predictions, event_times, cov_names, admin_cens, nobs)
    filename = paste('../datasets/', nobs, 
                     '/fp/training/dataset_train_fp_', count, '.csv', sep='') # HPC
    write.csv(simulated_data, filename, row.names = FALSE)
    count = count + 1
  }
  
  # Testing Datasets
  count = 1
  while(count <= r){
    simulated_data = simulate_new_times(type='test', real_data, predictions, event_times, cov_names, admin_cens, nobs)
    filename = paste('../datasets/', nobs, 
                     '/fp/testing/dataset_test_fp_', count, '.csv', sep='') # HPC
    write.csv(simulated_data, filename, row.names = FALSE)
    count = count + 1
  }
  sink()
  return(form)
}



############------------ SIMULATE FROM MFP METHODS ------------############

# Function to simulate r datasets using mfp model
simulate_r_mfp = function(real_data, nobs, r, admin_cens){
  n = nrow(real_data)
  
  # Sink true mfp
  true_mfp_path = '../Results/true_models/MFP_true_model.txt'
  sink(true_mfp_path)
  
  # Get the covariate names for the simulating method
  cov_names = names(real_data)[!(names(real_data) %in% c('t', 'd', 'pid'))]
  real_data = factor_data(real_data)
  
  # Get continuous and factor covariate names
  cont_covs = real_data %>% select(which(sapply(., is.numeric)))
  cont_covs = names(cont_covs)[!(names(cont_covs) %in% c('t', 'd', 'pid'))]
  factor_covs = names(real_data %>% select(which(sapply(., is.factor))))
  
  # Get the event times
  events = real_data[real_data$d == 1, ]
  event_times = sort(unique(events$t))
  
  # Fit the mfp model - fit once and simulate r times from this model
  measurevar = 'Surv(t, d)'
  cont_string = paste('fp(', paste(cont_covs, collapse= ') + fp('), sep='')
  factor_string = paste(factor_covs, collapse=' + ')
  predictor = paste(cont_string, ') + ',factor_string, sep = '')
  form = paste(measurevar, predictor, sep=' ~ ')
  form = as.formula(form)
  mfp = mfp(form, family = 'cox', verbose=FALSE, data=real_data)
  coefficient_length = length(mfp$coefficients)
  print(mfp) # sink to real model txt file
  
  # Get predictions for every event time for every individual
  ind_pred = real_data
  ind_pred$id = 1:n # Give individual's IDs
  ind_pred = subset(ind_pred, select = -c(t)) # Remove actual time
  ind_pred = ind_pred[rep(seq_len(n), each = length(event_times)),] # Repeat each individual
  ind_pred['t'] = rep(event_times, n) # Add in event times to each individual
  expected_predictions = predict(mfp, newdata = ind_pred, type = 'expected')
  ind_pred$surv = exp(-expected_predictions) # Transform to survival
  
  # Pivot to wide format (one column per event time)
  predictions = ind_pred %>% pivot_wider(id_cols=id, names_from = t, values_from = surv) %>%
    select(-id)
  
  # Turn into a matrix
  predictions = unname(as.matrix(predictions)) # unname = no column names from tibble
  
  # Save the predictions for performance measures
  filename = '../Results/true_models/true_predictions_mfp.csv'
  write.csv(predictions, filename, row.names = FALSE)
  
  # Training Datasets
  count = 1
  while(count <= r){
    simulated_data = simulate_new_times(type='train', real_data, predictions, event_times, cov_names, admin_cens, nobs)
    filename = paste('../datasets/', nobs, 
                     '/mfp/training/dataset_train_mfp_', count, '.csv', sep='') # HPC
    write.csv(simulated_data, filename, row.names = FALSE)
    count = count + 1
  }
  
  # Testing Datasets
  count = 1
  while(count <= r){
    simulated_data = simulate_new_times(type='test', real_data, predictions, event_times, cov_names, admin_cens, nobs)
    filename = paste('../datasets/', nobs, 
                     '/mfp/testing/dataset_test_mfp_', count, '.csv', sep='') # HPC
    write.csv(simulated_data, filename, row.names = FALSE)
    count = count + 1
  }
  sink()
  return_list = list(form=form, coefficient_length=coefficient_length)
  return(return_list)
}



############------------ SIMULATE FROM RSF METHODS ------------############

# Random search to select hyperparameter values
random_search = function(rsf_data, grid, max_iter){
  # Results dataframe
  last_score = 1000 # Only storing the minimum to save space
  results = data.frame(iteration = numeric(0), nTrees = numeric(0), nodesize = numeric(0), Mtry = numeric(0), 
                       score = numeric(0))
  
  # Do max_iter iterations of random search
  i = 1
  while(i<=max_iter){
    # Get a random sample for each parameter
    hyperparameters = lapply(grid, sample, size=1)
    
    # Get covariates in the dataset
    covariates = rsf_data[, !names(rsf_data) %in% c('t', 'd', 'pid')]
                          
    # Fit the RSF with these hyperparameters and get oob error
    rsf = ranger(x = covariates, y = rsf_data[c('t', 'd')], data = rsf_data, num.trees=hyperparameters$nTrees, 
                 min.node.size=hyperparameters$nodesize, mtry=hyperparameters$Mtry, 
                 splitrule='logrank')
    
    score = tail(rsf$prediction.error, n=1) # OOB error

    if(score <= last_score){ # Save only minimum score
      # Save to results DF
      results[1, ] = c(i, hyperparameters$nTrees, hyperparameters$nodesize, hyperparameters$Mtry, score)
      last_score = score
    }
    
    i = i + 1
    rm(rsf)
  }
  
  return(results)
}

# Function to simulate r datasets using rsf
simulate_r_rsf = function(real_data, nobs, r, admin_cens, max_iter){
  n = nrow(real_data)
  
  # Sink true RSF
  true_rsf_path = '../Results/true_models/RSF_true_model.txt'
  sink(true_rsf_path)

  real_data = factor_data(real_data)

  # Get covariate names
  cov_names = names(real_data)[!(names(real_data) %in% c('t', 'd', 'pid'))]
  
  # Get event times
  events = real_data[real_data$d == 1, ]
  event_times = sort(unique(events$t)) # Get sorted unique times
  
  all_times = sort(unique(real_data$t))
  # Get indexes of event times in the list of all times to subset predictions
  event_times_indexes = match(event_times, all_times)
  
  # Change to data_frame so factor columns don't break
  rsf_data = data.frame(real_data)
  
  # Random search hyperparameter grid
  grid = list(nTrees = 50:1000, nodesize = 3:15, Mtry = 2:9)
  hyperparameters = random_search(rsf_data, grid, max_iter)
  print(hyperparameters) # sink to real model txt file
  
  covariates = rsf_data[, !names(rsf_data) %in% c('t', 'd', 'pid')] 
  
  # Fit RSF to dataset using optimised hyperparams - fit once and this forest used for all r simulated datasets
  rsf = ranger(x = covariates, y = rsf_data[c('t', 'd')], data = rsf_data, num.trees=hyperparameters$nTrees, 
               min.node.size=hyperparameters$nodesize, mtry=hyperparameters$Mtry, 
               splitrule='logrank')
  print(rsf)
  
  # Get survival predictions
  predictions = predict(rsf, data=rsf_data)
  predictions = as.data.frame(predictions$survival)
  predictions = unname(as.matrix(predictions))
  
  # Subset predictions to only event times
  predictions = predictions[, event_times_indexes]
  
  # Save the predictions for performance measures
  filename = '../Results/true_models/true_predictions_rsf.csv'
  write.csv(predictions, filename, row.names = FALSE)

  
  # Training Datasets
  count = 1
  while(count <= r){
    simulated_data = simulate_new_times(type='train', real_data, predictions, event_times, cov_names, admin_cens, nobs)
    filename = paste('../datasets/', nobs, 
                     '/rsf/training/dataset_train_rsf_', count, '.csv', sep='') # HPC
    write.csv(simulated_data, filename, row.names = FALSE)
    count = count + 1
  }

  # Testing Datasets
  count = 1
  while(count <= r){
    simulated_data = simulate_new_times(type='test', real_data, predictions, event_times, cov_names, admin_cens, nobs)
    filename = paste('../datasets/', nobs, 
                     '/rsf/testing/dataset_test_rsf_', count, '.csv', sep='') # HPC
    write.csv(simulated_data, filename, row.names = FALSE)
    count = count + 1
  }
  sink()
  
  return(event_times)
}
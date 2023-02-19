# All fitting and prediction getting methods
library(survival)
library(mfp)
library(rstpm2)
library(ranger)
library(tibble)
library(tidyverse)

library(reticulate)

use_python('/lustre/ahome3/h/hrs18/python_venvs/env_simulation/bin/python', required=TRUE)
use_virtualenv('~/python_venvs/env_simulation')
source_python('/home/h/hrs18/Simulation_Study/Simulation/Running_Simulation/CoxCC_Simulation.py')
source_python('/home/h/hrs18/Simulation_Study/Simulation/Running_Simulation/CoxTime_Simulation.py')

########### COX MODEL ############

# Fit the cox model to a dataset
fit_cox_model = function(train_dataset){
  # Get the covariate column names and create formula for cox model
  covnames = names(train_dataset)[!(names(train_dataset) %in% c('X', 't', 'd'))]
  measurevar = 'Surv(t, d)'
  form = as.formula(paste(measurevar, paste(covnames, collapse=' + '), sep=' ~ '))

  # Fit Cox model
  cox = coxph(form, data = train_dataset, x=TRUE)

  return(cox)
}

# Get predictions from Cox model
predict_cox_model = function(model, test_dataset, eval_times){
  n = nrow(test_dataset)

  # Get predictions for every eval_time for every individual
  ind_pred = test_dataset
  ind_pred$id = 1:n # Give individual's IDs
  ind_pred = subset(ind_pred, select = -c(t)) # Remove actual time
  ind_pred = ind_pred[rep(seq_len(n), each = length(eval_times)),] # Repeat each individual
  ind_pred['t'] = rep(eval_times, n) # Add in event times to each individual
  ind_pred$surv = predict(model, newdata = ind_pred, type='survival')

  # Pivot to wide format (one column per event time)
  full_predictions = ind_pred %>% pivot_wider(id_cols=id, names_from = t, values_from = surv) %>%
    select(-id)

  return(full_predictions)
}

############ FP MODEL ############

# Fit FP model
fit_fp_model = function(train_dataset){
  # Get the covariate column names and create formula for cox model
  covnames = names(train_dataset)[!(names(train_dataset) %in% c('X', 't', 'd'))]
  measurevar = 'Surv(t, d)'
  form = as.formula(paste(measurevar, paste(covnames, collapse=' + '), sep=' ~ '))

  # Fit FP model
  fp = stpm2(form, data=train_dataset, df=3)

  return(fp)
}

# Get predictions from FP model
predict_fp_model = function(model, test_dataset, eval_time){
  # Get predictions for eval_time
  n = nrow(test_dataset)

  # Get predictions for every eval_time for every individual
  ind_pred = test_dataset
  ind_pred$id = 1:n # Give individual's IDs
  ind_pred = subset(ind_pred, select = -c(t)) # Remove actual time
  ind_pred = ind_pred[rep(seq_len(n), each = length(eval_times)),] # Repeat each individual
  ind_pred['t'] = rep(eval_times, n) # Add in event times to each individual
  ind_pred$surv = predict(model, newdata = ind_pred, type='surv')

  # Pivot to wide format (one column per event time)
  full_predictions = ind_pred %>% pivot_wider(id_cols=id, names_from = t, values_from = surv) %>%
    select(-id)

  # Turn into a matrix
  predictions = unname(as.matrix(full_predictions))

  return(predictions)
}



############ MFP MODEL ############

# Get fit and predictions from MFP model - has to be same method or it goes crazy
predict_mfp_model = function(train_dataset, test_dataset, eval_times){

  # Get the covariate column names and create formula for mfp model
  cont_covs = c('age', 'pr', 'er', 'nodes')
  covnames = names(train_dataset)[!(names(train_dataset) %in% c('X', 't', 'd', 'age','nodes', 'pr', 'er'))]
  measurevar = 'Surv(t, d)'
  string = paste('fp(', paste(cont_covs, ', scale = TRUE', collapse= ') + fp('), sep='')
  string = paste(string, ') + ', paste(covnames, collapse = ' + '))
  string = paste(measurevar, string, sep=' ~ ')

  form = as.formula(string)

  # Fit mfp model - catch any warnings/error
  tryCatch({
    mfp_model = mfp(form, data=train_dataset, family = cox)
  }, warnings = function(cond){
  }, error = function(cond){
  })
  
  # Try to get formula from mfp and catch error from any non-convergence
  try_mfp = function(){
    tryCatch(
      { # Fit mfp model
        result_formula = mfp_model$formula
        worked = TRUE
      }, 
      error = function(cond){
        print('into error')
        FALSE
      })
  }
  
  # If there is no mfp model due to non-convergence, predictions are NULL and performance measures will be NA
  if(try_mfp()==TRUE){
    result_formula = mfp_model$formula
  
    # Fit this formula in Cox model
    cox_mfp = coxph(result_formula, data=train_dataset, x=TRUE, model=TRUE)
  
    # Test data
    n = nrow(test_dataset)
  
    # Get predictions for every eval_time for every individual
    ind_pred = test_dataset
    ind_pred$id = 1:n # Give individual's IDs
    ind_pred = subset(ind_pred, select = -c(t)) # Remove actual time
    ind_pred = ind_pred[rep(seq_len(n), each = length(eval_times)),] # Repeat each individual
    ind_pred['t'] = rep(eval_times, n) # Add in event times to each individual
    ind_pred$surv = predict(cox_mfp, newdata = ind_pred, type='survival')
  
    # Pivot to wide format (one column per event time)
    full_predictions = ind_pred %>% pivot_wider(id_cols=id, names_from = t, values_from = surv) %>%
      select(-id)
  
    # Turn into a matrix
    predictions = unname(as.matrix(full_predictions))
  } else {
    predictions = NULL
  }
  return(predictions)
}



############ RSF MODEL ############

random_search_R = function(train_dataset, grid, max_iter){
  # Results dataframe
  last_score = 1000 # Only storing the minimum to save space
  results = data.frame(iteration = numeric(0), nTrees = numeric(0), nodesize = numeric(0), Mtry = numeric(0), score = numeric(0))
  # Do max_iter iterations
  i = 1
  while(i<=max_iter){
    # Get a random sample for each parameter
    hyperparameters = lapply(grid, sample, size=1)
    # Fit the network with these hyperparameters and get oob error
    rsf = ranger(x = train_dataset[1:9], y = train_dataset[c('t', 'd')], num.trees=hyperparameters$nTrees,
                 min.node.size=hyperparameters$nodesize, mtry=hyperparameters$Mtry,
                 splitrule='logrank')
    score = tail(rsf$prediction.error, n=1)
    if(score <= last_score){
      # Save to results DF
      results[1, ] = c(i, hyperparameters$nTrees, hyperparameters$nodesize, hyperparameters$Mtry, score)
      last_score = score
    }
    rm(rsf)
    i = i + 1
  }
  return(results)
}

# Fit RSF model
fit_rsf_model = function(train_dataset, nobs){
  # Random search hyperparameter grid
  max_iter = 40
  grid = list(nTrees = 50:500, nodesize = 3:15, Mtry = 2:9)
  hyperparameters = random_search_R(train_dataset, grid, max_iter)

  filename = paste('/home/h/hrs18/Simulation_Study/Simulation/simulation-results/', nobs,
                   '/Hyperparameters_RSF.csv', sep='')
  file = file(filename,open="at")
  hp = as.data.frame(hyperparameters)
  write.table(hp, filename, sep=',', append=TRUE, col.names=FALSE)
  close(file)

  # Default parameters
  #hyperparameters = list(nTrees = 500, min.node.size = 3, mtry = 3, splitrule='logrank')

  # Grow a random forest
   rsf = ranger(x = train_dataset[1:9], y = train_dataset[c('t', 'd')], num.trees=hyperparameters$nTrees,
               min.node.size=hyperparameters$nodesize, mtry=hyperparameters$Mtry,
               splitrule='logrank')

  return(rsf)
}

# Get predictions from RSF model
predict_rsf_model = function(model, test_dataset, eval_times){
  # Get survival predictions for new dataset at time t
  n = nrow(test_dataset)
  rsf_predict = predict(model, data=test_dataset)
  predictions = as.data.frame(rsf_predict$survival)
  times = rsf_predict$unique.death.times

  i=1
  indexes = c()
  while(i<=length(eval_times)){
    eval_time = eval_times[i]
    # Find the index of the closest event time to time t
    index_t = which.min(abs(times - eval_time))
    indexes[i] = index_t
    i = i + 1
  }

  # Get the predictions at time t
  predictions = predictions[, indexes]

  return(predictions)
}



############ CoxCC MODEL ############

# Fit CoxCC model
fit_CoxCC_model = function(train_dataset, nobs){
  model = fit_CoxCC(train_dataset, nobs)

  return(model)
}

# Get predictions from CoxCC model
predict_CoxCC_model = function(model, test_dataset, eval_times){
  predictions = predict_CoxCC(model, test_dataset, eval_times)

  return(predictions)
}


############ COX TIME MODEL ############

# Fit CoxTime model
fit_CoxTime_model = function(train_dataset, nobs){
  model = fit_CoxTime(train_dataset, nobs)

  return(model)
}

# Get predictions from CoxTime model
predict_CoxTime_model = function(model, test_dataset, eval_times){
  predictions = predict_CoxTime(model, test_dataset, eval_times)
  return(predictions)
}

# File for all performance measure methods
library(riskRegression)

# Time-dependent AUC
auc = function(predictions, test_dataset, eval_times){
  predictions = 1 - predictions
  predictions = as.matrix(predictions)

  auc_value = Score(list(predictions), formula=Surv(t,d)~1, data=test_dataset, metrics='AUC', times=eval_times, verbose=TRUE)
  auc_value = unlist(auc_value)

  auc_values = list()
  i = 1 # Get all the values of the AUC for each time point
  if(length(eval_times)>1){
    while(i<=length(eval_times)){
      auc = get(paste('AUC.score.AUC', i, sep = ''), auc_value)
      auc_values[i] = auc
      i = i + 1
    }
  } else {
    auc_values = auc_value$AUC.score.AUC
  }
  
  return(auc_values)
}

# MSE
mse = function(predictions, true_predictions, true_event_times, eval_times){
  predictions = as.matrix(predictions)
  mses = list()
  i = 1
  while(i<=length(eval_times)){
    eval_time = eval_times[i]
    # Find the index of the closest true_event time to eval_time
    index_t = which.min(abs(true_event_times - eval_time))
    # Get the true st at the closest time
    true_st = true_predictions[, index_t]

    # Get the predictions at that eval_time
    if(length(eval_times) == 1){
      t_predictions = predictions
    } else {
      t_predictions = predictions[, i]
    }

    mse = mean((t_predictions - true_st)^2)
    mses[i] = mse
    i = i + 1 
  }
  
  return(mses)
}

# MAE
mae = function(predictions, true_predictions, true_event_times, eval_times){
  predictions = as.matrix(predictions)
  
  maes = list()
  i = 1
  while(i<=length(eval_times)){
    eval_time = eval_times[i]
    # Find the index of the closest true_event time to eval_time
    index_t = which.min(abs(true_event_times - eval_time))
    # Get the true st at the closest time
    true_st = true_predictions[, index_t]
    
    # Get the predictions at that eval_time
    if(length(eval_times) == 1){
      t_predictions = predictions
    } else {
      t_predictions = predictions[, i]
    }
    # MAE
    difference = abs(t_predictions - true_st)
    mae = mean(difference)

    maes[i] = mae
    i = i + 1 
  }

  return(maes)
}

# Brier Score
brier_score = function(predictions, test_dataset, eval_times){
  predictions = as.matrix(1-predictions)
  
  brier_value = Score(list(predictions), formula=Surv(t,d)~1, data=test_dataset, metrics='Brier', times=eval_times, verbose=TRUE)
  brier_value = unlist(brier_value)

  brier_values = list()
  i = length(eval_times) + 1 # Get all the values of the Brier for each time point (starts with i null model values)
  j = 1 # eval_time list length
  while(j<=length(eval_times)){
    brier = get(paste('Brier.score.Brier', i, sep = ''), brier_value)
    brier_values[j] = brier
    i = i + 1
    j = j + 1
  }
  return(brier_values)
}

# Cox-Time Method
import numpy as np
import pandas as pd
import math
import random
import torch
import torchtuples as tt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

real_data = pd.read_csv('/home/h/hrs18/Simulation_Study/datasets/Real_Datasets/RGBC_data_sorted.csv')
n = len(real_data)
admin_cens = 15

nobs = 2982
r = 2000
max_RS_iter = 40

############------------ Sort the Dataset to train and validation and standardize ------------############

def sort_data_standardise(dataset, train=True):
    # Make sure variables are float 32 - this is required by torch
    dataset = dataset.astype(np.float32)

    df_train = dataset

    # Standardise the continuous covariates and leave the factors - all need to be of type float32 though
    continuous_cols = ['age', 'pr', 'er', 'nodes']
    factor_cols = ['meno', 'size', 'grade', 'hormon', 'chemo']
    standardize = [([col], StandardScaler()) for col in continuous_cols]
    leave = [(col, None) for col in factor_cols]
    x_mapper = DataFrameMapper(standardize + leave)
    x_train = x_mapper.fit_transform(df_train).astype('float32')

    # Get the target variables e.g. event time and indicator
    get_target = lambda df: (df['t'].values, df['d'].values)

    labtrans = CoxTime.label_transform() # Labels and durations also need to be float32 arrays
    y_train = labtrans.fit_transform(*get_target(df_train)) # standardise durations too

    if train == True:
    # Get validation set
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)
        x_val = x_mapper.transform(df_val).astype('float32')
        y_val = labtrans.transform(*get_target(df_val))
        val = tt.tuplefy(x_val, y_val)
        # Use tupletree to repeat the validation set multiple times - use this to get variance of validation loss
        val.repeat(2).cat().shapes()
    else:
    # We only want the whole dataset for simulating new times
        val = 0

    return x_train, y_train, val, labtrans


# Method to fit neural network within random search
def fit_CoxTime_RS(train_dataset, hyperparameters):
    # Get train x, train y and validation set - standardized
    x_train, y_train, val, labtrans = sort_data_standardise(train_dataset, train=True)

    ############------------ Fit the neural network ------------############
    layers = hyperparameters['hidden layers']
    nodes = hyperparameters['number of nodes']
    learning_rate = hyperparameters['learning rate']
    lam = hyperparameters['lambda penalty']

    # MLP, two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1] # Creates the input nodes - one for each covariate
    num_nodes = np.full(layers, nodes).tolist()
    batch_norm = True
    dropout = hyperparameters['drop out']
    output_bias = False
    activation = torch.nn.ReLU
    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm,
                                  dropout, activation)

    # Train the neural network
    # Use the Adam optimizer, manual learning rate to 0.01, batch size is 256, early stopping and 512 epochs
    model = CoxTime(net, tt.optim.Adam, shrink=lam, labtrans=labtrans)
    batch_size = 2982
    model.optimizer.set_lr(learning_rate)
    epochs = 500
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val.repeat(10).cat())
    score = model.partial_log_likelihood(*val).mean()

    return score

############------------ Random Search Functions ------------############

def random_search(train_dataset, grid, max_iter):
    cols = list(grid.keys()) # Get the hyperparameter names we are tuning for DF cols
    cols.append('Score')
    # Results DF
    results = pd.DataFrame(columns = cols,
                            index = list(range(max_iter)))

    # For each iteration, select and evaluate random hyperparameter values
    for i in range(max_iter):
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in grid.items()}
        score = objective_function(train_dataset, hyperparameters)
        results['Score'][i] = score
        results['hidden layers'][i] = hyperparameters['hidden layers']
        results['number of nodes'][i] = hyperparameters['number of nodes']
        results['drop out'][i] = hyperparameters['drop out']
        results['learning rate'][i] = hyperparameters['learning rate']
        results['lambda penalty'][i] = hyperparameters['lambda penalty']

    #results.to_csv('/home/h/hrs18/Simulation_Study/simulation-coding/simulation-results/CoxTime_RS_Iterations.csv')
    # Select and return minimum partial log likelihood
    results = results.sort_values('Score')
    minimum = results.iloc[0]

    return minimum

# Getting the score
def objective_function(train_dataset, hyperparameters):
    # Get the partial log likelihood for the validation set
    score = fit_CoxTime_RS(train_dataset, hyperparameters)
    return score

############------------ Simulating New Datasets ------------############
def simulate_new_times(model, surv_numpy, nobs, real_data, admin_cens, event_times, type):
    n = len(real_data)
    # Testing data - want the whole sample size
    if type == 'test':
        nobs = n
        data_sample = real_data
    else: # Training sample, we want nobs sample size
        rows = random.choices(range(0, n), k=nobs)
        data_sample = real_data.iloc[rows]
        predictions = pd.DataFrame(surv_numpy)
        predictions = predictions.iloc[:, rows] # Columns are each indvidual in predictions
        surv_numpy = predictions.to_numpy()

    # Simulate from uniform
    U = np.random.uniform(0,1,nobs)

    # Find S(t) closest to U and corresponding time
    abs_values = np.absolute(surv_numpy-U)
    abs_values = pd.DataFrame(abs_values)
    times = abs_values[::-1].idxmin(axis=0).to_numpy() # process rows in reverse order to find last occurence of minimum

    newtimes = []
    i = 0
    # Get the survival time for each individual
    while(i<nobs):
        index = times[i] # Get the index of the closest S(t) to U for individual_i
        surv_i = surv_numpy[index][i] # Getting the survival probability corresponding to that time

        # Get the event times at which S(ti) > U > S(ti+1)
        if U[i] > surv_i and index != 0:
            if surv_i == surv_numpy[index-1][i]:
                new_time = event_times[index]
            else:
                func = interp1d([surv_numpy[index-1][i], surv_i], [event_times[index-1], event_times[index]], kind='linear')
                new_time = func(U[i])
        elif U[i] > surv_i and index == 0:
            func = interp1d([1, surv_i], [0, event_times[index]])
            new_time = func(U[i])
        elif U[i] < surv_i and index != len(event_times)-1:
            if surv_i == surv_numpy[index+1][i]:
                new_time = event_times[index]
            else:
                func = interp1d([surv_i, surv_numpy[index+1][i]], [event_times[index], event_times[index+1]], kind='linear')
                new_time = func(U[i])
        elif U[i] < surv_i and event_times[index] == np.amax(event_times):
            new_time = np.random.uniform(event_times[index], admin_cens+1)
        else:
            new_time = event_times[index]

        newtimes.append(new_time)
        i += 1

    # Exponential and admin censoring
    lam = 0.01
    U = np.random.uniform(0,1,nobs)
    exp_cens_array = -np.log(U)/lam
    #exp_cens_array = np.random.exponential(scale = 30, size = nobs) # scale = beta, lambda = 0.15
    event_indicator_exp = np.less(newtimes, exp_cens_array)
    event_indicator_admin = np.less(newtimes, admin_cens)
    event_indicator = np.logical_and(event_indicator_exp, event_indicator_admin).astype(int)
    newtimes = np.minimum(newtimes, exp_cens_array)
    newtimes = np.minimum(newtimes, admin_cens)

    # Create new dataset with simulated times
    cov_names = ['age', 'pr', 'er', 'nodes', 'meno', 'size', 'grade', 'hormon', 'chemo']
    simulated_data = data_sample[cov_names].copy()
    simulated_data['t'] = newtimes
    simulated_data['d'] = event_indicator

    return simulated_data


############------------ Simulate r datasets ------------############
def simulate_r_datasets(real_data, nobs, r, admin_cens, max_RS_iter):
    # Get unique event times
    event_times = real_data[real_data['d']==1]
    event_times = np.around(np.unique(event_times['t']), decimals=3)

    # Get train x, train y and validation set - standardized
    x_train, y_train, val, labtrans = sort_data_standardise(real_data, train=True)

    # Get hyper parameters from random Search

    max_iter = max_RS_iter
    # Dictionary of hyperparameters and domains
    grid = {'hidden layers': list(range(1,5)),
            'number of nodes': list(range(5, 50)),
            'drop out': list([0.1, 0.2, 0.3, 0.4, 0.5]),
            'learning rate': list([0.01, 0.01]),
            'lambda penalty': list([0, 0.0001, 0.001, 0.01, 0.1])
            }

    # Results of random search
    results = random_search(real_data, grid, max_iter)
    hyperparameters = {'hidden layers': results['hidden layers'],
                        'number of nodes': results['number of nodes'],
                        'drop out': results['drop out'],
                        'learning rate': results['learning rate'],
                        'lambda penalty': results['lambda penalty']}

    results.to_csv('/home/h/hrs18/Simulation_Study/Simulation/simulation-results/true_models/CoxTime_true_model.txt')

    ############------------ Fit the neural network ------------############
    layers = hyperparameters['hidden layers']
    nodes = hyperparameters['number of nodes']
    learning_rate = hyperparameters['learning rate']
    lam = hyperparameters['lambda penalty']

    # MLP, two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1] # Creates the input nodes - one for each covariate
    num_nodes = np.full(layers, nodes).tolist()
    batch_norm = True
    dropout = hyperparameters['drop out']
    activation = torch.nn.ReLU
    output_bias = False

    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm,
                                  dropout, activation)

    # Train the neural network
    # Use the Adam optimizer, manual learning rate to 0.01, batch size is 256, early stopping and 512 epochs
    model = CoxTime(net, tt.optim.Adam, shrink=lam, labtrans=labtrans)
    batch_size = 256
    model.optimizer.set_lr(learning_rate)
    epochs = 500
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val.repeat(10).cat())

	# Save model parameters to csv filename
    model.save_net('/home/h/hrs18/Simulation_Study/Simulation/simulation-results/true_models/saved_CoxTime')
    weights_save = torch.load('/home/h/hrs18/Simulation_Study/Simulation/simulation-results/true_models/saved_CoxTime.pt')
    params = weights_save.named_parameters()
    params = pd.DataFrame(params)

    params.to_csv('/home/h/hrs18/Simulation_Study/Simulation/simulation-results/true_models/CoxTime_weights_biases.csv', index=False)

    # Standardise the dataset but don't split into validation
    x_full, y_full, val, labtrans = sort_data_standardise(real_data, train=False)

    # Get predictions for same real dataset
    _ = model.compute_baseline_hazards() # Compute the baseline
    surv = model.predict_surv_df(x_full) # Get survival probabilities
    surv['times'] = surv.index
    times = np.around(surv['times'].to_numpy(), decimals=3)

    #event_times = surv.index.values # Get the event times
    surv_numpy = surv.to_numpy()
    true = np.isin(times, event_times)
    surv_indexes = np.where(np.isin(times, event_times))
    surv_numpy = np.squeeze(surv_numpy[surv_indexes,:], axis=0)
    surv_numpy = np.delete(surv_numpy, -1, axis=1)

    # PD dataframe of numpy surv
    surv = pd.DataFrame(surv_numpy)

    # Write true predictions to file for performance measures in simulation
    filename = '/home/h/hrs18/Simulation_Study/Simulation/simulation-results/true_models/true_predictions_CoxTime.csv'
    pd.DataFrame(surv_numpy).to_csv(filename, index=False, header=False)

    # Training Datasets
    count = 1
    while count <= r:
        simulated_data = simulate_new_times(model, surv_numpy, nobs, real_data, admin_cens, event_times, type='train')
        filename = '/home/h/hrs18/Simulation_Study/datasets/RGBC/' + str(nobs) + '/CoxTime/training/dataset_train_CoxTime_' + str(count) + '.csv'
        simulated_data.to_csv(filename, index=False)
        count += 1

    # Testing Datasets
    count = 1
    while count <= r:
        simulated_data = simulate_new_times(model, surv_numpy, nobs, real_data, admin_cens, event_times, type='test')
        filename = '/home/h/hrs18/Simulation_Study/datasets/RGBC/' + str(nobs) + '/CoxTime/testing/dataset_test_CoxTime_' + str(count) + '.csv'
        simulated_data.to_csv(filename, index=False)
        count += 1

    return model

simulate_r_datasets(real_data, nobs, r, admin_cens, max_RS_iter)

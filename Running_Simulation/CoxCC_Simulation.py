# COX-CC script

#import numpy as np
import pandas as pd
import math
import random
import torch
import torchtuples as tt
from pycox.models import CoxCC # Kvamme PH cox - continuous time method
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

############------------ Sort the Dataset to train and validation and standardize ------------############

def sort_data_standardise_CoxCC(dataset, train=True):
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
    y_train = get_target(df_train)

    if train == True:
    # Get validation set
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)
        x_val = x_mapper.transform(df_val).astype('float32')
        y_val = get_target(df_val)
        val = tt.tuplefy(x_val, y_val)
        # Use tupletree to repeat the validation set multiple times - use this to get variance of validation loss
        val.repeat(2).cat().shapes()
    else:
    # We only want the whole dataset for simulating new times
        val = 0

    return x_train, y_train, val

# Method to fit neural network within random search
def fit_CoxCC_RS(train_dataset, hyperparameters):
    nobs = len(train_dataset)

    # Get train x, train y and validation set - standardized
    x_train, y_train, val = sort_data_standardise_CoxCC(train_dataset, train=True)

    ############------------ Fit the neural network ------------############
    layers = hyperparameters['hidden layers']
    nodes = hyperparameters['number of nodes']
    learning_rate = hyperparameters['learning rate']
    lam = hyperparameters['lambda penalty']

    # MLP, two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1] # Creates the input nodes - one for each covariate
    num_nodes = np.full(layers, nodes).tolist()
    out_features = 1
    batch_norm = True
    dropout = hyperparameters['drop out']
    activation = torch.nn.ReLU
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, activation, output_bias=output_bias)

    # Train the neural network
    # Use the Adam optimizer, manual learning rate to 0.01, batch size is 256, early stopping and 512 epochs
    model = CoxCC(net, tt.optim.Adam, shrink=lam)
    batch_size = nobs # Full sample for RS training batch
    model.optimizer.set_lr(learning_rate)
    epochs = 500
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val.repeat(10).cat())
    score = model.partial_log_likelihood(*val).mean()

    del model

    return score

############------------ Random Search Functions ------------############

def random_search(train_dataset, grid, max_iter):
    #drop_out = 0.5
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
        results['learning rate'][i] = hyperparameters['learning rate'] # Manually set
        results['lambda penalty'][i] = hyperparameters['lambda penalty']

    # Select and return minimum partial log likelihood
    results = results.sort_values('Score')

    return results # was minimum

# Getting the score
def objective_function(train_dataset, hyperparameters):
    # Get the partial log likelihood for the validation set
    score = fit_CoxCC_RS(train_dataset, hyperparameters)
    return score


# Method to the fit the neural network
def fit_CoxCC(train_dataset, nobs):
    nobs = len(train_dataset)

    # Get train x, train y and validation set - standardized
    x_train, y_train, val = sort_data_standardise_CoxCC(train_dataset, train=True)


    # Get hyper parameters from random Search
    max_iter = 40
    # Dictionary of hyperparameters and domains
    grid = {'hidden layers': list(range(1,5)),
            'number of nodes': list(range(5, 50)),
            'drop out': list([0.1, 0.2, 0.3, 0.4, 0.5]),
            'learning rate': list([0.01, 0.01]),
            'lambda penalty': list([0, 0.0001, 0.001, 0.01, 0.1])
            }


    # Results of random search - get minimum and append to file
    results = random_search(train_dataset, grid, max_iter)
    results = results.head(1)
    # Append selected parameters to file
    filename = '~/Simulation_Study/Simulation/simulation-results/' + str(nobs) + '/Hyperparameters_CoxCC.csv'
    results.to_csv(filename, mode='a', header=False)

    # Get minimum as dict for rest of simulation
    results = results.iloc[0]

    hyperparameters = {'hidden layers': results['hidden layers'],
                        'number of nodes': results['number of nodes'],
                        'drop out': results['drop out'],
                        'learning rate': results['learning rate'],
                        'lambda penalty': results['lambda penalty']}
    '''


    # Default Cox-CC hyperparameters
    hyperparameters = {'hidden layers': 2,
                        'number of nodes': 32,
                        'drop out': 0.1,
                        'learning rate': 0.01,
                        'lambda penalty': 0.1}




    # hyperparameters selected from DGM
    hyperparameters = {'hidden layers': 1,
                        'number of nodes': 33,
                        'drop out': 0.9081632653061225,
                        'learning rate': 0.41432138642734206,
                        'lambda penalty': 0.01}

    '''
    # Write Rs to file
    #filename = '/home/h/hrs18/Simulation_Study/Simulation/simulation-results/CoxCC_Checking/RS_results.csv'
    #pd.DataFrame(results).to_csv(filename, index=False, header=False)

    ############------------ Fit the neural network ------------############
    layers = hyperparameters['hidden layers']
    nodes = hyperparameters['number of nodes']
    learning_rate = hyperparameters['learning rate']
    lam = hyperparameters['lambda penalty']

    # MLP, two hidden layers, ReLU activations, batch norm and dropout
    in_features = x_train.shape[1] # Creates the input nodes - one for each covariate
    num_nodes = np.full(layers, nodes).tolist()
    out_features = 1
    batch_norm = True
    dropout = hyperparameters['drop out']
    activation = torch.nn.ReLU
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, activation, output_bias=output_bias)

    # Train the neural network
    # Use the Adam optimizer, manual learning rate to 0.01, batch size is 256, early stopping and 512 epochs
    model = CoxCC(net, tt.optim.Adam, shrink=lam)
    batch_size = 256 # Default
    model.optimizer.set_lr(learning_rate)
    epochs = 500 # Default
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = False

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val.repeat(10).cat())

    return model


# Method to get predictions from CoxCC for test dataset
def predict_CoxCC(model, test_dataset, eval_times):
    if(isinstance(eval_times, float) or isinstance(eval_times, int)): # If we only have one eval time, put it in an array so it has len()
        eval_times = [eval_times]

    x_test, y_test, val = sort_data_standardise_CoxCC(test_dataset, train=False)
    _ = model.compute_baseline_hazards()
    predictions_model = model.predict_surv_df(x_test)
    event_times = predictions_model.index.values
    predictions = predictions_model.to_numpy()

    # Find event times closest to eval_time and corresponding S(t) prediction
    i = 0
    indexes = []
    while(i<len(eval_times)):
        eval_time = eval_times[i]
        abs_values = np.absolute(event_times-eval_time)
        index_evaltime = np.argmin(abs_values)
        indexes.append(index_evaltime)
        i += 1
    predictions = predictions[indexes, :] # S(t) for every individual at time closest to eval_time
    predictions = predictions.transpose()
    predictions = pd.DataFrame(predictions)

    del model

    return predictions #, predictions_model

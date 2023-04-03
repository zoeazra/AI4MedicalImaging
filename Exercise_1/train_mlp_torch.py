################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022 & Oliver Gurney-Champion | Spring 2023
# Date modified: Jan 2023
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import random
import numpy as np
import Data_loader as dl
import MLP_Torch
from MLP_Torch import MLP
from train_mlp_numpy import error_metrics, plot_results
import torch
import torch.nn as nn


def train(hidden_dims, lr, batch_size, epochs, seed,
          bvalues=[0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 300, 500, 700, 850, 1000],
          optimizer_option='adam'):
    """
    Performs a full training cycle of MLP model.
    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.
    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')
    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # probe available devices
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Loading the dataset
    data_sim = dl.sim_signal(bvalues=bvalues,sims=200000,seed=np.random.randint(1,10000))
    # data_sim[0] = data; data_sim[1]=D; data_sim[2]=f and data_sim[3]=Dp
    #!!!note we now produce 1 big dataset which you will have to split between training and validation -->
    # do this using PyTorch data loaders.!!!

    model = MLP(np.size(data_sim[0],1), hidden_dims, 1).to(device)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Initialize loss module

    # TODO: split data. Then, train loader loads the trianing data. We want to shuffle to make sure data order is modified each epoch and
    #  different data is selected each epoch.

    # TODO: validation data is loaded here. By not shuffling, we make sure the same data is loaded for validation every time.
    #  We can use substantially more data per batch as we are not training.

    # TODO: initialize optimizer

    # loop over epochs
    for a in range(epochs):
        print("starting epoch "+str(a))
        # put your model in the trianing state
        model.train()
        # loop over batches
        for x in trainloader:
            batch=x[0].to(device)
            D_ref = x[1].to(device)
            f_ref = x[2].to(device)
            Dp_ref = x[3].to(device)
            # TODO: run model forward; define loss; propogate loss backward; update weights
        # TODO: generate loss curve to visualize training progress
        model.eval()
        for x in inferloader:
            batch=x[0].to(device)
            D_ref = x[1].to(device)
            f_ref = x[2].to(device)
            Dp_ref = x[3].to(device)
            #TODO: run model forward and calculate validation loss
        # TODO: generate validation loss curve to visualize training progress


        plot_results(D_ref, f_ref, Dp_ref, loss_train=loss_curve, f_pred=f_pred.detach().numpy(), data=batch,
                     bvalues=bvalues,val_loss=loss_curve_val)
    SD, sys = error_metrics(f_pred.cpu().detach().numpy(),f_ref.cpu().detach().numpy())

    print('the systematic error is ' + str(sys) + ' and the random error is ' + str(SD))

    #######################
    # END OF YOUR CODE    #
    #######################
    return model



def train_self_supervised(data, hidden_dims, lr, batch_size, epochs, seed,
          bvalues=[0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 300, 500, 700, 850, 1000],
          optimizer_option='sgd'):
    """
    Performs a full training cycle of MLP model.
    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.
    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')
    Hint: you can save your best model by deepcopy-ing it.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    #######################
    # END OF YOUR CODE    #
    #######################
    return model


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float, nargs='+',
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--optimizer_option', default='sgd', type=str,
                        help='optimizer options: adam and sgd (default)')
    # Other hyperparameters
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
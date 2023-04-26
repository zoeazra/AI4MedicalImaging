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
# Author: Deep Learning for Medical Imaging Amsterdam UMC Oliver Gurney-Champion | Spring 2023
# adapted from Deep Learning Course (UvA) | Fall 2022
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
from MLP_numpy import MLP
from modules import MSE
from matplotlib import pyplot as plt
from Data_loader import ivim


def train(hidden_dims, lr, batch_size, epochs, seed, parallel=True, bvalues=[0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 300, 500, 700, 850, 1000]):
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
    ## Loading the dataset
    data_sim, D, f, Dp = dl.sim_signal(bvalues=bvalues,sims=2000000,seed=np.random.randint(1,10000))
    data_sim_ev, D_val, f_val, Dp_val = dl.sim_signal(bvalues=bvalues,sims=10000,seed=np.random.randint(1,10000))
    data_sim_ev=np.transpose(data_sim_ev)
    model = MLP(np.size(data_sim, 1), hidden_dims, 1)

    #######################
    # PUT (and edit) YOUR CODE HERE  #
    #######################
    # TODO: Initialize loss modules. You will also want to generate lists to save the loss values over epochs for plotting training progress

    # loop over epochs

    for a in range(epochs):
        # setup random shuffle of data to ensure random data selection per batch
        print("starting epoch "+str(a))
        batchnums=list(range(len(D)))
        random.shuffle(batchnums)
        # TODO: reset losses

        # loop over batches -->
        # TODO: add your code for training the network
        for i in range(int(np.floor(len(batchnums)/batch_size))):
            # the code selects the batch data and ground truth references
            batch=batchnums[i*batch_size:i*batch_size+batch_size]
            x = np.transpose(data_sim[batch])
            D_ref = D[batch]
            f_ref = f[batch]
            Dp_ref = Dp[batch]
            # TODO: put your data through the network, calculate the loss, backpropagate the loss and update weights,
            # TODO: don't forget to save logging info on the loss
            #  loss_curve containing a list fo the total loss each epoch


        # TODO: evaluate validation data data_sim_ev (can be done in a loop, or all validation can be put through the network in 1 go
        #  loss_curve_val containing a list fo the total validation loss each epoch

        # TODO: save losses such that you can plot progres using plot_results below. Note there are many options in plot
        #  results that you may want to utalize if you do additional exercises. But for the basic exercise you only need
        #  to options below.
        # plot your results. Note you will need to generate the different inputs throughout your code for the plotting to work.
        # Note that f_pred and D_ref, f_ref, and Dp_ref should be the last values of your validation dataset (which will be plotted)
        plot_results(D_ref, f_ref, Dp_ref, loss_train=loss_curve, f_pred=f_pred, data=np.transpose(x),
                     bvalues=bvalues, val_loss=loss_curve_val)
    #######################
    # END OF YOUR CODE    #
    #######################

    # final evaluation: f_pred_val=the f_pred list from the final validation data and f_val is the reference of your validation
    SD, sys = error_metrics(f_pred_val,f_val)
    print('the systematic error is ' + str(sys) + ' and the random error is ' + str(SD))

    return model


def error_metrics(par_pred,par_ref):
    """
    Computes the random and systematic errors from the prediction.
    Args:
      par_pred: 1D float array of size [n], predictions of the model
      par_ref: 1D float array of size [n]. Ground truth reference for each sample
    Returns:
      CV: random error (coefficient of variation)
      Sys: systematic error
    """

    CV=np.std(par_pred-par_ref)/np.mean(par_ref)
    sys = np.mean(par_pred-par_ref)/np.mean(par_ref)

    return CV, sys


def plot_results(D_ref,f_ref,Dp_ref,D_pred=None,f_pred=None,Dp_pred=None, data=None,
                 bvalues=[0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 300, 500, 700, 850, 1000],
                 loss_train=None,train_loss_D=None,train_loss_f=None, train_loss_Dp=None,val_loss=None,
                 val_loss_curve_D=None,val_loss_curve_f=None,val_loss_curve_Dp=None):
    """
        This program can help visualize the progress of training and network performance.
        Args:
          Dref: An array of the ground trueth reference values for D [n].
          fref: An array of the ground trueth reference values for f [n].
          Dpref: An array of the ground trueth reference values for Dp [n].
          D_pred (optional): An array containing the corresponding D preditions from the network [n]] --> only used for bonus exercise
          f_pred (optional):  An array containing the corresponding f preditions from the network [n]
          Dp_pred (optional):  An array containing the corresponding Dp preditions from the network [n] --> only used for bonus exercise
          data (optional): An array containing the input data [n x m]
          bvalues (optional): An array containing the b-values [m]
          loss_train (optional): An array with total loss values during training [epochs]
          train_loss_D (optional): An array with loss values specific to D [epochs] --> only used for bonus exercise
          train_loss_f (optional): An array with loss values specific to f [epochs] --> only used for bonus exercise
          train_loss_Dp (optional): An array with loss values specific to Dp [epochs] --> only used for bonus exercise
          val_loss (optional):  An array with total loss values based on the validation data during training [epochs] --> only used for bonus exercise
          val_loss_curve_D (optional): An array with validation loss values specific to D [epochs] --> only used for bonus exercise
          val_loss_curve_f (optional): An array with validation loss values specific to f [epochs] --> only used for bonus exercise
          val_loss_curve_Dp (optional): An array with validation loss values specific to Dp [epochs] --> only used for bonus exercise

        Returns:
          Fits of the input data
        """
    if loss_train is not None:
        plt.figure(2)
        plt.clf()
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(loss_train)
        if val_loss is not None:
            axs[0, 0].plot(val_loss)
            axs[0,0].legend(('training loss','validation loss'))
        else:
            axs[0,0].legend('training loss')
        axs[0,0].set(yscale="log",xlabel='epoch #',ylabel='loss')
    if train_loss_D is not None:
        axs[1, 0].plot(train_loss_D)
        if val_loss_curve_D is not None:
            axs[1, 0].plot(val_loss_curve_D)
            axs[1,0].legend(('training loss D','validation loss D'))
        else:
            axs[1,0].legend('training loss D')
        axs[1,0].set(yscale="log",xlabel='epoch #',ylabel='loss')
    if train_loss_f is not None:
        axs[0, 1].plot(train_loss_f)
        if val_loss_curve_f is not None:
            axs[0, 1].plot(val_loss_curve_f)
            axs[0, 1].legend(('training loss f', 'validation loss f'))
        else:
            axs[0, 1].legend('training loss f')
        axs[0,1].set(yscale="log",xlabel='epoch #',ylabel='loss')
    if train_loss_Dp is not None:
        axs[1, 1].plot(train_loss_Dp)
        if val_loss_curve_Dp is not None:
            axs[1, 1].plot(val_loss_curve_Dp)
            axs[1, 1].legend(('training loss Dp', 'validation loss Dp'))
        else:
            axs[1, 1].legend('training loss Dp')
        axs[1,1].set(yscale="log",xlabel='epoch #',ylabel='loss')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    bvalues=np.array(bvalues)
    if D_pred is None:
        D_pred = D_ref
        skipD=True
    else:
        skipD = False
    if f_pred is None:
        f_pred = f_ref
        skipf=True
    else:
        skipf=False
    if Dp_pred is None:
        Dp_pred = Dp_ref
        skipDs=True
    else:
        skipDs=False
    plt.close('all')
    fig, axs = plt.subplots(2, 2)
    if data is not None:
        axs[0, 0].plot(bvalues, data[0,:], 'o')
        datapred=ivim(bvalues, D_pred[0], f_pred[0], Dp_pred[0], 1)
        axs[0, 0].plot(bvalues, datapred)
        axs[0, 0].set_ylim(0, 1.2)
        axs[0, 0].set(xlabel='b-value (s/mm2)', ylabel='normalised signal')
        plt.legend(('data', 'estimate from network'))
    if not skipD:
        axs[1, 0].plot(D_ref, D_pred, 'o')
        axs[1, 0].set_ylim(0, 0.005)
        axs[1, 0].set(xlabel='D_ref (mm2/s)', ylabel='D_pred (mm2/s)')
    if not skipf:
        axs[0, 1].plot(f_ref, f_pred, 'o')
        axs[0, 1].set_ylim(0, 1)
        axs[0, 1].set(xlabel='f_ref', ylabel='f_pred')
    if not skipDs:
        axs[1, 1].plot(Dp_ref, Dp_pred, 'o')
        axs[1, 1].set_ylim(0, 0.3)
        axs[1, 1].set(xlabel='Dp_ref (mm2/s)', ylabel='Dp_pred (mm2/s)')
    plt.ion()
    plt.show()
    plt.pause(0.001)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
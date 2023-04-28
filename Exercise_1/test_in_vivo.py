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
# Date modified: Jan 2023
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import torch as nn
import Data_loader as dl
from train_mlp_torch import train, train_self_supervised
import numpy as np

def eval_in_vivo(hidden_dims, lr, batch_size, epochs, seed,
          optimizer_option='sgd'):
    """
    Performs training and then apply to real data.
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
    data, valid_id, bvalues = dl.load_real_data(eval=True)
    network = train(hidden_dims, lr, batch_size, epochs, seed,
          bvalues=bvalues,optimizer_option=optimizer_option)
    out_supervised = network.forward(nn.tensor(data)).cpu().detach().numpy()

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # for 5.3 you will need to use:
    # to obtain in-vivo training data

    #######################
    # END OF YOUR CODE    #
    #######################
    dl.plot_example(np.squeeze(out_supervised),valid_id)
    dl.plot_ref()
    pass


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
    parser.add_argument('--optimizer_option', default='sgd', type=str,
                        help='optimizer options: adam and sgd (default)')
    # Other hyperparameters
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')


    args = parser.parse_args()
    kwargs = vars(args)

    eval_in_vivo(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
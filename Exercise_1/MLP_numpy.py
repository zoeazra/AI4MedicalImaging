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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes MLP object.
        Args:
          n_inputs: number of inputs.
          hidden_sizes: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          output_size: number of predicted parameters.
                     This number is required in order to specify the
                     output dimensions of the MLP
        TODO: Implement initialization of the network.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################


        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        TODO: Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################

        return x

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.
        Args:
          dout: gradients of the loss
        TODO: Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################

        return dout

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.
        TODO: Iterate over modules and call the 'clear_cache' function.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################


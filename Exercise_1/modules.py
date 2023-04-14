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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.
        Initializes weight parameters using Kaiming initialization.
        Initializes biases with zeros.
        Also, initializes gradients with zeros.
        """

        self.weights = np.random.randn(in_features, out_features)/(in_features+out_features)
        self.bias = np.zeros([out_features,1])
        self.grads = {'bias': np.zeros(in_features), 'weight': np.zeros(np.shape(self.weights))}
        self.input_layer = input_layer

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        TODO: Implement forward pass of the module. Hint: You can store intermediate variables inside the object.
        TODO: They can be used in backward pass computation.
        """

        self.x = x

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        TODO: Implement backward pass of the module. Store gradient of the loss with respect to layer parameters in
        TODO: self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################


        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """

        self.grads['bias'] = None
        self.grads['weight'] = None
        self.x = None


class RELUModule(object):
    """
    RELU activation module.
    """
    def __init__(self):
        pass

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        TODO: Implement forward pass of the module. Hint: You can store intermediate variables inside the object.
        TODO: They can be used in backward pass computation.
        """
        self.x = x

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        TODO: Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.x = None


class TanhModule(object):
    """
    Tanh activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        TODO: Implement forward pass of the module.
        """

        self.x = x
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        TODO: Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        self.x = None


class MSE:
    """
    MSE loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: predicted values
          y: ground truth values
        Returns:
          out: MSE
        TODO: Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self):
        """
        Backward pass.
        Returns:
          dx: gradient of the loss with the respect to the input x.
        TODO: Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx
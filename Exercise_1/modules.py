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
This module implements various modules of the network.
You should fill in code into indicated sections.

3 A.	Program the different neural network modules in “modules.py”. Use “do_unittest.py” to test whether the forward and backward passes are correct. Make only use of matrix multiplications, no for loops. Points will be deduced when for-loops are used where matrix multiplications were possible.
"""
import numpy as np
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



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
        list_linear_modules = list()
        list_RELU_activations = list()
        for hidden_size in hidden_sizes:
            list_linear_modules.append(LinearModule(input_size, hidden_size))
            list_RELU_activations.append(RELUModule())
            input_size = hidden_size
        list_linear_modules.append(LinearModule(input_size, output_size))
        TANH = TanhModule()
        self.LM = list_linear_modules
        self.RELU = list_RELU_activations
        self.TANH = TANH


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
        for objs in self.RELU:
            objs.clear_cache
        for objs in self.LM:
            objs.clear_cache
        self.TANH.clear_cache

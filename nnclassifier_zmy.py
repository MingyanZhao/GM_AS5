from __future__ import print_function
import matplotlib.pyplot as plt
import os
import sys
import timeit

import numpy
import noise
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
#from naiveLoader import load_data
from utils import tile_raster_images

class nnclassifier:

    def __init__(self,
                 input = None, # input samples
                 label = None,  # labels of the
                 nn_input_dim = 500,  # input sample size
                 nn_hdim = 300,  # size of hidden layer
                 nn_output_dim = 10,  # output
                 reg_lambda = numpy.float32(0.001),  # regularization strength,
                 epsilon = numpy.float32(0.01)  # learning rate for gradient descent
                 ):

        #
        # for neural network classifier
        #
        self.label = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        self.input = input
        self.nn_input_dim = nn_input_dim
        self.nn_hdim = nn_hdim
        self.nn_output_dim = nn_output_dim

        self.W1 = theano.shared(numpy.random.randn(nn_input_dim, nn_hdim), name='W1')
        self.b1 = theano.shared(numpy.zeros(nn_hdim), name='b1')
        self.W2 = theano.shared(numpy.random.randn(nn_hdim, nn_output_dim), name='W2')
        self.b2 = theano.shared(numpy.zeros(nn_output_dim), name='b2')
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        #
        # for neural network classifier
        #

        self.params = [self.W1, self.b1, self.W2, self.b2]

    def set_label(self, label):
        self.label = label

    def get_cost(self, y):

        y_hat = self.get_pred_prob()

        # Returns a class prediction

        self.y_pred = T.argmax(y_hat, axis=1)

        # The regularization term (optional)
        loss_reg = 1./20 * self.reg_lambda/2 * (T.sum(T.sqr(self.W1)) + T.sum(T.sqr(self.W2)))
        # the loss function we want to optimize
        loss_nc = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg
        return loss_nc

    def get_pred_prob(self):
        z1 = T.dot(self.input, self.W1) + self.b1
        a1 = T.tanh(z1)
        z2 = T.dot(a1, self.W2) + self.b2
        y_hat = T.nnet.softmax(z2) # output probabilties

        return y_hat

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        y_prob = self.get_pred_prob()
        y_pred = T.argmax(y_prob, axis=1)
        # check if y has same dimension of y_pred
        if y.ndim != y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction

            return T.mean(T.neq(y_pred, y))
        else:
            raise NotImplementedError()


    def printContingencyTable(self, y):
        y_prob = self.get_pred_prob()
        y_pred = T.argmax(y_prob, axis=1)



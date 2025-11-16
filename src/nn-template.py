import math

import numpy as np


class Perceptron:
    def __init__(self):
        self.learning_rate = ...
        self.n_epochs = ...
        self.bias = ...
        self.thresh = ...
        self.weights = ...
        self.errors = ...

    def fit(self, x, y): ...

    def calculate_weighted_sum(self, x, w, b): ...

    def predict(self, x): ...


def relu(x): ...


def sigmoid(x): ...


def soft_max(x): ...


def mean_squared_error(y_pred, y): ...


def mean_absolute_error(y_pred, y): ...


def hinge(y_pred, y): ...


class Network:
    def __init__(self, sizes):
        """
        Initialize the neural network

        :param sizes: a list of the number of neurons in each layer
        """
        # save the number of layers in the network
        self.L = ...

        # store the list of layer sizes
        self.layer_sizes = ...

        # initialize the bias vectors for each hidden and output layer
        self.bias = ...

        # initialize the matrices of weights for each hidden and output layer
        self.weights = ...

        # initialize the derivatives of biases for backprop
        self.db = ...

        # initialize the derivatives of weights for backprop
        self.dw = ...

        # initialize the activities on each hidden and output layer
        self.z_activities = ...

        # initialize the activations on each hidden and output layer
        self.activations = ...

        # initialize the deltas on each hidden and output layer
        self.deltas = ...

    def g(self, z):
        """
        sigmoid activation function

        :param z: vector of activities to apply activation to
        """
        ...

    def g_prime(self, z):
        """
        derivative of sigmoid activation function

        :param z: vector of activities to apply derivative of activation to
        """
        ...

    def grad_loss(self, a, y):
        """
        evaluate gradient of cost function for squared-loss C(a,y) = (a-y)^2/2

        :param a: activations on output layer
        :param y: vector-encoded label
        """
        ...

    def forward_prop(self, x):
        """
        take an feature vector and propagate through network

        :param x: input feature vector
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        # TODO: step 1. Initialize activation on initial layer to x

        ## TODO: step 2-4. Loop over layers and compute activities and activations

    def back_prop(self, x, y):
        """
        Back propagation to get derivatives of C wrt weights and biases for given training example

        :param x: training features
        :param y: vector-encoded label
        """

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # TODO: step 1. forward prop training example to fill in activities and activations
        ...
        # TODO: step 2. compute deltas on output layer (Hint: python index numbering starts from 0 ends at N-1)
        ...
        # TODO: step 3-6. loop backward through layers, backprop deltas, compute dWs and dbs
        ...

    def train(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        eta=0.25,
        num_epochs=10,
        isPrint=True,
        isVis=False,
    ):
        """
        Train the network with SGD

        :param X_train: matrix of training features
        :param y_train: matrix of vector-encoded labels
        """

        # initialize shuffled indices
        shuffled_inds = list(range(X_train.shape[0]))

        # loop over training epochs (step 1.)
        for _ in range(num_epochs):
            ...
            # TODO loop over training examples (step 2.)
            ...

            # TODO: step 3. back prop to get derivatives
            ...

            # TODO: step 4. update all weights and biases for all layers
            ...

    def compute_loss(self, X, y):
        """
        compute average loss for given data set

        :param X: matrix of features
        :param y: matrix of vector-encoded labels
        """
        ...

    def gradient_check(self, x, y, h=1e-5):
        """
        check whether the gradient is correct for X, y

        Assuming that back_prop has finished.
        """
        ...


if __name__ == "__main__":
    nn = Network([2, 3, 2])

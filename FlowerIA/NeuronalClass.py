import numpy as np


class NeuronalNetwork(object):

    # Init function
    def __init__(self):
        # Define neuronal size
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        # Create random weight
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    # Forward propagation function
    def forward(self, x_input):
        # Calculation of the application of weights between the input neuron and the hidden neurons
        z = np.dot(x_input, self.W1)
        # Application of the sigmoid function
        self.z2 = self.sigmoid(z)
        # Calculation of the application of weights between the hidden neuron and the output neurons
        z3 = np.dot(self.z2, self.W2)
        # Application of the sigmoid
        output = self.sigmoid(z3)
        return output

    # Sigmoid function
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    # Derived from the sigmoid function
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # Back propagation function
    def backward(self, X_input, Y, output):
        # Calculating the error
        o_error = Y - output
        # Application of the sigmoid derivative to this error
        o_delta = o_error * self.sigmoidPrime(output)

        # Calculating the error of our hidden neurons
        z2_error = o_delta.dot(self.W2.T)
        # Application of the sigmoid derivative to this error
        z2_delta = z2_error * self.sigmoidPrime(self.z2)

        # Update weight
        self.W1 += X_input.T.dot(z2_delta)
        self.W2 += self.z2.T.dot(o_delta)

    # Training function
    def train(self, inp, wait):
        f_output = self.forward(inp)
        self.backward(inp, wait, f_output)

    # Prediction function
    def predict(self, xPrediction):
        print("nb: " + str(self.forward(xPrediction)))

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

    # Fonction de propagation avant
    def forward(self, X):
        self.z = np.dot(X, self.W1)  # Multiplication matricielle entre les valeurs d'entrer et les poids W1
        self.z2 = self.sigmoid(self.z)  # Application de la fonction d'activation (Sigmoid)
        self.z3 = np.dot(self.z2, self.W2)  # Multiplication matricielle entre les valeurs cachés et les poids W2
        o = self.sigmoid(
            self.z3)  # Application de la fonction d'activation, et obtention de notre valeur de sortie final
        return o

    # Fonction d'activation
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    # Dérivée de la fonction d'activation
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # Fonction de rétropropagation
    def backward(self, X, y, o):
        o_error = y - o  # Calcul de l'erreur
        o_delta = o_error * self.sigmoidPrime(o)  # Application de la dérivée de la sigmoid à cette erreur

        z2_error = o_delta.dot(self.W2.T)  # Calcul de l'erreur de nos neurones cachés
        z2_delta = z2_error * self.sigmoidPrime(
            self.z2)  # Application de la dérivée de la sigmoid à cette erreur

        self.W1 += X.T.dot(z2_delta)  # On ajuste nos poids W1
        self.W2 += self.z2.T.dot(o_delta)  # On ajuste nos poids W2

    def train(self, inp, wait):
        f_output = self.forward(inp)
        self.backward(inp, wait, f_output)

    # Fonction de prédiction
    def predict(self, xPrediction):
        if self.forward(xPrediction) < 0.5:
            print("nb: " + self.forward(xPrediction))
        else:
            print("nb: " + str(self.forward(xPrediction)))

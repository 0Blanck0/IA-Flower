import numpy as np
import NeuronalClass as Neuronal

# Input value sent to AI
X_input = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4, 1.5]), dtype=float)
# Expected Output Value (Last input value is not included)
Y = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), dtype=float)

X_input = X_input / np.amax(X_input, axis=0)

# Get input value (Last input value is not included)
X = np.split(X_input, [8])[0]

# Get last input value
xPrediction = np.split(X_input, [8])[1]

# Create Neuronal AI object
NN = Neuronal.NeuronalNetwork()

# Training of the AI before sending the sought value
for i in range(100):
    NN.train(X, Y)

# Print input value and expected output value
print("\nInput value: \n" + str(X))
print("\n")
print("Expected Output value: \n" + str(Y))
print("\n")

# Search the last value of the input array and display are results
NN.predict(xPrediction)

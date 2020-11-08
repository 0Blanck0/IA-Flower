import numpy as np
import NeuronalClass as Neuronal

x_entre = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4, 1.5]), dtype=float)
y = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), dtype=float)

x_entre = x_entre / np.amax(x_entre, axis=0)

x = np.split(x_entre, [8])[0]
xPrediction = np.split(x_entre, [8])[1]

NN = Neuronal.NeuronalNetwork()

for i in range(100):
    print(i)
    NN.train(x, y)

print("Valeurs d'entrées: \n" + str(x))
print("Sortie actuelle: \n" + str(y))
print("\n")

NN.predict(xPrediction)

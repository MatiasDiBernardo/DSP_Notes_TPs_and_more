import numpy as np
import matplotlib.pyplot as plt

#Llegar a la rta en frecuecia de una media movil por linialidad es sumar exponenciales complejas
w = np.linspace(-np.pi, np.pi, 10000)
M = 4
kernel = np.ones(len(w), dtype=complex)

for i in range(M):
    kernel += np.exp(1j*w*(i+1)) / M
    plt.plot(w, kernel)

plt.plot(w, kernel)
plt.show()
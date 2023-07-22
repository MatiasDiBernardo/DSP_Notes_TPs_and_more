import numpy as np
import matplotlib.pyplot as plt

N = 256
M = np.exp(-1j*2*np.pi/N)
matrix = np.ones((N,N), dtype=complex)

for i in range(N):
    for j in range(N):
        if i != 0 and j != 0:
            matrix[i,j] = M**(i*j)

plt.imshow(matrix.real)
plt.show()
        
import numpy as np
import matplotlib.pyplot as plt

T = 0.1
t = np.arange(0, 1, T)
n = np.arange(1,len(t) - 1)

h_t =  np.exp(-10*t)
h_n = np.zeros(len(n) + 1)
h_n[0] = 1/3
h_n[1:] = 1/3*((1/3)**n + (1/3)**(n-1))

plt.plot(h_t)
plt.plot(h_n)
plt.show()

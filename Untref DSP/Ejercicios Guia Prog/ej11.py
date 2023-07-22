import numpy as np
import matplotlib.pyplot as plt


def generate_func(a0, a1, a2, N, graph=False):
    t = np.arange(N)
    print(t)
    x = a0 - a1 * np.cos((2*np.pi*t)/(N-1)) + a2 * np.cos((4*np.pi*t)/(N-1))
    
    if graph:
        plt.plot(x)
        plt.show()
    
generate_func(0.42, 0.5, 0.08, 10000, graph=True) 
    
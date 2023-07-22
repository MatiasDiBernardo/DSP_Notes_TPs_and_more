import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import math
#Transformada de un tran de deltas infinito es otro tren de deltas, por linealidad
#es sumar sinudoides de distintas frecuencias.
T = 1
fs = 1000 
seg = 5
M = fs * seg

delta_comb = np.zeros(M)
delta_comb[::T*fs] = 1

sine_sum = np.zeros(M)
t = np.linspace(0, seg, M)
iter_sum = 50
for i in range(iter_sum):
    sine_sum += np.cos(2*np.pi * i * t)
sine_sum /= iter_sum

print(math.lcm(6,8))

plt.figure("Sine sum")
plt.plot(t, sine_sum)
plt.figure("Comb")
plt.plot(t, delta_comb)
plt.show()
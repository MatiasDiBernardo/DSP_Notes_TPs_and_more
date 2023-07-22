import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

"""
Entiendiendo resolución como la capacidad de distinguir distintas frecuencias
dentro de una señal se puede ver como hacer zero pad no mejor la resolución sino que me da mas
muestras en el gráfico de respuesta el frecuencia. Pero para mejorar la resolución de la 
respuesta en frecuencia tengo que ajustar el tipo de ventana y cantidad de muestras de la ventana. 

Ejemplo del Oppenheim 10.6/7/8
L = 32 y N = 32, L = 64 y N = 64, L = 128 y N = 128 
L = 32 y N = 32, L = 32 y N = 64, L = 32 y N = 128, L = 32 y N = 1024
L = 32 y N = 1024, L = 42 y N = 1024, L = 64 y N = 1024 
"""

L = 32  #Muestas Señal
N = 64  #Muestras DFT
t = np.arange(L)
w = signal.windows.kaiser(L, 5.48)
x = w * np.cos(2*np.pi*t/14) + 0.75 * w * np.cos(4*np.pi*t/15)

print(f"Bin frecuencial del 1er cos: {L/14}")
print(f"Bin frecuencial del 2do cos: {L/7.5}")

rta_freq = fft(x, N) / (len(x)/2.0)  #Pregunta en la normalización usar x o N
freq_axis = fftfreq(N)

plt.plot(np.abs(rta_freq))
plt.show()

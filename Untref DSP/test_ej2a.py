import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

#Comparison between analítical answer and fft answer

fs = 1000
t = np.linspace(-5, 10, fs * 10)

x = (0.5)**t
x[:2*fs] = 0

w = np.linspace(-np.pi, np.pi, int(fs/4))
rta_analitic = ((0.5)**-3 * np.exp(1j*3*w))/(1 - (np.exp(1j*w)/2))

rta_fft = fft(x)
rta_freq = fftfreq(len(x), 1/fs)

plt.figure("")
plt.plot(t, x)
plt.figure("Comparisión")
plt.plot(w, np.abs(rta_analitic))
plt.figure()
plt.plot(rta_freq, np.abs(rta_fft))
plt.show()


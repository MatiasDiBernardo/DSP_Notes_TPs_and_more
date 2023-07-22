import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

x, fs = librosa.load("Data\\rta_imp.wav", sr=48000)
idx_max = np.argmax(x)
delta_t = 0.005
x_cut = x[idx_max - 20: idx_max + int(fs*delta_t) - 20]  # Acá en el recorte conviene restar un par de muestras para
#poder tener en el impulso recortado la parte de subida que tiene infrormación importante
L = len(x_cut)
print("Duración muestras señal: ", len(x_cut))

rta_freq = rfft(x_cut)/ L
rta_freq = np.log10(np.abs(rta_freq))

freq_axis = rfftfreq(L, 1/fs)

plt.figure("Señal original")
plt.plot(x)
plt.figure("Señal recortada")
plt.plot(x_cut)
plt.figure("Rta en freq")
plt.plot(freq_axis, rta_freq)
plt.xscale("log")
plt.show()
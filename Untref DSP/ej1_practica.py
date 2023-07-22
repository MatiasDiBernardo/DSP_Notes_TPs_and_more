import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate as scpcor
import librosa

piano_cerca, fs1 = librosa.load("Data\\piano_cerca.wav", sr=44100)
piano_lejos, fs2 = librosa.load("Data\\piano_lejos.wav", sr=44100)

cor_numpy = np.correlate(piano_cerca, piano_lejos, mode="full")
cor_scipy = scpcor(piano_cerca, piano_lejos, method="direct")

delay_numpy = np.argmax(cor_numpy)
delay_scipy = np.argmax(cor_scipy)

#delay_correccion = len(cor_numpy) - delay_numpy

delay_correccion = delay_numpy - len(piano_cerca)  #Esto no funciona porque tengo que tener en cuenta el desplazamiento
#de las dos señales, y tampoco es sumar las dos y restarlas, se usa una función que me calcule el lag entre las dos
#señales y me reescale el eje X en frecuencia.

print("Delay correción ", delay_correccion)
#delay2 = len(piano_cerca) - delay_numpy

print(f"Delay muestras: ", delay_numpy)

if delay_numpy != delay_scipy:
    print(f"Numpy delay: ", delay_numpy)
    print(f"Scipy delay: ", delay_scipy)

print(f"Son lo mismo: ", np.all(delay_numpy == delay_scipy))

piano_desplazado = piano_lejos[delay_correccion:]
piano_desplazado2 = np.roll(piano_lejos, delay_numpy) 


figure, ax = plt.subplots(2)
ax[0].plot(cor_numpy)
ax[1].plot(cor_scipy)
ax[1].set_xlabel("Muestras")
ax[0].set_ylabel("Correlación")
ax[0].set_ylabel("Correlación")

figure2, ax2 = plt.subplots(4)
ax2[0].set_title("Piano cerca")
ax2[0].plot(piano_cerca)
ax2[0].set_ylabel("Amplitud")

ax2[1].set_title("Piano lejos")
ax2[1].plot(piano_lejos)
ax2[1].set_ylabel("Amplitud")

ax2[2].set_title("Piano lejos desplazado")
ax2[2].plot(piano_desplazado)
ax2[2].set_xlabel("Muestras")
ax2[2].set_ylabel("Correlación")

ax2[3].set_title("Piano lejos desplazado2")
ax2[3].plot(piano_desplazado2)
ax2[3].set_xlabel("Muestras")
ax2[3].set_ylabel("Correlación")
plt.show()

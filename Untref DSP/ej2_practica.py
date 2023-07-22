import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft, fftfreq

def filtro_medio_movil(data, w_size): 
    num_iter = len(data) - w_size
    filt_data = []
    for i in range(num_iter):
        avg_data = np.mean(data[i: i + w_size])
        filt_data.append(avg_data)
    
    filt_data = np.array(filt_data)
    add_end =  np.ones(len(data) - len(filt_data)) * filt_data[-1]
    filt_data = np.concatenate([filt_data, add_end])

    return  filt_data
    
df = pd.read_csv("Data\\ruido_urbano.csv")
raw_data = df["value"].to_numpy()

#Como la señal esta en dB que no mantiene una relación lineal, conviene para filtrar usar
#una relación lineal, por eso se pasa a pascales y de úlitma luego se pasa devuelta a dB

#raw_data = np.log10(raw_data)  
seg = df["seg"].to_numpy()
fft_data = fft(raw_data)

fs = 1/120  #Un 2 min por muestra
fs_dia = 120
rta_freq = fftfreq(len(raw_data), 1/fs)

min_of_window = 60  #Each sample is 2 min
filt_data = filtro_medio_movil(raw_data, min_of_window//2)

plt.figure("Data raw/filrada")
plt.plot(raw_data)
plt.plot(filt_data)
plt.figure("Rta en freq de la señal")
plt.plot(rta_freq,fft_data)
plt.show()


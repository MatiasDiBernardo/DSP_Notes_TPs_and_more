import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_analog():
    n=27 #orden del filtro analógico
    wc_1 = 10.09 #frecuencia de corte analógica
    b_1, a_1 = signal.butter(n, wc_1, 'low', analog=True) #diseño del filtro
    w_1, h_1 = signal.freqs(b_1, a_1) #respuesta en frecuencia del filtro
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)
    ax.semilogx(w_1, 20 * np.log10(abs(h_1)))
    ax.set_xlabel('frecuencia angular [radianes / segundo]')
    ax.set_ylabel('amplitud [db]')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')
    ax.axvline(wc_1, color='green') #frecuencia de corte
    ax.axvline(10,color='red') #final banda de paso
    ax.axvline(11,color='red') #comienzo banda de atenuación
    ax.axhline(-3,xmax=0.51, color='green',ls='--') # -3 db
    ax.set_xticks([2,5,10,20,50,100])
    ax.set_yticks([-3,-12,-20,-40,-60,-80])
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.get_xaxis().set_major_formatter(formatter)

    return b_1, a_1

def plot_digi():
    b_1, a_1 = plot_analog()
    t_1 = 0.02 #periodo de muestreo
    fs_1 = 1/t_1 #frecuencia de muestreo
    b_digital, a_digital = signal.bilinear(b_1, a_1, fs_1) #mapeo de s a z con T. Bilineal
    wz, hz = signal.freqz(b_digital, a_digital) #respuesta en frecuencia del filtro digital
    #print(hz)


    fig = plt.figure("Digital")
    ax = fig.add_subplot(111)
    #ax.semilogx(wz, 20 * np.log10(abs(hz)))
    ax.plot(wz, hz)
    ax.set_xlabel('frecuencia angular [radianes / muestra]')
    ax.set_ylabel('amplitud [db]')
    #ax.margins(0, 0.1)
    #ax.grid(which='both', axis='both')
    #ax.set_ylim([-40,2])
    #wc_digital = 2 * np.arctan((wc*t_1)/2)
    #ax.axvline(0.16*np.pi,color='red') #final banda de paso
    #ax.axvline(0.295*np.pi,color='red') #comienzo banda de atenuación
    #ax.axhline(-2, color='green',ls='--') # -1 db
    #ax.axhline(-20, color='green',ls='--') # -20 dB
    #ax.set_xticks([0.5,1,1.5,2,2.5,3])
    #ax.set_yticks([-1,-10,-20,-40,-60,-80])
    #formatter = ScalarFormatter()
    #formatter.set_scientific(False)
    #ax.get_xaxis().set_major_formatter(formatter)

    plt.show()

plot_digi()
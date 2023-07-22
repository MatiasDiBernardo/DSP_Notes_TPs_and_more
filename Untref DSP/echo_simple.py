import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import librosa
import soundfile as sf

def echo_simple(x, fs, delay, alpha):
    delay_samples = int(delay * fs)
    add_extra_time = len(x) * delay_samples
    y = np.zeros(len(x) + add_extra_time)
    y += np.hstack([x, np.zeros(len(y) - len(x))])
    if (len(x) + delay_samples) < len(y):
        y += alpha * np.hstack([np.zeros(delay_samples), x, np.zeros(len(y) - len(x) - delay_samples)])
    else:
        y += alpha * np.hstack([np.zeros(delay_samples), x[:len(y) - delay_samples]])
    
    return y[:len(x)]  #To match sizes

def echo_simple_conv(x, fs, delay, alpha):
    delay_samples = int(delay * fs)
    conv_signal = np.zeros(len(x))
    conv_signal[0] = 1
    conv_signal[delay_samples] = alpha
    y = np.convolve(x, conv_signal)
    
    return y[:len(x)]

def echo_simple_freq(x, fs, delay, alpha):
    """
    Función de transferencia sacada del power clase_7bis__Aplicaciones_de_la_TZ_al_Audio__2020_2C
    diapo 12.
    """
    w = np.linspace(-np.pi, np.pi, len(x))
    D = int(delay*fs)  #D is in samples
    transfer_function = 1 + alpha * np.exp(-1j*w*D)
    x_freq = fft(x)

    Y_z = x_freq * transfer_function
    x_mod = ifft(Y_z)
    
    return np.real(x_mod)

def echo_inf(x, fs, delay, alpha):
    delay_samples = int(delay * fs)
    buffer_delay = np.zeros(delay_samples)
    y = np.zeros(len(x))

    for i in range(len(x)):
        y[i] = x[i] + alpha * buffer_delay[-1]
        buffer_delay = np.hstack([y[i], buffer_delay[:-1]])
    
    return y

def echo_inf_freq(x, fs, delay, alpha):
    """
    Función de transferencia sacada del power clase_7bis__Aplicaciones_de_la_TZ_al_Audio__2020_2C
    diapo 19.
    """
    w = np.linspace(-np.pi, np.pi, len(x))
    D = int(delay * fs)
    transfer_function = 1/(1 - alpha * np.exp(-1j*w*D))

    x_freq = fft(x)
    Y_z = x_freq * transfer_function
    x_mod = ifft(Y_z)
    
    return np.real(x_mod)

def plot_single_echo(x, fs, delay, alpha, range=None):
    time_shift = echo_simple(x, fs, delay, alpha)
    time_conv = echo_simple_conv(x, fs, delay, alpha)
    freq_rta = echo_simple_freq(x, fs, delay, alpha)
    t = np.linspace(0, len(x)/fs, len(x))

    if not range:
        range = len(x)

    fig, ax = plt.subplots(3, 1)
    cutt = int(range*fs)
    ax[0].plot(t[:cutt], time_shift[:cutt])
    ax[1].plot(t[:cutt], time_conv[:cutt])
    ax[2].plot(t[:cutt], freq_rta[:cutt])
    plt.show()

def plot_inf_echo(x, fs, delay, alpha, range=None):
    echo_time = echo_inf(x, fs, delay, alpha)
    echo_freq = echo_inf_freq(x, fs, delay, alpha)
    t = np.linspace(0, len(x)/fs, len(x))

    if not range:
        range = len(x)

    fig, ax = plt.subplots(2, 1)
    cutt = int(range*fs)
    ax[0].plot(t[:cutt], echo_time[:cutt])
    ax[1].plot(t[:cutt], echo_freq[:cutt])
    plt.show()

def save_wav(x, fs, delay, alpha):
    echo_time = echo_inf(x, fs, delay, alpha)
    echo_freq = echo_inf_freq(x, fs, delay, alpha)

    sf.write("Data\\echo_inf_time.wav", echo_time, fs)
    sf.write("Data\\echo_inf_freq.wav", echo_freq, fs)

x, fs = librosa.load("data\\rta_imp.wav")
delay = 0.2  #In seconds
alpha = 0.8

plot_inf_echo(x, fs, delay, alpha, 1)
save_wav(x, fs, delay, alpha)

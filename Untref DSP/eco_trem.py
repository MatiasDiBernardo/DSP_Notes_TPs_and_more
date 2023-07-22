import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fftpack import fft, ifft, fftshift, ifftshift

def echo(x, fs, D, N, alpha):
    """Echo effect

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        D (float): Time between echos.
        N (int): Amount of echos.
        alpha (float): Decay factor.
    """
    
    add_extra_time = N * int(D * fs)
    echo_signal = np.zeros(len(x) + add_extra_time)

    for i in range(N  + 1):
        amp_factor = alpha/(alpha + i)
        delay = int(D * fs)
        echo_signal[i*delay:i*delay + len(x)] += x * amp_factor
    
    return echo_signal

def single_echo(x, D, alpha):
    w = np.linspace(-np.pi, np.pi, len(x))
    transfer_function = (np.exp(1j*w*D) + alpha)/np.exp(1j*w*D) 
    x_freq = fft(x)

    Y_z = x_freq * transfer_function
    x_mod = ifft(Y_z)

    return np.real(x_mod)

def inf_echo(x, D, alpha):
    w = np.linspace(-np.pi, np.pi, len(x))
    transfer_function = 1/(1 - alpha * np.exp(1j*w * -D))
    x_freq = fft(x)
    #plt.plot(transfer_function)
    #plt.plot(x_freq)
    #plt.show()
    #x_freq = fftshift(x_freq)

    Y_z = x_freq * transfer_function
    x_mod = ifft(Y_z)
    #x_mod = ifftshift(x_mod)

    return np.real(x_mod), np.imag(x_mod) 

def tremolo(x, fs, f0, amp):
    """Tremolo effect.

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        f0 (float): Frequency of modulation signal.
        amp (float): Amount of effect.
    """
    n = np.linspace(0, len(x)/fs, len(x))
    mod_signal = amp * np.cos(2*np.pi*f0*n) + 1

    return x * mod_signal

def mix_effects(x1, x2):
    if len(x1) > len(x2):
        diff = len(x1) - len(x2)
        mixed = x1 + np.concatenate([x2, np.zeros(diff)])
        return mixed/2
    else:
        diff = len(x2) - len(x1)
        mixed = x2 + np.concatenate([x1, np.zeros(diff)])
        return mixed/2

x, fs = librosa.load("Data\\piano_cerca.wav")

delay = 0.1
alpha1 = 5
alpha2 = 0.5
N_echos = 4

echo_effect =  echo(x, fs, delay, N_echos, alpha1)
one_echo = single_echo(x, delay, alpha2)
echo_inf1, echo_inf2 = inf_echo(x, delay, alpha2)
tremolo_effect = tremolo(x, fs, 5, 2)
mixed_effect = mix_effects(echo_effect, tremolo_effect)

#plt.plot(echo_inf1)
#plt.plot(echo_inf2)
#plt.show()

sf.write("Data\\test_echo2.wav", echo_effect, fs)
sf.write("Data\\test_single_echo.wav", one_echo, fs)
sf.write("Data\\test_echo_inf.wav", echo_inf1, fs)
sf.write("Data\\test_trem2.wav", tremolo_effect, fs)
sf.write("Data\\test_mixed2.wav", mixed_effect, fs)
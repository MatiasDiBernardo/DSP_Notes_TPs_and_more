import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, ifft
from scipy.io import wavfile

#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
#import stft
#import utilFunctions as UF
eps = np.finfo(float).eps


"""
A4-Part-2: Measuring noise in the reconstructed signal using the STFT model 

Write a function that measures the amount of noise introduced during the analysis and synthesis of a 
signal using the STFT model. Use SNR (signal to noise ratio) in dB to quantify the amount of noise. 
Use the stft() function in stft.py to do an analysis followed by a synthesis of the input signal.

A brief description of the SNR computation can be found in the pdf document (A4-STFT.pdf, in Relevant 
Concepts section) in the assignment directory (A4). Use the time domain energy definition to compute
the SNR.

With the input signal and the obtained output, compute two different SNR values for the following cases:

1) SNR1: Over the entire length of the input and the output signals.
2) SNR2: For the segment of the signals left after discarding M samples from both the start and the 
end, where M is the analysis window length. Note that this computation is done after STFT analysis 
and synthesis.

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N), and hop size (H). The function should return a python 
tuple of both the SNR values in decibels: (SNR1, SNR2). Both SNR1 and SNR2 are float values. 

Test case 1: If you run your code using piano.wav file with 'blackman' window, M = 513, N = 2048 and 
H = 128, the output SNR values should be around: (67.57748352378475, 304.68394693221649).

Test case 2: If you run your code using sax-phrase-short.wav file with 'hamming' window, M = 512, 
N = 1024 and H = 64, the output SNR values should be around: (89.510506656299285, 306.18696700251388).

Test case 3: If you run your code using rain.wav file with 'hann' window, M = 1024, N = 2048 and 
H = 128, the output SNR values should be around: (74.631476225366825, 304.26918192997738).

Due to precision differences on different machines/hardware, compared to the expected SNR values, your 
output values can differ by +/-10dB for SNR1 and +/-100dB for SNR2.
"""

tol = 1e-14 
def dftAnal(x, w, N):
    """
	Analysis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size 
	returns mX, pX: magnitude and phase spectrum
	"""



    if w.size > N:  # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N // 2) + 1  # size of positive spectrum, it includes sample 0
    hM1 = (w.size + 1) // 2  # half analysis window size by rounding
    hM2 = w.size // 2  # half analysis window size by floor
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    w = w / sum(w)  # normalize analysis window
    xw = x * w  # window the input sound
    fftbuffer[:hM1] = xw[hM2:]  # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)  # compute FFT
    absX = abs(X[:hN])  # compute ansolute value of positive side
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)  # magnitude spectrum of positive frequencies in dB
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0  # for phase calculation set to 0 the small values
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0  # for phase calculation set to 0 the small values
    pX = np.unwrap(np.angle(X[:hN]))  # unwrapped phase spectrum of positive frequencies
    return mX, pX


def dftSynth(mX, pX, M):
    """
	Synthesis of a signal using the discrete Fourier transform
	mX: magnitude spectrum, pX: phase spectrum, M: window size
	returns y: output signal
	"""

    hN = mX.size  # size of positive spectrum, it includes sample 0
    N = (hN - 1) * 2  # FFT size

    hM1 = int(math.floor((M + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(M / 2))  # half analysis window size by floor
    y = np.zeros(M)  # initialize output array
    Y = np.zeros(N, dtype=complex)  # clean output spectrum
    Y[:hN] = 10 ** (mX / 20) * np.exp(1j * pX)  # generate positive frequencies
    Y[hN:] = 10 ** (mX[-2:0:-1] / 20) * np.exp(-1j * pX[-2:0:-1])  # generate negative frequencies
    fftbuffer = np.real(ifft(Y))  # compute inverse FFT
    y[:hM2] = fftbuffer[-hM2:]  # undo zero-phase window
    y[hM2:] = fftbuffer[:hM1]
    return y

def stft(x, w, N, H):
    """
	Analysis/synthesis of a sound using the short-time Fourier transform
	x: input sound, w: analysis window, N: FFT size, H: hop size
	returns y: output sound
	"""

    if (H <= 0):  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size  # size of analysis window
    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM1))  # add zeros at the end to analyze last sample
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    y = np.zeros(x.size)  # initialize output array
    while pin <= pend:  # while sound pointer is smaller than last sample
        # -----analysis-----
        x1 = x[pin - hM1:pin + hM2]  # select one frame of input sound
        mX, pX = dftAnal(x1, w, N)  # compute dft
        # -----synthesis-----
        y1 = dftSynth(mX, pX, M)  # compute idft
        y[pin - hM1:pin + hM2] += H * y1  # overlap-add to generate output sound
        pin += H  # advance sound pointer
    y = np.delete(y, range(hM2))  # delete half of first window which was added in stftAnal
    y = np.delete(y, range(y.size - hM1, y.size))  # delete half of the last window which as added in stftAnal
    return y

def computeSNR(inputFile, window, M, N, H):
    """
    Input:
            inputFile (string): wav file name including the path 
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming, 
                    blackman, blackmanharris)
            M (integer): analysis window length (odd positive integer)
            N (integer): fft size (power of two, > M)
            H (integer): hop size for the stft computation
    Output:
            The function should return a python tuple of both the SNR values (SNR1, SNR2)
            SNR1 and SNR2 are floats.
    """

    sr, audio = wavfile.read(inputFile)
    w = get_window(window, M) 
        
    audioProcess = stft(audio, w, N, H)

    energyAudio = sum(audioProcess**2)
    noise = audio - audioProcess
    energyNoise = sum(noise**2)

    energyAudioWin = sum(audioProcess[M:-M]**2)
    noiseWin  = audio[M:-M] - audioProcess[M:-M]
    energyNoiseWin = sum(noiseWin**2)

    SNR1 = 10 * np.log10(energyAudio/energyNoise)
    SNR2 = 10 * np.log10(energyAudioWin/energyNoiseWin)

    return SNR1, SNR2

"""
Test case 1: If you run your code using piano.wav file with 'blackman' window, M = 513, N = 2048 and 
H = 128, the output SNR values should be around: (67.57748352378475, 304.68394693221649).

Test case 2: If you run your code using sax-phrase-short.wav file with 'hamming' window, M = 512, 
N = 1024 and H = 64, the output SNR values should be around: (89.510506656299285, 306.18696700251388).

Test case 3: If you run your code using rain.wav file with 'hann' window, M = 1024, N = 2048 and 
H = 128, the output SNR values should be around: (74.631476225366825, 304.26918192997738).

Due to precision differences on different machines/hardware, compared to the expected SNR values, your 
output values can differ by +/-10dB for SNR1 and +/-100dB for SNR2.
"""
#All good except for the first case

    
import numpy as np
import sys
sys.path.append('../../software/models/')
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import get_window
import math
from scipy.fftpack import fft, ifft
#from dftModel import dftAnal
"""
A3-part-5: FFT size and zero-padding (Optional)

Write a function that takes in an input signal, computes three different FFTs on the input and returns 
the first 80 samples of the positive half of the FFT magnitude spectrum (in dB) in each case. 

This part is a walk-through example to provide some insights into the effects of the length of signal 
segment, the FFT size, and zero-padding on the FFT of a sinusoid. The input to the function is x, which
is 512 samples of a real sinusoid of frequency 110 Hz and the sampling frequency fs = 1000 Hz. You will 
first extract the first 256 samples of the input signal and store it as a separate variable xseg. You 
will then generate two 'hamming' windows w1 and w2 of size 256 and 512 samples, respectively (code given
below). The windows are used to smooth the input signal. Use dftAnal to obtain the positive half of the 
FFT magnitude spectrum (in dB) for the following cases:
Case-1: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 256
Case-2: Input signal x (512 samples), window w2 (512 samples), and FFT size of 512
Case-3: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 512 (Implicitly does a 
        zero-padding of xseg by 256 samples)
Return the first 80 samples of the positive half of the FFT magnitude spectrum output by dftAnal. 

To understand better, plot the output of dftAnal for each case on a common frequency axis. Let mX1, mX2, 
mX3 represent the outputs of dftAnal in each of the Cases 1, 2, and 3 respectively. You will see that 
mX3 is the interpolated version of mX1 (zero-padding leads to interpolation of the DFT). You will also 
observe that the 'mainlobe' of the magnitude spectrum in mX2 will be much smaller than that in mX1 and 
mX3. This shows that choosing a longer segment of signal for analysis leads to a narrower mainlobe with 
better frequency resolution and less spreading of the energy of the sinusoid. 

If we were to estimate the frequency of the sinusoid using its DFT, a first principles approach is to 
choose the frequency value of the bin corresponding to the maximum in the DFT magnitude spectrum. 
Some food for thought: if you were to take this approach, which of the Cases 1, 2, or 3 will give you 
a better estimate of the frequency of the sinusoid ? Comment and discuss on the forums!

Test case 1: The input signal is x (of length 512 samples), the output is a tuple with three elements: 
(mX1_80, mX2_80, mX3_80) where mX1_80, mX2_80, mX3_80 are the first 80 samples of the magnitude spectrum 
output by dftAnal in cases 1, 2, and 3, respectively.

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

def zpFFTsizeExpt(x, fs):
    """
    Inputs:
        x (numpy array) = input signal (2*M = 512 samples long)
        fs (float) = sampling frequency in Hz
    Output:
        The function should return a tuple (mX1_80, mX2_80, mX3_80)
        mX1_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-1
        mX2_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-2
        mX3_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-3
        
    The first few lines of the code to generate xseg and the windows have been written for you, 
    please use it and do not modify it. 

    Case-1: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 256
Case-2: Input signal x (512 samples), window w2 (512 samples), and FFT size of 512
Case-3: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 512 (Implicitly does a 
        zero-padding of xseg by 256 samples)
    """
    
    M = int(len(x)/2)
    xseg = x[:M]
    w1 = get_window('hamming',M)
    w2 = get_window('hamming',2*M)

    case1, phase1 = dftAnal(xseg, w1, len(xseg))
    case2, phase2 = dftAnal(x, w2, len(x))
    case3, phase3 = dftAnal(xseg, w1, len(x))

    return (case1[:80], case2[:80], case3[:80])
    
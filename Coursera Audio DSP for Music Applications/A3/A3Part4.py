import sys
#sys.path.append('../../software/models/')
#from dftModel import dftAnal, dftSynth
from scipy.signal import get_window
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fftpack import fft, ifft
"""
A3-Part-4: Suppressing frequency components using DFT model

Given a frame of the signal, write a function that uses the dftModel functions to suppress all the 
frequency components <= 70Hz in the signal and returns the output of the dftModel 
with and without filtering. 

You will use the DFT model to implement a very basic form of filtering to suppress frequency components. 
When working close to mains power lines, there is a 50/60 Hz hum that can get introduced into the 
audio signal. You will try to remove that using a basic DFT model based filter. You will work on just 
one frame of a synthetic audio signal to see the effect of filtering. 

You can use the functions dftAnal and dftSynth provided by the dftModel file of sms-tools. Use dftAnal 
to obtain the magnitude spectrum (in dB) and phase spectrum of the audio signal. Set the values of 
the magnitude spectrum that correspond to frequencies <= 70 Hz to -120dB (there may not be a bin 
corresponding exactly to 70 Hz, choose the nearest bin of equal or higher frequency, e.g., using np.ceil()).
If you have doubts converting from frequency (Hz) to bins, you can review the beginning of theory lecture 2T1.

Use dftSynth to synthesize the filtered output signal and return the output. The function should also return the 
output of dftSynth without any filtering (without altering the magnitude spectrum in any way). 
You will use a hamming window to smooth the signal. Hence, do not forget to scale the output signals 
by the sum of the window values (as done in sms-tools/software/models_interface/dftModel_function.py). 
To understand the effect of filtering, you can plot both the filtered output and non-filtered output 
of the dftModel. 

Please note that this question is just for illustrative purposes and filtering is not usually done 
this way - such sharp cutoffs introduce artifacts in the output. 

The input is a M length input signal x that contains undesired frequencies below 70 Hz, sampling 
frequency fs and the FFT size N. The output is a tuple with two elements (y, yfilt), where y is the 
output of dftModel with the unaltered original signal and yfilt is the filtered output of the dftModel.

Caveat: In python (as well as numpy) variable assignment is by reference. if you assign B = A, and 
modify B, the value of A also gets modified. If you do not want this to happen, consider using B = A.copy(). 
This creates a copy of A and assigns it to B, and hence, you can modify B without affecting A.

Test case 1: For an input signal with 40 Hz, 100 Hz, 200 Hz, 1000 Hz components, yfilt will only contain
100 Hz, 200 Hz and 1000 Hz components. 

Test case 2: For an input signal with 23 Hz, 36 Hz, 230 Hz, 900 Hz, 2300 Hz components, yfilt will only contain
230 Hz, 900 Hz and 2300 Hz components. 
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

def suppressFreqDFTmodel(x, fs, N):
    """
    Inputs:
        x (numpy array) = input signal of length M (odd)
        fs (float) = sampling frequency (Hz)
        N (positive integer) = FFT size
    Outputs:
        The function should return a tuple (y, yfilt)
        y (numpy array) = Output of the dftSynth() without filtering (M samples long)
        yfilt (numpy array) = Output of the dftSynth() with filtering (M samples long)
    The first few lines of the code have been written for you, do not modify it. 
    """
    M = len(x)

    w = get_window('hamming', M)
    outputScaleFactor = sum(w)

    Xamp, Xphase = dftAnal(x, w, N)

    cleanX = np.copy(Xamp)

    bin = fs/N
    indexFreq70 = int(np.ceil(70 / bin)) + 1  #Add plus one for the correct bin
    Xamp[0:indexFreq70] = -120

    yFilt = dftSynth(Xamp, Xphase, M)
    yFilt *= outputScaleFactor

    y = dftSynth(cleanX, Xphase, M)
    y *= outputScaleFactor

    return (y, yFilt)


def get_test_case(part_id, case_id):
    import loadTestCases
    testcase = loadTestCases.load(part_id, case_id)
    return testcase


def test_case_1():
    testcase = get_test_case(4, 1)
    y, yfilt = suppressFreqDFTmodel(**testcase['input'])
    assert np.allclose(testcase['output'][0], y, atol=1e-6, rtol=0)
    assert np.allclose(testcase['output'][1], yfilt, atol=1e-6, rtol=0)


def test_case_2():
    testcase = get_test_case(4, 2)
    y, yfilt = suppressFreqDFTmodel(**testcase['input'])
    assert np.allclose(testcase['output'][0], y, atol=1e-6, rtol=0)
    assert np.allclose(testcase['output'][1], yfilt, atol=1e-6, rtol=0)

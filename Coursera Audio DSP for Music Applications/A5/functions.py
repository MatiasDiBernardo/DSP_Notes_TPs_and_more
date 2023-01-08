import numpy as np
from scipy.fftpack import fft, ifft
import math

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
    #if not (UF.isPower2(N)):  # raise error if N not a power of two, thus mX is wrong
        #raise ValueError("size of mX is not (N/2)+1")

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

def stftAnal(x, w, N, H):
    """
	Analysis of a sound using the short-time Fourier transform
	x: input array sound, w: analysis window, N: FFT size, H: hop size
	returns xmX, xpX: magnitude and phase spectra
	"""
    if (H <= 0):  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size  # size of analysis window
    hM1 = (M + 1) // 2  # half analysis window size by rounding
    hM2 = M // 2  # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))  # add zeros at the end to analyze last sample
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    xmX = []  # Initialise empty list for mX
    xpX = []  # Initialise empty list for pX
    while pin <= pend:  # while sound pointer is smaller than last sample
        x1 = x[pin - hM1:pin + hM2]  # select one frame of input sound
        mX, pX = dftAnal(x1, w, N)  # compute dft
        xmX.append(np.array(mX))  # Append output to list
        xpX.append(np.array(pX))
        pin += H  # advance sound pointer
    xmX = np.array(xmX)  # Convert to numpy array
    xpX = np.array(xpX)
    return xmX, xpX
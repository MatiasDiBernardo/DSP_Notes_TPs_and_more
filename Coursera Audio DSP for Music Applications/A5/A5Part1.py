import numpy as np
from scipy.signal import get_window
import math
import sys, os
from scipy.fftpack import fft, ifft
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'software/models/'))
#sys.path.insert(1, 'software/models')
import dftModel as DFT
import utilFunctions as UF

""" 
A5-Part-1: Minimizing the frequency estimation error of a sinusoid

Write a function that estimates the frequency of a sinusoidal signal at a given time instant. The 
function should return the estimated frequency in Hz, together with the window size and the FFT 
size used in the analysis.  

The input arguments to the function are the wav file name including the path (inputFile) containing 
the sinusoidal signal, and the frequency of the sinusoid in Hz (f). The frequency of the input sinusoid  
can range between 100Hz and 2000Hz. The function should return a three element tuple of the estimated 
frequency of the sinusoid (fEst), the window size (M) and the FFT size (N) used.

The input wav file is a stationary audio signal consisting of a single sinusoid of length >=1 second. 
Since the signal is stationary you can just perform the analysis in a single frame, for example in 
the middle of the sound file (time equal to .5 seconds). The analysis process would be to first select 
a fragment of the signal equal to the window size, M, centered at .5 seconds, then compute the DFT 
using the dftAnal function, and finally use the peakDetection and peakInterp functions to obtain the 
frequency value of the sinusoid.

Use a Blackman window for analysis and a magnitude threshold t = -40 dB for peak picking. The window
size and FFT size should be chosen such that the difference between the true frequency (f) and the 
estimated frequency (fEst) is less than 0.05 Hz for the entire allowed frequency range of the input 
sinusoid. The window size should be the minimum positive integer of the form 100*k + 1 (where k is a 
positive integer) for which the frequency estimation error is < 0.05 Hz. For a window size M, take the
FFT size (N) to be the smallest power of 2 larger than M. 

HINT: Computing M theoritically using a formula might be complex in such cases. Instead, you need to 
follow a heuristic approach to determine the optimal value of M and N for a particular f. You can iterate
over all allowed values of window size M and stop when the condition is satisfied (i.e. the frequency
estimation error < 0.05 Hz).

Test case 1: If you run your code with inputFile = '../../sounds/sine-490.wav', f = 490.0 Hz, the optimal
values are M = 1101, N = 2048, fEst = 489.963 and the freqency estimation error is 0.037.

Test case 2: If you run your code with inputFile = '../../sounds/sine-1000.wav', f = 1000.0 Hz, the optimal
values are M = 1101, N = 2048, fEst = 1000.02 and the freqency estimation error is 0.02.

Test case 3: If you run your code with inputFile = '../../sounds/sine-200.wav', f = 200.0 Hz, the optimal
values are M = 1201, N = 2048, fEst = 200.038 and the freqency estimation error is 0.038.
"""

def peakDetection(mX, t):
    """
	Detect spectral peak locations
	mX: magnitude spectrum, t: threshold
	returns ploc: peak locations
	"""

    thresh = np.where(np.greater(mX[1:-1], t), mX[1:-1], 0)  # locations above threshold
    next_minor = np.where(mX[1:-1] > mX[2:], mX[1:-1], 0)  # locations higher than the next one
    prev_minor = np.where(mX[1:-1] > mX[:-2], mX[1:-1], 0)  # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor  # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1  # add 1 to compensate for previous steps

    """
    Understanding how the algorithm works.
    example array: [2, 3, 4, 5, 7, 6, 4, 3]
    if t = 4
    thresh = [0 0 5 7 6 0]  Discard the peaks found under the threshold

    mX[1:-1]   = [3, 4, 5, 7, 6, 4]
    mX[2:]     = [4, 5, 7, 6, 4, 3]
    next_minor = [0, 0, 0, 7, 6, 4]  Where condition 3 > 4 is false then 0

    mX[1:-1]   = [3, 4, 5, 7, 6, 4]
    mX[:-2]    = [2, 3, 4, 5, 7, 6]
    next_minor = [3, 4, 5, 7, 0, 0]  Where condition 6 > 7 is false then 0

    ploc = [0, 0, 0, 343, 0, 0]  When multiplying all conditions only the peaks are non zeros

    ploc.nonzero() + 1 = [4] Return the indices where the original ploc is non zero and then add 1
    to compensate for the shifting in the previous steps

    return a list of index where a peak is found under the conditions specified
    """
    return ploc

def peakInterp(mX, pX, ploc):
    """
	Interpolate peak values using parabolic interpolation
	mX, pX: magnitude and phase spectrum, ploc: locations of peaks
	returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
	"""

    val = mX[ploc]  # magnitude of peak bin
    lval = mX[ploc - 1]  # magnitude of bin at left
    rval = mX[ploc + 1]  # magnitude of bin at right
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)  # center of parabola
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)  # magnitude of peaks
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)  # phase of peaks by linear interpolation
    return iploc, ipmag, ipphase


def estimateNSize(M):
    """
    A better implementation
    N = int(pow(2, np.ceil(np.log2(M))))  # FFT Size, power of 2 larger than M
    """
    N = 128  #First posible N
    for i in range(8,18): #Posible FFT N size
        if M < N:
            return N
        N = 2**i


def minFreqEstErr(inputFile, f):
    """
    Inputs:
            inputFile (string) = wav file including the path
            f (float) = frequency of the sinusoid present in the input audio signal (Hz)
    Output:
            fEst (float) = Estimated frequency of the sinusoid (Hz)
            M (int) = Window size
            N (int) = FFT size
    """
    # analysis parameters:
    fs, x = UF.wavread(inputFile)

    window = 'blackman'
    t = -40
    freqPredict = 0
    count = 1

    while abs(f - freqPredict) > 0.05:
        M = 100 * count + 1
        w = get_window(window, M)
        
        xFrame = x[fs//2 - M//2: fs//2 + M//2 + 1]
        N = estimateNSize(M)

        xMag, xPhase = DFT.dftAnal(xFrame, w, N)

        peaksIndex = peakDetection(xMag, t)

        indexInter, fPeaksInter, phaseInter = peakInterp(xMag, xPhase, peaksIndex)

        freqPredict = fs * indexInter/float(N)

        count += 1

    return float(freqPredict), int(M), int(N)

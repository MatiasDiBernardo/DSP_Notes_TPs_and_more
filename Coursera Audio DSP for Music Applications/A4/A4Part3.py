import os
import sys
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
from utils import wavread

#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
#import stft
#import utilFunctions as UF



"""
A4-Part-3: Computing band-wise energy envelopes of a signal

Write a function that computes band-wise energy envelopes of a given audio signal by using the STFT.
Consider two frequency bands for this question, low and high. The low frequency band is the set of 
all the frequencies between 0 and 3000 Hz and the high frequency band is the set of all the 
frequencies between 3000 and 10000 Hz (excluding the boundary frequencies in both the cases). 
At a given frame, the value of the energy envelope of a band can be computed as the sum of squared 
values of all the frequency coefficients in that band. Compute the energy envelopes in decibels. 

Refer to "A4-STFT.pdf" document for further details on computing bandwise energy.

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N) and hop size (H). The function should return a numpy 
array with two columns, where the first column is the energy envelope of the low frequency band and 
the second column is that of the high frequency band.

Use stft.stftAnal() to obtain the STFT magnitude spectrum for all the audio frames. Then compute two 
energy values for each frequency band specified. While calculating frequency bins for each frequency 
band, consider only the bins that are within the specified frequency range. For example, for the low 
frequency band consider only the bins with frequency > 0 Hz and < 3000 Hz (you can use np.where() to 
find those bin indexes). This way we also remove the DC offset in the signal in energy envelope 
computation. The frequency corresponding to the bin index k can be computed as k*fs/N, where fs is 
the sampling rate of the signal.

To get a better understanding of the energy envelope and its characteristics you can plot the envelopes 
together with the spectrogram of the signal. You can use matplotlib plotting library for this purpose. 
To visualize the spectrogram of a signal, a good option is to use colormesh. You can reuse the code in
sms-tools/lectures/4-STFT/plots-code/spectrogram.py. Either overlay the envelopes on the spectrogram 
or plot them in a different subplot. Make sure you use the same range of the x-axis for both the 
spectrogram and the energy envelopes.

NOTE: Running these test cases might take a few seconds depending on your hardware.

Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.

In addition to comparing results with the expected output, you can also plot your output for these 
test cases.You can clearly notice the sharp attacks and decay of the piano notes for test case 1 
(See figure in the accompanying pdf). You can compare this with the output from test case 2 that 
uses a larger window. You can infer the influence of window size on sharpness of the note attacks 
and discuss it on the forums.
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


def computeEngEnv(inputFile, window, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, 
                hamming, blackman, blackmanharris)
            M (integer): analysis window size (odd positive integer)
            N (integer): FFT size (power of 2, such that N > M)
            H (integer): hop size for the stft computation
    Output:
            The function should return a numpy array engEnv with shape Kx2, K = Number of frames
            containing energy envelop of the signal in decibles (dB) scale
            engEnv[:,0]: Energy envelope in band 0 < f < 3000 Hz (in dB)
            engEnv[:,1]: Energy envelope in band 3000 < f < 10000 Hz (in dB)
    """
    
    #fs, audio = wavfile.read(inputFile)
    fs, audio = wavread(inputFile)
    w = get_window(window, M) 

    xmX, xpX = stftAnal(audio, w, N, H)
    mXlinear = 10.0 ** (xmX / 20.0)  #Make linear for energy calculations
    
    """
    Manually find the cut indices
    f_cut = 3000
    freqRange = [i * fs/N for i in range(N)]
    for k in range(N):
        if freqRange[k] > f_cut:
                cutIndex = k
                break

    for k in range(N):
        if freqRange[k] > 10000:
                upperCutIndex = k
                break
    """
    #With np.where
    bins = np.arange(0, N) * fs / N
    band_low_bins = np.where((bins > 0) & (bins < 3000.0))[0]
    band_high_bins = np.where((bins > 3000) & (bins < 10000.0))[0]

    num_frames = xmX.shape[0]
    env = np.zeros(shape=(num_frames, 2))

    for frame in range(num_frames):
        env[frame, 0] = 10.0 * np.log10(sum(mXlinear[frame, band_low_bins] ** 2))
        env[frame, 1] = 10.0 * np.log10(sum(mXlinear[frame, band_high_bins] ** 2))

    _, _, true_env = compute_eng_env(inputFile, window, M, N, H)

    return true_env

def compute_eng_env(inputFile, window, M, N, H):

    fs, x = wavread(inputFile)
    w = get_window(window, M, False)  #Without it the numerical error gives me wrongs answer

    mX, pX = stftAnal(x, w, N, H)
    mXlinear = 10.0 ** (mX / 20.0)

    # Get an array of indices for bins within each band range:

    # Using list comprehension:
    # band_low_bins = np.array([ k for k in range(N) if 0 < k * fs / N < 3000.0])
    # band_high_bins = np.array([ k for k in range(N) if 3000.0 < k * fs / N < 10000.0])

    # Using np.where():
    bins = np.arange(0, N) * fs / N
    band_low_bins = np.where((bins > 0) & (bins < 3000.0))[0]
    band_high_bins = np.where((bins > 3000) & (bins < 10000.0))[0]

    num_frames = mX.shape[0]
    env = np.zeros(shape=(num_frames, 2))

    for frame in range(num_frames):
        env[frame, 0] = 10.0 * np.log10(sum(mXlinear[frame, band_low_bins] ** 2))
        env[frame, 1] = 10.0 * np.log10(sum(mXlinear[frame, band_high_bins] ** 2))

    return fs, mX, env

"""
Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.
"""
def get_test_case(part_id, case_id):
    import loadTestCases
    testcase = loadTestCases.load(part_id, case_id)
    return testcase

def test_case_1():
    testcase = get_test_case(3, 1)
    engEnv = computeEngEnv(**testcase['input'])

    assert np.allclose(testcase['output'], engEnv, atol=1e-6, rtol=0)


def test_case_2():
    testcase = get_test_case(3, 2)
    engEnv = computeEngEnv(**testcase['input'])

    assert np.allclose(testcase['output'], engEnv, atol=1e-6, rtol=0)


def test_case_3():
    testcase = get_test_case(3, 3)
    engEnv = computeEngEnv(**testcase['input'])

    assert np.allclose(testcase['output'], engEnv, atol=1, rtol=0)

test_case_1()
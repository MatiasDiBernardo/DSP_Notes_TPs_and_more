import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt
import loadTestCases
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import harmonicModel as HM
import stft

eps = np.finfo(float).eps

#This is my first implementation and for small diferences in the indexing the answers was not consider correct

"""
A6Part2 - Segmentation of stable note regions in an audio signal

Complete the function segmentStableNotesRegions() to identify the stable regions of notes in a specific 
monophonic audio signal. The function returns an array of segments where each segment contains the 
starting and the ending frame index of a stable note.

The input argument to the function are the wav file name including the path (inputFile), threshold to 
be used for deciding stable notes (stdThsld) in cents, minimum allowed duration of a stable note (minNoteDur), 
number of samples to be considered for computing standard deviation (winStable), analysis window (window), 
window size (M), FFT size (N), hop size (H), error threshold used in the f0 detection (f0et), magnitude 
threshold for spectral peak picking (t), minimum allowed f0 (minf0) and maximum allowed f0 (maxf0). 
The function returns a numpy array of shape (k,2), where k is the total number of detected segments. 
The two columns in each row contains the starting and the ending frame indexes of a stable note segment. 
The segments must be returned in the increasing order of their start times. 

In order to facilitate the assignment we have configured the input parameters to work with a particular 
sound, '../../sounds/sax-phrase-short.wav'. The code and parameters to estimate the fundamental frequency 
is completed. Thus you start from an f0 curve obtained using the f0Detection() function and you will use 
that to obtain the note segments. 

All the steps to be implemented in order to solve this question are indicated in segmentStableNotesRegions() 
as comments. These are the steps:

1. In order to make the processing musically relevant, the f0 values should be converted first from 
Hertz to Cents, which is a logarithmic scale. 
2. At each time frame (for each f0 value) you should compute the standard deviation of the past winStable 
number of f0 samples (including the f0 sample at the current audio frame). 
3. You should then apply a deviation threshold, stdThsld, to determine if the current frame belongs 
to a stable note region or not. Since we are interested in the stable note regions, the standard 
deviation of the previous winStable number of f0 samples (including the current sample) should be less 
than stdThsld i.e. use the current sample and winStable-1 previous samples. Ignore the first winStable-1 
samples in this computation.
4. All the consecutive frames belonging to the stable note regions should be grouped together into 
segments. For example, if the indexes of the frames corresponding to the stable note regions are 
3,4,5,6,12,13,14, we get two segments, first 3-6 and second 12-14. 
5. After grouping frame indexes into segments filter/remove the segments which are smaller in duration 
than minNoteDur. Return the segment indexes in the increasing order of their start frame index.
                              
Test case 1: Using inputFile='../../sounds/cello-phrase.wav', stdThsld=10, minNoteDur=0.1, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return 9 segments. Please use loadTestcases.load() 
to check the expected segment indexes in the output.

Test case 2: Using inputFile='../../sounds/cello-phrase.wav', stdThsld=20, minNoteDur=0.5, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return 6 segments. Please use loadTestcases.load() 
to check the expected segment indexes in the output.

Test case 3: Using inputFile='../../sounds/sax-phrase-short.wav', stdThsld=5, minNoteDur=0.6, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return just one segment. Please use loadTestcases.load() 
to check the expected segment indexes in the output. 

We also provide the function plotSpectogramF0Segments() to plot the f0 contour and the detected 
segments on the top of the spectrogram of the audio signal in order to visually analyse the outcome 
of your function. Depending on the analysis parameters and the capabilities of the hardware you 
use, the function might take a while to run (even half a minute in some cases). 

"""

def segmentStableNotesRegions(inputFile = '../../sounds/sax-phrase-short.wav', stdThsld=10, minNoteDur=0.1, 
                              winStable = 3, window='hamming', M=1024, N=2048, H=256, f0et=5.0, t=-100, 
                              minf0=310, maxf0=650):
    """
    Function to segment the stable note regions in an audio signal
    Input:
        inputFile (string): wav file including the path
        stdThsld (float): threshold for detecting stable regions in the f0 contour (in cents)
        minNoteDur (float): minimum allowed segment length (note duration)  
        winStable (integer): number of samples used for computing standard deviation
        window (string): analysis window
        M (integer): window size used for computing f0 contour
        N (integer): FFT size used for computing f0 contour
        H (integer): Hop size used for computing f0 contour
        f0et (float): error threshold used for the f0 computation
        t (float): magnitude threshold in dB used in spectral peak picking
        minf0 (float): minimum fundamental frequency in Hz
        maxf0 (float): maximum fundamental frequency in Hz
    Output:
        segments (np.ndarray): Numpy array containing starting and ending frame indexes of every segment.
    """
    fs, x = UF.wavread(inputFile)                               # reading inputFile
    w = get_window(window, M)                                   # obtaining analysis window
    f0 = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)  # estimating F0

    # 1. convert f0 values from Hz to Cents (as described in pdf document)
    f0_cents = np.maximum(1200.0 * np.log2(f0 / 55.0), 0.0)

    # 2. create an array containing standard deviation of last winStable samples
    sd = np.zeros(len(f0_cents))
    for i in range(winStable, len(f0_cents)):
        sd[i] = np.std(f0_cents[i-winStable:i])

    # 3. apply threshold on standard deviation values to find indexes of the stable points in melody
    stable_indices = np.where(sd < stdThsld)[0]

    # 4. create segments of continuous stable points such that consecutive stable points belong to same segment
    all_segments = np.empty(shape=(0, 2))
    start = None
    for i in range(1, len(stable_indices)):
        if stable_indices[i] == stable_indices[i - 1] + 1:
            if start is None:
                start = i - 1
        else:
            if start is not None:
                first_index = stable_indices[start] - 1
                last_index_inclusive = stable_indices[i - 1] - 1
                segment = np.array([[first_index, last_index_inclusive]])
                all_segments = np.concatenate((all_segments, segment))
                start = None

    # 5. apply segment filtering, i.e. remove segments with are < minNoteDur in length
    minNoteDurSamples = fs * minNoteDur
    minNoteDurFrames = minNoteDurSamples / H
    segments = np.array([x for x in all_segments if x[1] - x[0] > minNoteDurFrames])

    #plotSpectogramF0Segments(x, fs, w, N, H, f0, segments)  # Plot spectrogram and F0 if needed

    # return segments
    return segments

def plotSpectogramF0Segments(x, fs, w, N, H, f0, segments):
    """
    Code for plotting the f0 contour on top of the spectrogram
    """
    # frequency range to plot
    maxplotfreq = 1000.0    
    fontSize = 16

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mX, pX = stft.stftAnal(x, w, N, H)                      #using same params as used for analysis
    mX = np.transpose(mX[:,:int(N*(maxplotfreq/fs))+1])
    
    timeStamps = np.arange(mX.shape[1])*H/float(fs)                             
    binFreqs = np.arange(mX.shape[0])*fs/float(N)
    
    plt.pcolormesh(timeStamps, binFreqs, mX)
    plt.plot(timeStamps, f0, color = 'k', linewidth=2)

    for ii in range(segments.shape[0]):
        plt.plot(timeStamps[int(segments[ii,0]):int(segments[ii,1])], f0[int(segments[ii,0]):int(segments[ii,1])], color = '#A9E2F3', linewidth=5)        
    
    plt.autoscale(tight=True)
    plt.ylabel('Frequency (Hz)', fontsize = fontSize)
    plt.xlabel('Time (s)', fontsize = fontSize)
    plt.legend(('f0','segments'))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    ax.set_aspect((xLim[1]-xLim[0])/(2.0*(yLim[1]-yLim[0])))    
    plt.autoscale(tight=True) 
    #plt.show()
    
"""
Test case 1: Using inputFile='../../sounds/cello-phrase.wav', stdThsld=10, minNoteDur=0.1, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return 9 segments. Please use loadTestcases.load() 
to check the expected segment indexes in the output.

Test case 2: Using inputFile='../../sounds/cello-phrase.wav', stdThsld=20, minNoteDur=0.5, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return 6 segments. Please use loadTestcases.load() 
to check the expected segment indexes in the output.

Test case 3: Using inputFile='../../sounds/sax-phrase-short.wav', stdThsld=5, minNoteDur=0.6, 
winStable = 3, window='hamming', M=1025, N=2048, H=256, f0et=5.0, t=-100, minf0=310, maxf0=650, 
the function segmentStableNotesRegions() should return just one segment. Please use loadTestcases.load() 
to check the expected segment indexes in the output. 

"""

#in2 = '../../sounds/cello-phrase.wav'

#segmentStableNotesRegions(inputFile=in2, stdThsld=20, minNoteDur=0.5)

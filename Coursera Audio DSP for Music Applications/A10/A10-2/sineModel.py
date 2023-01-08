# functions that implement analysis and synthesis of sounds using the Sinusoidal Model
# (for example usage check the examples in models_interface)

import numpy as np
from scipy.signal.windows import blackmanharris, triang
from scipy.fftpack import ifft, fftshift
import math
import dftModel as DFT
import utilFunctions as UF

def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
    """
	Tracking sinusoids from one frame to the next
	pfreq, pmag, pphase: frequencies and magnitude of current frame
	tfreq: frequencies of incoming tracks from previous frame
	freqDevOffset: minimum frequency deviation at 0Hz 
	freqDevSlope: slope increase of minimum frequency deviation
	returns tfreqn, tmagn, tphasen: frequency, magnitude and phase of tracks
	"""

    tfreqn = np.zeros(tfreq.size)  # initialize array for output frequencies
    tmagn = np.zeros(tfreq.size)  # initialize array for output magnitudes
    tphasen = np.zeros(tfreq.size)  # initialize array for output phases
    pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]  # indexes of current peaks
    incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0]  # indexes of incoming tracks
    newTracks = np.zeros(tfreq.size, dtype=np.int) - 1  # initialize to -1 new tracks
    magOrder = np.argsort(-pmag[pindexes])  # order current peaks by magnitude
    pfreqt = np.copy(pfreq)  # copy current peaks to temporary array
    pmagt = np.copy(pmag)  # copy current peaks to temporary array
    pphaset = np.copy(pphase)  # copy current peaks to temporary array

    # continue incoming tracks
    if incomingTracks.size > 0:  # if incoming tracks exist
        for i in magOrder:  # iterate over current peaks
            if incomingTracks.size == 0:  # break when no more incoming tracks
                break
            track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))  # closest incoming track to peak
            freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]])  # measure freq distance
            if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):  # choose track if distance is small
                newTracks[incomingTracks[track]] = i  # assign peak index to track index
                incomingTracks = np.delete(incomingTracks, track)  # delete index of track in incomming tracks
    indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]  # indexes of assigned tracks
    if indext.size > 0:
        indexp = newTracks[indext]  # indexes of assigned peaks
        tfreqn[indext] = pfreqt[indexp]  # output freq tracks
        tmagn[indext] = pmagt[indexp]  # output mag tracks
        tphasen[indext] = pphaset[indexp]  # output phase tracks
        pfreqt = np.delete(pfreqt, indexp)  # delete used peaks
        pmagt = np.delete(pmagt, indexp)  # delete used peaks
        pphaset = np.delete(pphaset, indexp)  # delete used peaks

    # create new tracks from non used peaks
    emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[0]  # indexes of empty incoming tracks
    peaksleft = np.argsort(-pmagt)  # sort left peaks by magnitude
    if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):  # fill empty tracks
        tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
        tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
        tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
    elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):  # add more tracks if necessary
        tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
        tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
        tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
        tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
        tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
        tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
    return tfreqn, tmagn, tphasen


def cleaningSineTracks(tfreq, minTrackLength=3):
    """
	Delete short fragments of a collection of sinusoidal tracks 
	tfreq: frequency of tracks
	minTrackLength: minimum duration of tracks in number of frames
	returns tfreqn: output frequency of tracks
	"""

    if tfreq.shape[1] == 0:  # if no tracks return input
        return tfreq
    nFrames = tfreq[:, 0].size  # number of frames
    nTracks = tfreq[0, :].size  # number of tracks in a frame
    for t in range(nTracks):  # iterate over all tracks
        trackFreqs = tfreq[:, t]  # frequencies of one track
        trackBegs = np.nonzero((trackFreqs[:nFrames - 1] <= 0)  # begining of track contours
                               & (trackFreqs[1:] > 0))[0] + 1
        if trackFreqs[0] > 0:
            trackBegs = np.insert(trackBegs, 0, 0)
        trackEnds = np.nonzero((trackFreqs[:nFrames - 1] > 0)  # end of track contours
                               & (trackFreqs[1:] <= 0))[0] + 1
        if trackFreqs[nFrames - 1] > 0:
            trackEnds = np.append(trackEnds, nFrames - 1)
        trackLengths = 1 + trackEnds - trackBegs  # lengths of trach contours
        for i, j in zip(trackBegs, trackLengths):  # delete short track contours
            if j <= minTrackLength:
                trackFreqs[i:i + j] = 0
    return tfreq


def sineModel(x, fs, w, N, t):
    """
	Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
	x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
	returns y: output array sound
	"""

    hM1 = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    Ns = 512  # FFT size for synthesis (even)
    H = Ns // 4  # Hop size used for analysis and synthesis
    hNs = Ns // 2  # half of synthesis FFT size
    pin = max(hNs, hM1)  # init sound pointer in middle of anal window
    pend = x.size - max(hNs, hM1)  # last sample to start a frame
    yw = np.zeros(Ns)  # initialize output sound frame
    y = np.zeros(x.size)  # initialize output array
    w = w / sum(w)  # normalize analysis window
    sw = np.zeros(Ns)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hNs - H:hNs + H] = ow  # add triangular window
    bh = blackmanharris(Ns)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window
    while pin < pend:  # while input sound pointer is within sound
        # -----analysis-----
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # detect locations of peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values by interpolation
        ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
        # -----synthesis-----
        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)  # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))  # compute inverse FFT
        yw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
        yw[hNs - 1:] = fftbuffer[:hNs + 1]
        y[pin - hNs:pin + hNs] += sw * yw  # overlap-add and apply a synthesis window
        pin += H  # advance sound pointer
    return y

def sineModel3(x, fs, w, N, t, B):
    """
	Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
	x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
	returns y: output array sound
	"""

    hM1 = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    Ns = 512  # FFT size for synthesis (even)
    H = Ns // 4  # Hop size used for analysis and synthesis
    hNs = Ns // 2  # half of synthesis FFT size
    pin = max(hNs, hM1)  # init sound pointer in middle of anal window
    pend = x.size - max(hNs, hM1)  # last sample to start a frame
    yw = np.zeros(Ns)  # initialize output sound frame
    y = np.zeros(x.size)  # initialize output array
    w = w / sum(w)  # normalize analysis window
    sw = np.zeros(Ns)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hNs - H:hNs + H] = ow  # add triangular window
    bh = blackmanharris(Ns)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window
    Y_spec = np.array([])
    while pin < pend:  # while input sound pointer is within sound
        # -----analysis-----
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # detect locations of peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values by interpolation
        ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
        indexFreq = np.where((ipfreq >= B[0]) & (ipfreq < B[1]))
        # -----synthesis-----
        Y = UF.genSpecSines(ipfreq[indexFreq], ipmag[indexFreq], ipphase[indexFreq], Ns, fs)  # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))  # compute inverse FFT
        yw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
        yw[hNs - 1:] = fftbuffer[:hNs + 1]
        y[pin - hNs:pin + hNs] += sw * yw  # overlap-add and apply a synthesis window
        Y_spec = np.append(Y_spec, Y)
        pin += H  # advance sound pointer
    return y, Y_spec

def sineModelMultiresFinal(x, fs, w, N, t, B):

    y = []
    Y = []

    for i in range(len(w)):
        ySynth, Yspec = sineModel3(x, fs, w[i], N[i], t, B[i])

        y.append(ySynth)
        Y.append(Yspec)

    finalAudio = 0
    for audio in y:
        finalAudio += audio
    
    #finalAudio /= len(B)

    return finalAudio , Y

def sineModelMultires(x, fs, w, N, t, B):
    y = []

    for i in range(len(w)):
        ySynth= sineModel3(x, fs, w[i], N[i], t, B[i])
        y.append(ySynth)

    finalAudio = 0
    for audio in y:
        finalAudio += audio
    
    return finalAudio

"""
In the code of other people they separate all the process os windows in the same while loop and then
separete the ipfreq, ipamp, ipphase in the frequency index according to B and append the arrays so at the end
with only one synthesis the process is done. The only dude I have with this approach is that the pin calculation depend
on the window size but the result is the same so I think the implementations are equivalent.
"""
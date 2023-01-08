import matplotlib.pyplot as plt
import numpy as np
import utilFunctions as UF
import dftModel as DFT
import harmonicModel as HM
import hpsModel as HPS
from scipy.signal import get_window

#First I analize the sound with SonicVisualizer and define the parameters

inputFile = '../../sounds/speech-female.wav'

fs, x = UF.wavread(inputFile)

winType = "blackman"
M = 1765  #6 * 44100/150 = 1764
w = get_window(winType, M)

N = 2048
t = -100
nH = 50
f0et = 5.0
f0Min = 100
f0Max = 300
stoF = 0.1

outputSound, harmonicSound, stochasticSound = HPS.hpsModel(x, fs, w, N, t, nH, f0Min, f0Max, f0et, stoF)

UF.wavwrite(outputSound, fs, "speech-reconstructed.wav")
UF.wavwrite(harmonicSound, fs, "speech-harmonic.wav")
UF.wavwrite(stochasticSound, fs, "speech-stochastic.wav")

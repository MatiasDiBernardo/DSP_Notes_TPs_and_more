import matplotlib.pyplot as plt
import numpy as np
import utilFunctions as UF
import dftModel as DFT
import harmonicModel as HM
import hpsModel as HPS
from scipy.signal import get_window

#First I analize the sound with SonicVisualizer and define the parameters

inputFile = '../../sounds/violin-short.wav'

fs, x = UF.wavread(inputFile)

winType = "blackman"
M = 631
w = get_window(winType, M)

N = 2048
t = -80
nH = 15
f0et = 5.0
f0Min = 380
f0Max = 680
stoF = 0.1

outputSound, harmonicSound, stochasticSound = HPS.hpsModel(x, fs, w, N, t, nH, f0Min, f0Max, f0et, stoF)

UF.wavwrite(outputSound, fs, "a7p2-reconstructed.wav")
UF.wavwrite(harmonicSound, fs, "a7p2-harmonic.wav")
UF.wavwrite(stochasticSound, fs, "a7p2-stochastic.wav")

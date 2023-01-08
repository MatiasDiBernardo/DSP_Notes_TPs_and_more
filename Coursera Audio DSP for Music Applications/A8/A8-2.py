import numpy as np
from scipy.signal import get_window
import utilFunctions as UF
import harmonicTransformations as HT
import hpsTransformations as HPST
import hpsModel as HPS

inputFile = '../../sounds/violin-short.wav'

fs, x = UF.wavread(inputFile)

print("Timepo song: ", len(x)/fs)

winType = "blackman"
M = 631  #6 * 44100/150 = 1764
w = get_window(winType, M)

N = 2048
t = -80
nH = 25
f0et = 5.0
f0Min = 380
f0Max = 680
stoF = 0.1
Ns = 512
H = Ns // 4  

hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, f0Min, f0Max, 5, 0.1, 0.1, Ns, stoF)

#Freq scaling
freqScaling = np.array([0, 1.5, 1, 1.5])
freqStretching = np.array([0,0.8,1,0.8])
timbrePreservation = 0.9

hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)

# time scaling the sound
timeScaling = np.array([0, 0, 1, 0.8])
yhfreq, yhmag, ystocEnv = HPST.hpsTimeScale(hfreqt, hmagt, stocEnv, timeScaling)

# synthesis from the trasformed hps representation 
#Remember that if I pass empty in the phase index the phase is automatically generated
y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)

UF.wavwrite(y,fs, "a8p2-transformed-3.wav")
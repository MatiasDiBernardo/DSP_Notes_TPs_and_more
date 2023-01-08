import numpy as np
from scipy.signal import get_window
import utilFunctions as UF
import harmonicTransformations as HT
import hpsTransformations as HPST
import hpsModel as HPS

inputFile = '../../sounds/speech-female.wav'

fs, x = UF.wavread(inputFile)

print("Timepo song: ", len(x)/fs)

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
Ns = 512
H = Ns // 4  

hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, f0Min, f0Max, 5, 0.1, 0.1, Ns, stoF)

#Freq scaling
#freqScaling = np.array([0, 1, 0.5, 2, 1, 0.5])
freqScaling = np.array([0, 1, 1, 1])
freqStretching = np.array([0,1.5,1,2])
timbrePreservation = 1

hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)

# time scaling the sound
timeScaling = np.array([0,0,1,2])
yhfreq, yhmag, ystocEnv = HPST.hpsTimeScale(hfreqt, hmagt, stocEnv, timeScaling)

# synthesis from the trasformed hps representation 
#Remember that if I pass empty in the phase index the phase is automatically generated
y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)

UF.wavwrite(y,fs, "speech-transformed-5.wav")


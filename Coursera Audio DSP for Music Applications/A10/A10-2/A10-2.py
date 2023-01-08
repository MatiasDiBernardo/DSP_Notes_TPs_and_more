import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sineModel as SM
import utilFunctions as UF

"""
B1: 0 <= f < 1000Hz, B2: 1000 <= f < 5000, B3: 5000 <= f < 22050, and three window sizes M1 = 4095, M2 = 2047, M3 = 1023, 
we generate three windows w1, w2, w3 with sizes M1, M2, M3 respectively. For every frame of audio x1 = x[pin-hM1:pin+hM2], 
we compute three DFTs X1 = dftAnal(x1, w1, N1), X2 = dftAnal(x2, w2, N2) and X3 = dftAnal(x3, w3, N3). Choose N1, N2 and N3 as needed.
"""

fs, x = UF.wavread("pianmono.wav")
x = x[:5*fs]

B = [(0, 500), (500, 3000), (3000, 22050)]
w1 = get_window("blackman", 4070)
w2 = get_window("blackman", 2001)
w3 = get_window("blackman", 1001)
N = [4096, 2048, 1024]
w = [w1, w2, w3]

y, Y = SM.sineModelMultiresFinal(x, fs, w, N, -80, B)


UF.wavwrite(y, fs, "pianogod.wav")




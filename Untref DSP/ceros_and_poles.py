import numpy as np
import matplotlib.pyplot as plt

def e(w, root):
    return (np.exp(1j*w) - root)

def transfer_function(zeros, poles):
    res = 1000
    w = np.linspace(-np.pi, np.pi, res)

    num = np.ones(res, dtype=complex)
    for zero in zeros:
        num *= e(w, zero)

    den = np.ones(res, dtype=complex)
    for pole in poles:
        den *= e(w, pole)
    
    H_z = num/den

    mag = np.abs(H_z)
    ang = np.angle(H_z)

    return mag, ang

def plot_mag_phase(mag, phase, db=True, half=True):
    w = np.linspace(-np.pi, np.pi, len(mag))
    if half:
        cut = len(mag)//2
    else:
        cut = 0

    fig, ax = plt.subplots(2)
    if db:
        ax[0].plot(w[cut:], 20*np.log10(mag[cut:]))
    else:
        ax[0].plot(w[cut:], mag[cut:])
    ax[1].plot(w[cut:], phase[cut:])
    ax[1].set_ylim(-np.pi, np.pi)
    plt.show()
    
zeros = [0.5]
poles = [0]

mag, phase = transfer_function(zeros, poles)
plot_mag_phase(mag, phase)


    

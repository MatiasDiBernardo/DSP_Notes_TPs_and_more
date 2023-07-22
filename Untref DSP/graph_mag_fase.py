import numpy as np
import matplotlib.pyplot as plt

def plot_z_plane(num, den):
    ceros = np.roots(num)
    poles = np.roots(den)

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    #ax.set_ylim(0,1.2)
    ax.scatter(np.angle(ceros), np.abs(ceros), marker="o")
    ax.scatter(np.angle(poles), np.abs(poles), marker="x")
    plt.show()

def plot_rta(func, db=False, half=False):
    mag = np.abs(func)
    phase = np.angle(func)
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

w = np.linspace(-np.pi, np.pi, 1000)
z = np.exp(1j*w)
z_1 = np.exp(-1j*w)

alpha = np.sqrt(2)/4 + 1j*(np.sqrt(2)/4)  
alpha_conj = np.conjugate(alpha)
beta = -0.62

#Escribir funcion
#func = (z - beta)/(z-alpha)
#func = (((z*alpha_conj) - 1)/(z-alpha) + 1)*1/2
func = (z**4 + 4.25*z**2 + 1)/(z**2)

#Coeficientes de los polinomios (del mas grande al mas chico)
num = [-beta,1]
den = [1,-beta]

plot_rta(func)
plot_z_plane(num, den)
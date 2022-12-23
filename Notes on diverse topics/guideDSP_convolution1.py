import numpy as np
import matplotlib.pyplot as plt

#Esta sería la implementación desde el punto de vista de la señal de entrada 
#Idea crear una applet que te perimita crear señales discretas con poner el número de muestras y mover sliders

t = np.linspace(0,5,40)

x = np.sin(2*np.pi*2*t)

x = np.load('x2.npy')
print(len(x))

h = np.hstack([np.linspace(0,2,10), np.linspace(0,2,10)[::-1]])

len_conv = len(x) + len(h) - 1

y = np.zeros(len_conv)

for i in range(len(x)):  #Loopea toda la señal
	for j in range(len(h)):  #Por cada muestra de la señal, multiplica y desplaza la rta al impulso. Pag 129 del libro
		y[i+j] = y[i+j] + x[i] * h[j]

plt.figure('Metodo 1')
plt.plot(y)

#Implementación desde el punto de vista de la señal de salida

y2 = np.zeros(len_conv)

for i in range(len_conv):
	for j in range(len(h)):
		if i - j > 0 and i - j < len(x):  #Control de border, sino se saca y se hace zero pad a la señal de entrada
			y2[i] = y2[i] + h[j] * x[i-j]


plt.figure('Metodo 2')
plt.plot(y2)
plt.show()
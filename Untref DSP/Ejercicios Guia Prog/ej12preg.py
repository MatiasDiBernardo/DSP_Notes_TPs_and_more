import numpy as np

fs = 10
T = 1/fs
seg = 0.653

t = np.linspace(0, seg, int(seg*fs))
t2 = np.arange(0, seg, T)

print(np.all(t == t2))

print(t)
print(t2)
print(len(t2))
print("-----------------------------")

#Resolver pensando muestras como puntos (hay 10 puntos en un segundo sin contar el segundo)
t = np.linspace(0, seg - T, int(seg*fs))
t2 = np.arange(0, seg, T)

print(np.all(t == t2))

print(t)
print(t2)
print(len(t2))
print("-----------------------------")

#Resolver pensando en muestras como intervalos, osea 10 intervalos en un segundo necesito 11 muestras
t = np.linspace(0, seg, int(seg*fs) + 1)
t2 = np.arange(0, seg + T, T)

print(np.all(t == t2))

print(t)
print(t2)
print(len(t2))
print("-----------------------------")

#Resolver pensando en muestras como puntos pero teniendo en cuenta el último punto
t = np.linspace(0, seg, int(seg*fs))
t2 = np.arange(0, seg + T, 1/(fs - 1))

print(np.all(t == t2))

print(t)
print(t2)
print(len(t2))
print("-----------------------------")


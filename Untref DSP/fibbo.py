import numpy as np

n = np.arange(12)
phi1 = (1+np.sqrt(5))/2
phi2 = (1-np.sqrt(5))/2

rta = phi1**(n+1) + phi2**(n+1)
rta2 = 1/np.sqrt(5) * ((-0.618)**n - (1.618**n))
rta3 = 0.724*(phi1)**n + 0.276*(phi2)**n

print(np.round(rta3, 2))
print("        ")
print(np.round(rta2[1:], 2))
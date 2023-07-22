from scipy.signal import residue

# Ej z/(z**2-z-1)
num = [0,0,1]
den = [1, -1, -1]

ceros, poles, _ = residue(num, den)

#Ceros are asigned to poles in same index
print("Ceros: ", ceros)  #A, B, etc
print("Poles: ", poles)  #Roots
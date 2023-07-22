import numpy as np
import time

#x = np.array([0, 1, 2, 1, 0])
x = np.random.random(1000)

def time_test(x_test, func):
    start_time = time.time()
    _ = func(x_test)
    finish_time = time.time()

    return finish_time - start_time

def DFT_double_for(x):
    x_k = np.zeros(len(x), dtype=complex)
    for k in range(len(x)):
        for n in range(len(x)):
            x_k[k] += x[n] * np.exp(-1j*2*np.pi*k*n/len(x))
    return x_k

def DFT_half_vectorize(x):
    x_k = np.zeros(len(x), dtype=complex)
    n = np.arange(len(x))
    for k in range(len(x)):
        x_k[k] += np.sum(x * np.exp(-1j*2*np.pi*k*n/len(x)))
    
    return x_k

def DFT_matrix(N): 
    
    M = np.exp(-1j*2*np.pi/N)
    matrix = np.ones((N,N), dtype=complex)

    for i in range(N):
        for j in range(N):
            if i != 0 and j != 0:
                matrix[i,j] = M**(i*j)
    
    return matrix

def DFT_all_vectorize(x):
    dft_matrix = DFT_matrix(len(x))

    return np.dot(x, dft_matrix)

def same_result_pass(x):
    y1 = DFT_all_vectorize(x)
    y2 = DFT_half_vectorize(x)
    y3 = DFT_double_for(x)

    if np.allclose(y1, y2) and np.allclose(y2, y3) and np.allclose(y1, y3):
        print("Equality check")
    else:
        print("Not equal")
        
#print("For: ", np.abs(DFT_double_for(x.real)))
#print("1 vec: ", np.abs(DFT_half_vectorize(x.real)))
#print("2 vec: ", np.abs(dft_all_vectorize(x.real)))

same_result_pass(x)

print("----------------------")

print(time_test(x, DFT_double_for))
print(time_test(x, DFT_half_vectorize))
print(time_test(x, DFT_all_vectorize))

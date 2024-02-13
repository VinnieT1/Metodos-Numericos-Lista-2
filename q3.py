import numpy as np

def positiva_definida(n):
    A = 100 * np.random.rand(n, n)
    A = 0.5 * (A + A.T)
    A = A + n * np.eye(n)
    print(A)
    
positiva_definida(2)
positiva_definida(3)
positiva_definida(4)
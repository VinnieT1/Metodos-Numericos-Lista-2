import numpy as np

# resolve Ax = b, onde A é uma matriz quadrada e b é um vetor; x é o vetor solução.
def eliminacao_gaussiana(A, b):
    m = len(A)
    for i in range(m):
        pivot = i
        for j in range(i + 1, m):
            if abs(A[j][i]) > abs(A[pivot][i]):
                pivot = j

        if abs(A[i][i]) < 1e-10 and i == m - 1 and b[pivot] != 0:
            print('Sem solução')
            return None
    
        if abs(A[pivot][i]) < 1e-10:
            print('Sem solução única')
            return None

        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]

        for j in range(i + 1, m):
            fator = A[j][i] / A[i][i]
            A[j] -= fator * A[i]
            b[j] -= fator * b[i]

    x = np.zeros(m)
    for i in range(m - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

    return x

def decomposicao_PtLU(A):
    m = len(A)
    U = np.copy(A)
    P = np.eye(m)
    L = np.eye(m)
    
    for k in range(m - 1):
        i = np.argmax(np.abs(U[k:, k]))
        U[k, k:m], U[i, k:m] = U[i, k:m].copy(), U[k, k:m].copy()
        L[k, 1:k - 1], L[i, 1:k - 1] = L[i, 1:k - 1].copy(), L[k, 1:k - 1].copy()
        P[k,:], P[i,:] = P[i,:].copy(), P[k,:].copy()
        
        for j in range(k + 1, m):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:m] = U[j, k:m] - L[j, k] * U[k, k:m]
    
    return P, L, U

# A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
# P, L, U = decomposicao_PtLU(A)
# print(P)

A1 = np.array([[3.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 2.0]])
b1 = np.array([1.0, 0.0, 1.0])

print(eliminacao_gaussiana(A1, b1))

A2 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
b2 = np.array([1.0, 0.0, 2.0])

print(eliminacao_gaussiana(A2, b2))

A3 = np.array([[6.0, -5.0, 0.0], [15.0, -20.0, 10.0], [0.0, 10.0, -20.0]])
b3 = np.array([1.0, -1.0, 4.0])

#6x - 5y = 1
#15x - 20y + 10z = -1
#10y - 20z = 4
print(eliminacao_gaussiana(A3, b3))
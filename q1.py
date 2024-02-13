import numpy as np

def eliminacao_gaussiana(A, b):
    m = len(A)
    for i in range(m):
        pivot = i
        for j in range(i + 1, m):
            if abs(A[j][i]) > abs(A[pivot][i]):
                pivot = j

        if abs(A[pivot][i]) < 1e-10 and i == m - 1 and b[pivot] != 0:
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
    U = A.copy()
    m = len(A)
    L = np.eye(m)
    P = np.eye(m)
    
    for k in range(m - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if pivot != k:
            U[[k, pivot], k:] = U[[pivot, k], k:]
            L[[k, pivot], :k] = L[[pivot, k], :k]
            P[[k, pivot], :] = P[[pivot, k], :]
        
        for i in range(k + 1, m):
            fator = U[i, k] / U[k, k]
            L[i, k] = fator
            U[i, k:] -= fator * U[k, k:]
    
    return P, L, U

# ----------------------------------- ELIMINAÇÃO GAUSSIANA -----------------------------------

A1 = np.array([[3.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 2.0]])
b1 = np.array([1.0, 0.0, 1.0])
print(eliminacao_gaussiana(A1, b1))

A2 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
b2 = np.array([1.0, 0.0, 2.0])
print(eliminacao_gaussiana(A2, b2))

A3 = np.array([[6.0, -5.0, 0.0], [15.0, -20.0, 10.0], [0.0, 10.0, -20.0]])
b3 = np.array([1.0, -1.0, 4.0])
print(eliminacao_gaussiana(A3, b3))

# ----------------------------------- DECOMPOSIÇÃO PtLU -----------------------------------

P1, L1, U1 = decomposicao_PtLU(A1)
P1b1 = np.matmul(P1, b1)
Y1 = eliminacao_gaussiana(L1, P1b1)
print('P1:', P1)
print('sol 1:', eliminacao_gaussiana(U1, Y1))

P2, L2, U2 = decomposicao_PtLU(A2)
P2b2 = np.matmul(P2, b2)
Y2 = eliminacao_gaussiana(L2, P2b2)
print('P2:', P2)
print('sol 2:', eliminacao_gaussiana(U2, Y2))

P3, L3, U3 = decomposicao_PtLU(A3)
P3b3 = np.matmul(P3, b3)
Y3 = eliminacao_gaussiana(L3, P3b3)
print('P3:', P3)
print('sol 3:', eliminacao_gaussiana(U3, Y3))

# ----------------------------------- CHOLESKY -----------------------------------

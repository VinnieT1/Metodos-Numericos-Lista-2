import numpy as np

def substituicao_inversa(A, b):
    m = len(A)
    x = np.zeros(m)
    for i in range(m - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

    return x

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

    return substituicao_inversa(A, b)

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

def cholesky_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    L[0][0] = np.sqrt(A[0][0])

    for j in range(1, n):
        L[j][0] = A[j][0] / L[0][0]
    for i in range(1, n - 1):
        L[i][i] = np.sqrt(A[i][i] - np.sum(L[i][:i]**2))
        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - np.sum(L[j][:i] * L[i][:i])) / L[i][i]
    L[n - 1][n - 1] = np.sqrt(A[n - 1][n - 1] - np.sum(L[n - 1][:n - 1]**2))

    return L, L.T

def substituicao_direta(A, b):
    m = len(A)
    x = np.zeros(m)
    for i in range(m):
        x[i] = (b[i] - np.dot(A[i][:i], x[:i])) / A[i][i]

    return x

# ----------------------------------- ELIMINAÇÃO GAUSSIANA -----------------------------------

print('\nELIMINAÇÃO GAUSSIANA\n')

A1 = np.array([[4.0, -1.0, 1.0], [-1.0, 4.0, -1.0], [1.0, -1.0, 4.0]])
b1 = np.array([1.0, 0.0, 1.0])
print('sol 1:', eliminacao_gaussiana(A1, b1))

A2 = np.array([[0.0, -1.0, 2.0], [-1.0, 2.0, -1.0], [2.0, -1.0, 0.0]])
b2 = np.array([2.0, 0.0, 1.0])
print('sol 2:', eliminacao_gaussiana(A2, b2))

A3 = np.array([[2.0, -0.5, 0.0], [-0.5, 2.0, -1.0], [0.0, -1.0, 2.0]])
b3 = np.array([1.0, -1.0, 7.0])
print('sol 3:', eliminacao_gaussiana(A3, b3))

# ----------------------------------- DECOMPOSIÇÃO PtLU -----------------------------------

print('\nDECOMPOSIÇÃO PtLU\n')

A1 = np.array([[4.0, -1.0, 1.0], [-1.0, 4.0, -1.0], [1.0, -1.0, 4.0]])
b1 = np.array([1.0, 0.0, 1.0])
P1, L1, U1 = decomposicao_PtLU(A1)
P1b1 = np.dot(P1, b1)
Y1 = substituicao_direta(L1, P1b1)
X1 = substituicao_inversa(U1, Y1)
print('P1:', P1)
print('sol 1:', X1)

A2 = np.array([[0.0, -1.0, 2.0], [-1.0, 2.0, -1.0], [2.0, -1.0, 0.0]])
b2 = np.array([2.0, 0.0, 1.0])
P2, L2, U2 = decomposicao_PtLU(A2)
P2b2 = np.dot(P2, b2)
Y2 = substituicao_direta(L2, P2b2)
X2 = substituicao_inversa(U2, Y2)
print('P2:', P2)
print('sol 2:', X2)

A3 = np.array([[2.0, -0.5, 0.0], [-0.5, 2.0, -1.0], [0.0, -1.0, 2.0]])
b3 = np.array([1.0, -1.0, 7.0])
P3, L3, U3 = decomposicao_PtLU(A3)
P3b3 = np.dot(P3, b3)
Y3 = substituicao_direta(L3, P3b3)
X3 = substituicao_inversa(U3, Y3)
print('P3:', P3)
print('sol 3:', X3)

# ----------------------------------- CHOLESKY -----------------------------------

print('\nCHOLESKY\n')

A1 = np.array([[4.0, -1.0, 1.0], [-1.0, 4.0, -1.0], [1.0, -1.0, 4.0]])
b1 = np.array([1.0, 0.0, 1.0])
L1, L1t = cholesky_decomposition(A1)
Y1 = substituicao_direta(L1, b1)
X1 = substituicao_inversa(L1t, Y1)
print('sol 1:', X1)

A2 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
b2 = np.array([1.0, 0.0, 2.0])
L2, L2t = cholesky_decomposition(A2)
Y2 = substituicao_direta(L2, b2)
X2 = substituicao_inversa(L2t, Y2)
print('sol 2:', X2)


A3 = np.array([[2.0, -0.5, 0.0], [-0.5, 2.0, -1.0], [0.0, -1.0, 2.0]])
b3 = np.array([1.0, -1.0, 7.0])
L3, L3t = cholesky_decomposition(A3)
Y3 = substituicao_direta(L3, b3)
X3 = substituicao_inversa(L3t, Y3)
print('sol 3:', X3)
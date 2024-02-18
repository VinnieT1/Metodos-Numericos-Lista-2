import numpy as np
from time import time
from matplotlib import pyplot as plt

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

def positiva_definida(n):
    A = 100 * np.random.rand(n, n)
    A = 0.5 * (A + A.T)
    A = A + n * np.eye(n)
    return A

valores_de_n = [10, 100, 500, 1000]

gauss_a, t_gauss_a = [], []
ptlu_a, t_ptlu_a = [], []
gauss_b, t_gauss_b = [], []
ptlu_b, t_ptlu_b = [], []

# 10 matrizes e 10 vetores
print('\n----- Letra a) 10 matrizes e 10 vetores -----\n')
for n in valores_de_n:
    inicio = time()
    for i in range(10):
        t1 = time()
        A = positiva_definida(n)
        b = 100 * np.random.rand(n)
        t2 = time()
        inicio -= (t2 - t1) # descontando o tempo para gerar a matriz e o vetor

        x = eliminacao_gaussiana(A, b)
    fim = time()

    print(f'Eliminação Gaussiana p/ n = {n}: {fim - inicio}s')
    gauss_a.append(n)
    t_gauss_a.append(fim - inicio)

    inicio = time()
    for i in range(10):
        t1 = time()
        A = positiva_definida(n)
        b = 100 * np.random.rand(n)
        t2 = time()
        inicio -= (t2 - t1) # descontando o tempo para gerar a matriz e o vetor

        P, L, U = decomposicao_PtLU(A)
        Pb = np.dot(P, b)
        y = substituicao_inversa(L, Pb)
        x = substituicao_inversa(U, y)
    fim = time()

    print(f'Decomposição PtLU p/ n = {n}: {fim - inicio}s')
    ptlu_a.append(n)
    t_ptlu_a.append(fim - inicio)

# 1 matriz e 10 vetores
print('\n----- Letra b) 1 matriz e 10 vetores -----\n')
for n in valores_de_n:
    A = positiva_definida(n)

    inicio = time()
    for i in range(10):
        t1 = time()
        b = 100 * np.random.rand(n)
        t2 = time()
        inicio -= (t2 - t1) # descontando o tempo para gerar a matriz e o vetor

        x = eliminacao_gaussiana(A, b)
    fim = time()

    print(f'Eliminação Gaussiana p/ n = {n}: {fim - inicio}s')
    gauss_b.append(n)
    t_gauss_b.append(fim - inicio)

    P, L, U = None, None, None
    inicio = time()
    for i in range(10):
        t1 = time()
        b = 100 * np.random.rand(n)
        t2 = time()
        inicio -= (t2 - t1) # descontando o tempo para gerar a matriz e o vetor

        if P is None:
            P, L, U = decomposicao_PtLU(A)

        Pb = np.dot(P, b)
        y = substituicao_inversa(L, Pb)
        x = substituicao_inversa(U, y)
    fim = time()

    print(f'Decomposição PtLU p/ n = {n}: {fim - inicio}s')
    ptlu_b.append(n)
    t_ptlu_b.append(fim - inicio)

figure, axis = plt.subplots(2, 1)
axis[0].plot(gauss_a, t_gauss_a, label='Eliminação Gaussiana (10 matrizes e 10 vetores)', marker='o')
axis[0].plot(ptlu_a, t_ptlu_a, label='Decomposição PtLU (10 matrizes e 10 vetores)', marker='o')
axis[0].set_title('10 matrizes e 10 vetores')
axis[0].set_xlabel('n')
axis[0].set_ylabel('Tempo (s)')
axis[0].legend(['Eliminação Gaussiana', 'Decomposição PtLU'])
axis[0].grid(True)

axis[1].plot(gauss_b, t_gauss_b, label='Eliminação Gaussiana (1 matriz e 10 vetores)', marker='o')
axis[1].plot(ptlu_b, t_ptlu_b, label='Decomposição PtLU (1 matriz e 10 vetores)', marker='o')
axis[1].set_title('1 matriz e 10 vetores')
axis[1].set_xlabel('n')
axis[1].set_ylabel('Tempo (s)')
axis[1].legend(['Eliminação Gaussiana', 'Decomposição PtLU'])
axis[1].grid(True)

plt.show()
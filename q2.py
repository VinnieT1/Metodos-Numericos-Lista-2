import numpy as np

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

def coef(m, x):
    return [float(x**i) for i in range(m, -1, -1)]

A = np.array([
    coef(6, -1),
    coef(6, 1),
    coef(6, 3),
    coef(6, 4),
    coef(6, 5),
    coef(6, 6),
    coef(6, 7)
])
b = np.array([3.0, 0.0, 2.5, 4.0, -2.0, 8.0, 3.0])

result = eliminacao_gaussiana(A, b)
print(result)
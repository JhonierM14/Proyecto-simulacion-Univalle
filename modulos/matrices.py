import numpy as np

def matrizCentral(matriz):
    """
    Retorna una matriz con los valores centrales de la matriz
    es decir se ignoran los bordes en la nueva matriz
    """

    filas, columnas = matriz.shape
    newMatriz = np.zeros((filas-2, columnas-2), float)
    for i in range(filas-2):
        for j in range(columnas-2):
            newMatriz[i,j] = matriz[i+1,j+1]

    return newMatriz

def norma_infinita_jacobi(A):
    n = A.shape[0]
    return max([
        sum([abs(A[i, j] / A[i, i]) for j in range(n) if j != i])
        for i in range(n)
    ])

def es_diagonal_dominante(A):
    n = A.shape[0]
    for i in range(n):
        suma_fuera_diagonal = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= suma_fuera_diagonal:
            print(f"Fila {i}: |{A[i, i]:.3f}| <= suma {suma_fuera_diagonal:.3f} ❌ No dominante")
            return False
        else:
            print(f"Fila {i}: |{A[i, i]:.3f}| > suma {suma_fuera_diagonal:.3f} ✅")
    return True

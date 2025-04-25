import numpy as np

def matrizCentral(matriz):
    """
    Retorna una matriz con los valores centrales de otra matriz
    dada, es decir se ignoran los bordes en la nueva matriz

    Args
    matriz
    """

    filas, columnas = matriz.shape
    newMatriz = np.zeros((filas-2, columnas-2), float)
    for i in range(filas-2):
        for j in range(columnas-2):
            newMatriz[i,j] = matriz[i+1,j+1]

    return newMatriz

def calcular_norma_infinito(A):
    """
    Calcula la norma infinito de una matriz

    Args
    matriz
    """
    n = A.shape[0]
    norma_infinito = 0
    for i in range(n):
        suma = sum(abs(A[i, j] / A[i, i]) for j in range(n) if j != i)
        norma_infinito = max(norma_infinito, suma)
    return norma_infinito

def es_diagonal_dominante(A):
    """
    Verifica si una matriz A es diagonalmente dominante

    Args
    
    matriz
    """
    n = A.shape[0]
    for i in range(n):
        suma_fuera_diagonal = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= suma_fuera_diagonal:
            print(f"Fila {i}: |{A[i, i]:.3f}| <= suma {suma_fuera_diagonal:.3f} ❌ No dominante")
            return False
        else:
            print(f"Fila {i}: |{A[i, i]:.3f}| > suma {suma_fuera_diagonal:.3f} ✅")
    return True

def verificar_criterio_convergencia(A, Q):
    """
    Verifica si una matriz cumple el criterio de convergencia

    Args
    
    matriz A -> Jacobiano
    matriz Q -> Matriz calculo inversa
    """
    Q_inv = np.linalg.inv(Q)
    T = np.eye(A.shape[0]) - Q_inv @ A
    norma = np.linalg.norm(T, ord=np.inf)
    print(f"‖I - Q⁻¹A‖_inf = {norma}")
    if norma <= 1:
        print("✅ Cumple el criterio de convergencia.")
    else:
        print("❌ No cumple el criterio de convergencia.")

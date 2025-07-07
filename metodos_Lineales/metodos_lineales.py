import numpy as np

def Richardson(a: np.ndarray, b: np.ndarray, x0: np.ndarray, Q: np.ndarray, iteraciones: int, umbral: float) -> np.ndarray:
    n = b.shape[0]

    if x0 is None:
        x = np.zeros((n,1))
    else:
        x = x0.reshape((n,1))
        
    b = b.reshape(n,1)
    for k in range (iteraciones):
        n = b.shape[0]
        r = np.zeros((n,1))
        for i in range(n):
            v = 0
            for j in range(n):
                v = v + a[i,j]*x[j,0]
            r[i,0] = b[i,0] + v
        for i in range(n):
            x[i,0] = x[i,0] + r[i,0]
        if np.linalg.norm(r, ord=np.inf) < umbral:
            # print(f"Convergencia alcanzada en iteración {k} con umbral {umbral}")
            return x

    # print("No se alcanzó la convergencia dentro del número máximo de iteraciones.")
    return x

def metodo_jacobi(a, b, x0, M: int, umbral: float):
    n = len(b)
    
    if x0 is None:
        x = np.zeros((n,1))
    else:
        x = x0.reshape((n,1))

    u = np.zeros_like(x)

    for k in range(M):
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += a[i][j] * x[j]
            u[i] = (b[i] - suma) / a[i][i]
    # Verificar convergencia usando norma infinito
        diff = np.abs(u - x)
        norma_inf = np.max(diff)
        if norma_inf < umbral:
            print(f"Convergencia alcanzada en la iteración {k+1} con norma infinito {norma_inf:.2e}")
            return u
        
        x[:] = u  # actualizar x para la siguiente iteración
        print(f"Iteración {k+1}")

    print("No se alcanzó la convergencia en el número de iteraciones dado.")
    
    return x

def gauss_seidel(a, b, x0, M):
    """
    a = J,
    b = F,
    x0: valores iniciales,
    M = iteraciones.
    """
    n = len(b)

    if x0 is None:
        x = np.zeros((n,1))
    else:
        x = x0.reshape((n,1))

    u = x

    for k in range(M):
        print(f"iteracion {k+1}")
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += a[i][j] * u[j]
            u[i] = (b[i] - suma) / a[i][i]
        x = u
    return x

def gradiente_descendente(A, b, x0, iteraciones: int, umbral: float):
    """
    Resuelve Ax = b usando el método del gradiente descendente.

    Args:
        A: Matriz cuadrada del sistema,
        b: Vector columna (n x 1),
        x0: valores iniciales,
        iteraciones: Número máximo de iteraciones,
        umbral: umbral de convergencia.

    Returns:
        np.ndarray: Solución x en forma de vector columna (n x 1).
    """

    n = b.shape[0]

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.reshape(n)

    b = b.reshape((n, 1))  # Asegura que b es columna

    for k in range(iteraciones):
        r = b - A @ x
        gradiente = -r
        numerador = np.dot(r.T, r)
        denominador = np.dot(gradiente.T, A @ gradiente)

        if denominador == 0:
            print("División por cero en el cálculo del paso alpha. Método detenido.")
            break

        alpha = numerador / denominador
        x = x + alpha * gradiente

        if np.linalg.norm(r, ord=np.inf) < umbral:
            print(f"✔ Convergencia en iteración {k} con umbral {umbral}")
            return x

    print("No se alcanzó convergencia con gradiente descendente.")
    return x

def gradiente_conjugado(A, b, x0, iteraciones: int, umbral: float):
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.reshape(n)

    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    for i in range(iteraciones):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)

        if np.linalg.norm(r, ord=1) < umbral:
            print(f"Convergió en {i+1} iteraciones.")
            break


        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x
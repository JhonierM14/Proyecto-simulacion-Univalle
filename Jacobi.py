import numpy as np
from modulos.valoresIniciales import *
from modulos.grafico import *
from modulos.matrices import *
from Tensores import *

# tamaño de la malla
filas = 8
columnas = 50

# condiciones de frontera
velocidadInicial = 1
velocidadInicialY = 0.1
cantidadDecimales = 8

# Richardson
iteraciones = 500
presicion = pow(10, -6)

MatrizVelocidadEjeX = np.zeros((filas, columnas), float)
MatrizVelocidadEjeY = np.zeros((filas, columnas), float)

# configuracion de numpy que me permite ver la matriz jacobiana sin salto de linea
np.set_printoptions(suppress=True, precision=cantidadDecimales, threshold=np.inf, linewidth=np.inf)  # se intenta evitar saltos de línea

# valores de frontera
def ValsFrontMatrizVelX(MatrizVelocidadEjeX):
    for i in range(filas-1):
        if(i!=0 and i != (filas - 1)):
                MatrizVelocidadEjeX[i, 0] = velocidadInicial

ValsFrontMatrizVelX(MatrizVelocidadEjeX)
ValsFrontMatrizVelX(MatrizVelocidadEjeY)

# funcion del modulo valoresIniciales
# se llena el centro de la matriz
ValIniCentroMatrizVelX(filas, columnas, MatrizVelocidadEjeX, velocidadInicial, cantidadDecimales)
distParabolica(filas, columnas,MatrizVelocidadEjeX, cantidadDecimales)

ValIniMatrizVelY(filas, columnas, MatrizVelocidadEjeY, velocidadInicialY, cantidadDecimales)

###########################################################################################
#                                       SOLUCION
###########################################################################################

# Ax = b, siendo a = A (el jacobiano), b=b (El vector F), Q la diagonal de A, y M la cantidad de iteraciones.

F = llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales)
J = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales)

def metodo_jacobi(a, b, x0, M, tol):
    n = len(b)
    x = x0.copy()
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
        if norma_inf < tol:
            print(f"Convergencia alcanzada en la iteración {k+1} con norma infinito {norma_inf:.2e}")
            return u
        
        x[:] = u  # actualizar x para la siguiente iteración
        print(f"Iteración {k+1}: {x}")

    print("No se alcanzó la convergencia en el número de iteraciones dado.")
    
    return x

Q = copiarDiagonal(J)

H = matrizCentral(MatrizVelocidadEjeX)
H = H.ravel()
resul = metodo_jacobi(J, F, H, iteraciones, presicion)

correcion = resul.reshape(filas-2, columnas-2)

for i in range(filas-2):
    for j in range(columnas-2):
        MatrizVelocidadEjeX[i+1, j+1] -= correcion[i,j]

grafico_matriz_velocidad(MatrizVelocidadEjeX)

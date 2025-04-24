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
iteraciones = 1
presicion = pow(10, -6)

MatrizVelocidadEjeX = np.zeros((filas, columnas), float)
MatrizVelocidadEjeY = np.zeros((filas, columnas), float)

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

# se resuelve iterativamente la ecuacion H = -J^(-1)xF que es equivalente a
# Xn+1 = Xn - J^(-1)xF

# se tienen las matrices F (288) Y J(288x288) y vector de incognitas H(288), para dar solucion
# con el metodo de richardson se hace uso de una matriz Q que sera la matriz identidad, pasando de
# x^k = (I-Q^(-1)*A)*x^(k-1) + Q^(-1)*b a quedar con la forma x^k = (I-A)*x^(k-1) + b
# obteniendo un metodo iterativo para dar solucion al sistema no lineal.

# Ax = b, siendo a = A (el jacobiano), b=b (El vector F), Q la matriz identidad, y M la cantidad de iteraciones.

F = llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales)
J = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales)

# def GaussSeidel(a, b, M):
#     n = b.shape[0]
#     x = np.zeros((n,1))
#     b = b.reshape(n, 1)
#     for k in range(M):
#         for i in range(n):
#             suma = 0
#             for j in range(n):
#                 if j != i:
#                     suma += a[i,j] * x[j,0]
#             x[i,0] = (b[i,0] - suma) / a[i,i]
#     return x

def gauss_seidel(a, b, x0, M):
    n = len(b)
    x = x0.copy()

    for k in range(M):
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += a[i][j] * x[j]
            x[i] = (b[i] - suma) / a[i][i]
    
    return x

H = matrizCentral(MatrizVelocidadEjeX)
H = H.ravel()

resul = gauss_seidel(J, F, H, 100)
print("Resultado: ", resul)

correcion = resul.reshape(filas-2, columnas-2)

for i in range(filas-2):
    for j in range(columnas-2):
        MatrizVelocidadEjeX[i+1, j+1] -= correcion[i,j]

# Calcular el residuo
residuo = np.linalg.norm(F - np.dot(J, resul))
print("Norma del residuo:", residuo)

cond_num = np.linalg.cond(J)
print("Número de condición de J:", cond_num)

det_J = np.linalg.det(J)
print("Determinante de J:", det_J)

grafico_matriz_velocidad(MatrizVelocidadEjeX)

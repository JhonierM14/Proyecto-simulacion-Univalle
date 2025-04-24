import numpy as np
from modulos.valoresIniciales import *
from modulos.grafico import *
from Tensores import *

# tamaño de la malla
filas = 8
columnas = 50

# condiciones de frontera
velocidadInicialX = 1
velocidadInicialY = 0.1
cantidadDecimales = 8

# Newton
iteraciones = 100
presicion = pow(10, -5)

MatrizVelocidadEjeX = np.zeros((filas, columnas), float)
MatrizVelocidadEjeY = np.zeros((filas, columnas), float)

# configuracion de numpy que me permite ver la matrices sin salto de linea
np.set_printoptions(suppress=True, precision=cantidadDecimales, threshold=np.inf, linewidth=np.inf)

# valores de frontera
def ValsFrontMatrizVel(MatrizVelocidadEjeX):
    for i in range(filas-1):
        if(i!=0 and i != (filas - 1)):
                MatrizVelocidadEjeX[i, 0] = velocidadInicialX

ValsFrontMatrizVel(MatrizVelocidadEjeX)
ValsFrontMatrizVel(MatrizVelocidadEjeY)

# funcion del modulo valoresIniciales
ValIniCentroMatrizVelX(filas, columnas, MatrizVelocidadEjeX, velocidadInicialX, cantidadDecimales)

# se llena el resto de la matriz con valores iniciales
distParabolica(filas, columnas, MatrizVelocidadEjeX, cantidadDecimales)
ValIniMatrizVelY(filas, columnas, MatrizVelocidadEjeY, velocidadInicialY, cantidadDecimales)

    ###########################################################################################
    #                                       SOLUCION
    ###########################################################################################

contador = 0
while contador<=iteraciones:
    F = llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales)
    J = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales)
    if np.linalg.det(J) == 0:
        raise ValueError("La matriz Jacobiana es singular. Newton no puede continuar.")

    # se resuelve iterativamente la ecuacion H = -J^(-1)xF que es equivalente a
    # Xn+1 = Xn - J^(-1)xF

    #  J * H = -F
    H = np.linalg.solve(J, -F)
    H = np.round(H, cantidadDecimales)
    H = np.reshape(H, (filas-2, columnas-2))

    # abs(MatrizVelocidad[1,1] - (MatrizVelocidad[1,1] + H[1, 1])) < umbral o
    # Se calcula la diferencia máxima entre la matriz anterior y la nueva
    # norma infinito de un vector
    delta = np.max(np.abs(H))

    # se llena la matriz velocidad con los nuevos valores calculados
    MatrizVelocidadEjeX[1:-1, 1:-1] += np.round(H, cantidadDecimales)

    # se verifica convergencia
    if delta < presicion or contador >= iteraciones:
        print(f"Convergencia alcanzada en iteración {contador} con delta = {delta:.6f}")
        # valores solucion al sistema de ecuaciones
        print(np.array2string(MatrizVelocidadEjeX, separator=', ', threshold=np.inf))

        grafico_matriz_velocidad(MatrizVelocidadEjeX)
        break
    else:
        print(f"Iteración {contador}, delta = {delta:.6f}")
    contador+=1

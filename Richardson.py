import numpy as np
from modulos.valoresIniciales import *
from modulos.grafico import *
from modulos.matrices import *
from Tensores import *

# tamaño de la malla
filas = 8
columnas = 50

# condiciones de frontera
velocidadInicialX = 1
velocidadInicialY = 0.1
cantidadDecimales = 8

# Richardson
iteraciones = 500
presicion = pow(10, -5)

MatrizVelocidadEjeX = np.zeros((filas, columnas), float)
MatrizVelocidadEjeY = np.zeros((filas, columnas), float)

# configuracion de numpy que me permite ver la matrices sin salto de linea
np.set_printoptions(suppress=True, precision=cantidadDecimales, threshold=np.inf, linewidth=np.inf)

# valores de frontera
def ValsFrontMatrizVelX(MatrizVelocidadEjeX):
    for i in range(filas-1):
        if(i!=0 and i != (filas - 1)):
                MatrizVelocidadEjeX[i, 0] = velocidadInicialX

ValsFrontMatrizVelX(MatrizVelocidadEjeX)
ValsFrontMatrizVelX(MatrizVelocidadEjeY)

# funcion del modulo valoresIniciales se llena el centro de la matriz
ValIniCentroMatrizVelX(filas, columnas, MatrizVelocidadEjeX, velocidadInicialX, cantidadDecimales)

# se llena el resto de la matriz con valores iniciales
distParabolica(filas, columnas,MatrizVelocidadEjeX, cantidadDecimales)
ValIniMatrizVelY(filas, columnas, MatrizVelocidadEjeY, velocidadInicialY, cantidadDecimales)

###########################################################################################
#                                       SOLUCION                                          #
###########################################################################################

# Ax = b, siendo a = J (el jacobiano), b=F (El vector F), Q la matriz identidad, y iteraciones la cantidad de iteraciones.

def Richarson(a, b, Q, M, tol):
    n = b.shape[0]
    x = np.zeros((n,1))
    b = b.reshape(n,1)
    for k in range (M):
        n = b.shape[0]
        r = np.zeros((n,1))
        for i in range(n):
            v = 0
            for j in range(n):
                v = v + a[i,j]*x[j,0]
            r[i,0] = b[i,0] + v
        for i in range(n):
            x[i,0] = x[i,0] + r[i,0]
        if np.linalg.norm(r, ord=np.inf) < tol:
            print(f"Convergencia alcanzada en iteración {k} con tolerancia {tol}")
            return x

    print("No se alcanzó la convergencia dentro del número máximo de iteraciones.")
    return x

F = llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales)
J = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales)

print(verificar_criterio_convergencia(J, np.eye(J.shape[0])))

Q = np.eye(F.shape[0])
resul = Richarson(J, F, Q, iteraciones, presicion)

# se llena la matriz velocidad con los nuevos valores calculados
MatrizVelocidadEjeX[1:-1, 1:-1] += np.round(resul.reshape(filas-2, columnas-2), cantidadDecimales)

grafico_matriz_velocidad(MatrizVelocidadEjeX)

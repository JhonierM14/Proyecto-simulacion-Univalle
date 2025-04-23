import numpy as np
from modulos.valoresIniciales import *
from modulos.grafico import *
from modulos.matrices import *

# tamaño de la malla
filas = 8
columnas = 50

# condiciones de frontera
velocidadInicial = 1
velocidadInicialY = 0.1
cantidadDecimales = 8

# Richardson
iteraciones = 100
presicion = pow(10, -5)

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
#                                       ECUACIONES
###########################################################################################

# dada la ecuacion de Navier-stokes, al ser discretizada se obtiene un total 
# de nueve ecuaciones, de acuerdo a la posicion de la casilla velocidad del rectangulo 

# el metodo de Richardson resuelve problemas matriciales de la forma Ax= b
# por lo tanto puede ser usado para resolver el sistema J x H = F con A=J, H=x y F = b 

def ecuacionDiscretizada(Vx0, Vy0, CSup, CIzq, CInf, CDer, vij):
    # f es una representacion de una valor, que en teoria deberia ser 0
    coeficiente1 = (CInf-CSup)/2 
    coeficiente2 = (CDer-CIzq)/2
    sumaCasillasAdyacentes = CSup + CIzq + CDer + CInf
    f = 1/4*(-Vx0*coeficiente1 - Vy0*coeficiente2 + sumaCasillasAdyacentes) - vij
    return round(f, cantidadDecimales)

def llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY):
    F = np.zeros((filas - 2 , columnas - 2), float)
    for i in range(filas-2):
        for j in range(columnas-2):
            # vx en este caso es vij, debido a que la velocidad solo 
            # depende de la componente en x
            vx = MatrizVelocidadEjeX[i+1, j+1]
            vSup = MatrizVelocidadEjeX[i, j+1]
            vIzq = MatrizVelocidadEjeX[i+1, j] 
            vy = MatrizVelocidadEjeY[i+1, j+1]
            vInf = MatrizVelocidadEjeX[i+2, j+1] 
            vDer = MatrizVelocidadEjeX[i+1, j+2]
            vij = MatrizVelocidadEjeX[i+1, j+1]

            F[i,j] = round(ecuacionDiscretizada(vx, vy, vSup, vIzq, vInf, vDer, vij), cantidadDecimales)

    vector = F.ravel()
    return vector

def MatrizJacobiana(M_V_x, M_V_y):
    filas_internas = filas - 2
    columnas_internas = columnas - 2
    tam = filas_internas * columnas_internas
    J = np.zeros((tam, tam), float)

    def pos(i, j):
        return i * columnas_internas + j

    for i in range(filas_internas):
        for j in range(columnas_internas):
            fila_eq = pos(i, j)
            vx = M_V_x[i+1, j+1]
            vy = M_V_y[i+1, j+1]

            # Derivadas parciales
            # dv/d(v_ij)
            # J[fila_eq, fila_eq] =  1

            # dv/d(v_ij)
            #J[fila_eq, fila_eq] = 1/4*((M_V_x[i+2, j+1]- M_V_x[i, j+1])/2) - 1
            J[fila_eq, fila_eq] = - 1

            # dv/d(v_{i-1,j})
            if i > 0:
                J[fila_eq, pos(i-1, j)] = 0.25 * (vx / 2 + 1)

            # dv/d(v_{i+1,j})
            if i < filas_internas - 1:
                J[fila_eq, pos(i+1, j)] = 0.25 * (-vx / 2 + 1)

            # dv/d(v_{i,j-1})
            if j > 0:
                J[fila_eq, pos(i, j-1)] = 0.25 * (vy / 2 + 1)

            # dv/d(v_{i,j+1})
            if j < columnas_internas - 1:
                J[fila_eq, pos(i, j+1)] = 0.25 * (-vy / 2 + 1)

    return J

contador = 0
while contador<iteraciones:
    H = np.zeros((288,1), float)
    H = H.ravel()
    b = llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY)
    b = b.ravel()
    A = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY)
    r = b - (A @ H)
    alpha = np.dot(r, r) / np.dot(r, np.dot(A, r))
    gradiente = A @ H - b
    newX = H - alpha* gradiente
    newX = newX.reshape(filas-2, columnas-2)

    for i in range(filas-2):
        for j in range(columnas-2):
            MatrizVelocidadEjeX[i+1,j+1] = newX[i, j]

    contador+= 1

graficoValoresIniciales(MatrizVelocidadEjeX, "eje x")

import numpy as np
import matplotlib.pyplot as plt
from modulos.valoresIniciales import *
from modulos.grafico import *

# el eje x es el unico que estamos teniendo en cuenta al momento
# por lo tanto solo la matriz de velocidad en el eje x tendra 
# valores entre 1 y 0 en forma decreciente, asegurando
# una buena coherencia con el sentido fisico del fluido
# haciendo uso de las 9 ecuaciones que hemos definido en
# en la primera entrega.

# tamaño de la malla
filas = 8
columnas = 50

# condiciones de frontera
velocidadInicialX = 1
velocidadInicialY = 0.1
cantidadDecimales = 8

# Newton
iteraciones = 5
presicion = pow(10, -6)

MatrizVelocidadEjeX = np.zeros((filas, columnas), float)
MatrizVelocidadEjeY = np.zeros((filas, columnas), float)

# configuracion de numpy que me permite ver la matriz jacobiana sin salto de linea
np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # se intenta evitar saltos de línea

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
#                                       ECUACIONES
###########################################################################################

# dada la ecuacion de Navier-stokes, al ser discretizada se obtiene un total 
# de nueve ecuaciones, de acuerdo a la posicion de la casilla velocidad del rectangulo 

# debemos declarar una matriz F que contenga la funciones evaluadas en 
# los valores inciales, y una matriz jacobiana con las derivadas evaluadas en los
# valores iniciales

# matriz F
# se resta dos debido a que se evalua el interior de la matriz velocidad

def ecuacionDiscretizada(Vx0, Vy0, CSup, CIzq, CInf, CDer, vij):
    # f es una representacion de una valor, que en teoria deberia ser 0
    coeficiente1 = (CInf-CSup)/2 
    coeficiente2 = (CDer-CIzq)/2
    sumaCasillasAdyacentes = CSup + CIzq + CDer + CInf
    f = 1/4*(-Vx0*coeficiente1 - Vy0*coeficiente2 + sumaCasillasAdyacentes) - vij
    return round(f, cantidadDecimales)

contador = 0
while contador<=iteraciones:
    F = np.zeros((filas - 2 , columnas - 2), float)
    def llenarMatrizF(F):
        for i in range(filas-2):
            for j in range(columnas-2):
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

    F = llenarMatrizF(F)

    # matriz jacobiana
    # todas las variables van a estar representadas en una sola fila, pero solo 
    # 5 valores valores van a tener coeficiente distinto de 0, que es la cantidad
    # de variables independientes en la ecuacion que representa cada casilla

    # matriz jacobiana de F
    tamaño = (filas-2)*(columnas-2)
    J = np.zeros((tamaño, tamaño), float)

    # las filas del jacobiano representan la cantidad de ecuaciones, es decir la cantidad
    # de filas que debe tener el jacobiano es igual a 3x9

    # las columnas del jacobiano representan las derivadas respecto a las variables independientes, es decir
    # el numero de columas debe ser igual al numero de filas

    # configuracion de numpy que me permite ver la matriz jacobiana sin salto de linea
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # se intenta evitar saltos de línea

    # la formula del metodo de Newton es J^(-1)xH = -F al pasar a multiplicar por el inverso de J^(-1)
    # nos queda H = -J^(-1)xF, por lo tanto se tendra en cuenta el signo negativo de una vez
    # al momento de llenar la matriz jacobiana

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
                # J[fila_eq, fila_eq] = 1/4*((M_V_x[i+2, j+1]- M_V_x[i, j+1]))/2 - 1
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

    J = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY)

    ###########################################################################################
    #                                       SOLUCION
    ###########################################################################################

    # se resuelve iterativamente la ecuacion H = -J^(-1)xF que es equivalente a
    # Xn+1 = Xn - J^(-1)xF

    # se tienen las matrices F Y J
    def valoresSolucion(J, F, cantidad_decimales):
        delta_x = np.linalg.solve(J, -F)
        delta_x = np.round(delta_x, cantidad_decimales)  # Ajuste de decimales
        return delta_x

    H = valoresSolucion(J, F, cantidadDecimales)

    ###########################################################################################
    #                                  Valores de la velocidad
    ###########################################################################################

    # se va a restar los valores que habiamos declarado a los valores iniciales para 
    # encontrar los nuevos valores de las velocidades en la matriz de velocidades 
    # del eje x y se plantear un algoritmo iterativo para entradas de nxm

    # funcion que convierte un vector a matriz
    H = np.reshape(H, (filas-2, columnas-2))

    # abs(MatrizVelocidad[1,1] - (MatrizVelocidad[1,1] + H[1, 1])) < umbral o
    # Se calcula la diferencia máxima entre la matriz anterior y la nueva
    # norma infinito de un vector
    delta = np.max(np.abs(H))

    # se llena la matriz velocidad con los nuevos valores calculados
    for i in range(filas-2):
        for j in range(columnas-2):
            MatrizVelocidadEjeX[i+1, j+1] = round(MatrizVelocidadEjeX[i+1, j+1] + H[i, j], cantidadDecimales)

    # se verifica si ya se alcanzó el umbral de convergencia o el máximo de iteraciones
    if delta < presicion or contador >= iteraciones:
        print(f"Convergencia alcanzada en iteración {contador} con delta = {delta:.6f}")
        
        grafico_matriz_velocidad(MatrizVelocidadEjeX)

        break
    else:
        print(f"Iteración {contador}, delta = {delta:.6f}")
    contador+=1

import numpy as np

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

def ecuacionDiscretizada(Vx0, Vy0, CSup, CIzq, CInf, CDer, vij, cantidadDecimales):
    # f es una representacion de una valor, que en teoria deberia ser 0
    coeficiente1 = (CInf-CSup)/2 
    coeficiente2 = (CDer-CIzq)/2
    sumaCasillasAdyacentes = CSup + CIzq + CDer + CInf
    f = 1/4*(-Vx0*coeficiente1 - Vy0*coeficiente2 + sumaCasillasAdyacentes) - vij
    return round(f, cantidadDecimales)

def llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY, cantidadDecimales):
    filas, columnas = MatrizVelocidadEjeX.shape
    F = np.zeros((filas - 2 , columnas - 2), float)
    for i in range(filas-2):
        for j in range(columnas-2):
            vx = MatrizVelocidadEjeX[i+1, j+1]
            vSup = MatrizVelocidadEjeX[i, j+1]
            vIzq = MatrizVelocidadEjeX[i+1, j] 
            vy = MatrizVelocidadEjeY[i+1, j+1]
            vInf = MatrizVelocidadEjeX[i+2, j+1] 
            vDer = MatrizVelocidadEjeX[i+1, j+2]
            vij = MatrizVelocidadEjeX[i+1, j+1]

            F[i,j] = ecuacionDiscretizada(vx, vy, vSup, vIzq, vInf, vDer, vij, cantidadDecimales) 
    
    vector = F.ravel()
    return vector


# matriz jacobiana
# todas las variables van a estar representadas en una sola fila, pero solo 
# 5 valores valores van a tener coeficiente distinto de 0, que es la cantidad
# de variables independientes en la ecuacion que representa cada casilla

# matriz jacobiana de F

def MatrizJacobiana(M_V_x, M_V_y, presicion):
    filas, columnas = M_V_x.shape
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
            J[fila_eq, fila_eq] = -1

            # dv/d(v_{i-1,j})
            if i > 0:
                J[fila_eq, pos(i-1, j)] = round(0.25 * (vx / 2 + 1), presicion)

            # dv/d(v_{i+1,j})
            if i < filas_internas - 1:
                J[fila_eq, pos(i+1, j)] = round(0.25 * (-vx / 2 + 1), presicion)

            # dv/d(v_{i,j-1})
            if j > 0:
                J[fila_eq, pos(i, j-1)] = round(0.25 * (vy / 2 + 1), presicion)

            # dv/d(v_{i,j+1})
            if j < columnas_internas - 1:
                J[fila_eq, pos(i, j+1)] = round(0.25 * (-vy / 2 + 1), presicion)

    return J

# las filas del jacobiano representan la cantidad de ecuaciones, es decir la cantidad
# de filas que debe tener el jacobiano es igual a 6x48

# las columnas del jacobiano representan las derivadas respecto a las variables independientes, es decir
# el numero de columas debe ser igual al numero de filas

# la formula del metodo de Newton es J^(-1)xH = -F al pasar a multiplicar por el inverso de J^(-1)
# nos queda H = -J^(-1)xF, por lo tanto se tendra en cuenta el signo negativo de una vez
# al momento de llenar la matriz jacobiana

import numpy as np
from modulos.valoresIniciales import *
from modulos.grafico import *
from modulos.matrices import *

# tamaño de la malla
filas = 8
columnas = 50

# condiciones de frontera
velocidadInicialX = 1
velocidadInicialY = 0.1
cantidadDecimales = 8

# Richardson
iteraciones = 10
presicion = pow(10, -5)

MatrizVelocidadEjeX = np.zeros((filas, columnas), float)
MatrizVelocidadEjeY = np.zeros((filas, columnas), float)

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

# matriz jacobiana
# todas las variables van a estar representadas en una sola fila, pero solo 
# 5 valores valores van a tener coeficiente distinto de 0, que es la cantidad
# de variables independientes en la ecuacion que representa cada casilla

# matriz jacobiana de F

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

###########################################################################################
#                                       SOLUCION
###########################################################################################

# se resuelve iterativamente la ecuacion H = -J^(-1)xF que es equivalente a
# Xn+1 = Xn - J^(-1)xF

# se tienen las matrices F (288) Y J(288x288) y vector de incognitas H(288), para dar solucion
# con el metodo de richardson se hace uso de una matriz Q equivalente a la identidad, pasando de
# x^k = (I-Q^(-1)*A)*x^(k-1) + Q^(-1)*b a quedar con la forma x^k = (I-A)*x^(k-1) + b
# obteniendo un metodo iterativo para intentar dar solucion al sistema no lineal.

# Ax = b, siendo a = A (el jacobiano), b=b (El vector F), Q la matriz identidad, y iteraciones la cantidad de iteraciones.

def Richarson(a, b, Q, M):
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
    return x

F = llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY)
J = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY)
resul = Richarson(J, F, np.eye(F.shape[0]), 100)
print(resul)

correcion = resul.reshape(filas-2, columnas-2)

for i in range(filas-2):
    for j in range(columnas-2):
        MatrizVelocidadEjeX[i+1, j+1] += correcion[i,j]

residuo = J @ resul + F.reshape(-1,1)
print("Norma del residuo:", np.linalg.norm(residuo))

print(MatrizVelocidadEjeX)
grafico_matriz_velocidad(MatrizVelocidadEjeX)

# tamaño = (filas-2)*(columnas-2)
# contador = 0 
# while contador<=iteraciones: 
#     F = llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY)
#     J = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY)

#     # Verificar normas
#     norm_1 = np.linalg.norm(J, 1)
#     norm_2 = np.linalg.norm(J, 2)  # Cuidado: costoso
#     norm_inf = np.linalg.norm(J, np.inf)

#     print(f"Norma 1 de la jacobiana: {norm_1:.4f}")
#     print(f"Norma 2 de la jacobiana: {norm_2:.4f}")
#     print(f"Norma infinito de la jacobiana: {norm_inf:.4f}")

#     if norm_1 < 1 and norm_2 < 1 and norm_inf < 1:
#         print("Posible convergencia con método de Richardson")
#     else:
#         print("Advertencia: posible divergencia por norma")

#     newH = np.zeros(F.shape[0], float)      # vector para los nuevos valores de H
#     I = np.eye(tamaño)                      # matriz identidad 
#     coeficiente = I - J                     # I-A 

#     H = matrizCentral(MatrizVelocidadEjeX)  # se toman los valores inciales de la matriz de velocidad x

#     # vector de soluciones
#     newH = np.round(coeficiente @ H.ravel() + F, cantidadDecimales)
#     newH = newH.reshape(filas-2 ,columnas-2)

#     # se ingresa en la matriz velocidad los nuevos valores calculados
#     for i in range(filas-2):
#         for j in range(columnas-2):
#             MatrizVelocidadEjeX[i+1, j+1] = newH[i,j]

#     delta = np.max(np.abs(H - newH))


#     if(delta<presicion or contador>=iteraciones):
#         print(f"Convergencia alcanzada en iteración {contador} con delta = {delta:.6f}")
#         graficoValoresIniciales(MatrizVelocidadEjeX, "eje x")
#         # valores solucion al sistema de ecuaciones
#         print(np.array2string(MatrizVelocidadEjeX, separator=', ', threshold=np.inf))
#         break
#     else:
#         print(f"Iteración {contador}, delta = {delta:.6f}")
#     contador +=1

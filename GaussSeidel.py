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
iteraciones = 1
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

F = llenarMatrizF(MatrizVelocidadEjeX, MatrizVelocidadEjeY)
J = MatrizJacobiana(MatrizVelocidadEjeX, MatrizVelocidadEjeY)
print(np.array2string(J, separator=', ', threshold=np.inf))


es_diagonal_dominante(J)

def GaussSeidel(a, b, M):
    n = b.shape[0]
    x = np.zeros((n,1))
    b = b.reshape(n, 1)
    for k in range(M):
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += a[i,j] * x[j,0]
            x[i,0] = (b[i,0] - suma) / a[i,i]
    return x

a = np.array([[7.0, 1.0, -2.0],
              [-3.0, -5.0, 1.0],
              [2.0, 2.0, -6.0]])
b = np.array([5.0, -20.0, -8.0])
prueba = GaussSeidel(a, b, 5)
print("Prueba: ", prueba)
print("aquie termina Prueba: ")


resul = GaussSeidel(J, F, 1)
print("Resultado: ", resul)

correcion = resul.reshape(filas-2, columnas-2)

for i in range(filas-2):
    for j in range(columnas-2):
        MatrizVelocidadEjeX[i+1, j+1] += correcion[i,j]

# Calcular el residuo
residuo = np.linalg.norm(F - np.dot(J, resul))
print("Norma del residuo:", residuo)

cond_num = np.linalg.cond(J)
print("Número de condición de J:", cond_num)

det_J = np.linalg.det(J)
print("Determinante de J:", det_J)

print(MatrizVelocidadEjeX)
#grafico_matriz_velocidad(MatrizVelocidadEjeX)

#contador = 0 
#while contador<=iteraciones: 

# matriz jacobiana
# todas las variables van a estar representadas en una sola fila, pero solo 
# 5 valores valores van a tener coeficiente distinto de 0, que es la cantidad
# de variables independientes en la ecuacion que representa cada casilla

# matriz jacobiana de F
# tamaño = (filas-2)*(columnas - 2)

# # las filas del jacobiano representan la cantidad de ecuaciones, es decir la cantidad
# # de filas que debe tener el jacobiano es igual a (filas-2)x(columnas-2)

# # las columnas del jacobiano representan las derivadas respecto a las variables independientes, es decir
# # el numero de columas debe ser igual al numero de filas

# # configuracion de numpy que me permite ver la matriz jacobiana sin salto de linea
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # se intenta evitar saltos de línea

# # la formula del metodo de Newton es J^(-1)xH = -F al pasar a multiplicar por el inverso de J^(-1)
# # nos queda H = -J^(-1)xF, por lo tanto se tendra en cuenta el signo negativo de una vez
# # al momento de llenar la matriz jacobiana

# # Mismo tamaño del jacobiano
# Q = copiarDiagonal(J)

# # vector con los valores calculados, nuevo H 
# newH = np.zeros(F.shape[0], float)

# # a = J, H = x and b = F 
# inversaQ = np.linalg.inv(Q) 

# # I-A 
# coeficiente = np.round(matrizIdentidad(tamaño, tamaño, 1) - (inversaQ @ J), cantidadDecimales )

# H = matrizCentral(MatrizVelocidadEjeX) 

# # vector de soluciones
# newH = np.round(coeficiente @ H.ravel() + (inversaQ @ F), cantidadDecimales)
# # print(" nuevos valores calculados para los valores iniciales")
# print(np.array2string(newH, separator=', ', threshold=np.inf))
# # print("----------------")

# # convertir vector a Matriz
# newH = newH.reshape(filas-2 ,columnas-2)

# #Norma infinita (máxima suma por fila)
# norm_inf = norma_infinita_jacobi(J)

# # if(norm_inf<=1):
# #     print("converge para algun valor x_0")
# #     diferencia = newH - H
# #     norma_diferencia = np.linalg.norm(diferencia, ord=np.inf)
# #     cota_error = (norm_inf / (1 - norm_inf)) * norma_diferencia
# #     print(f"Cota del error verdadero: {cota_error:.6f}")
# # else:
# #     print("divergue, norma infinito mayor a 1")
# #     break

# # # se ingresa en la matriz velocidad los nuevos valores calculados
# # for i in range(filas-2):
# #     for j in range(columnas-2):
# #         MatrizVelocidadEjeX[i+1, j+1] = newH[i,j]

# # delta = np.max(np.abs(H - newH))

# # if(delta<presicion or contador>=iteraciones):
# #     print(f"Convergencia alcanzada en iteración {contador} con delta = {delta:.6f}")
# #     #grafico_matriz_velocidad(MatrizVelocidadEjeX)
# #     graficoValoresIniciales(MatrizVelocidadEjeX, "eje x")
# # else:
# #     print(f"Iteración {contador}, delta = {delta:.6f}")

# # contador +=1

# # #print(np.array2string(MatrizVelocidadEjeX, separator=', ', threshold=np.inf))

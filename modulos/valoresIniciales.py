import numpy as np

# para llenar la matriz con los valores iniciales, tenemos que asignar valores 
# iniciales de acuerdo al comportamiento fisico o sentido fisico del fluido
# al desplazarse en una region bidimensional

def decremento(numeroInicial, cantidadColumnas):
    """
    funcion para el decremento de las n 
    filas distintas al centro, y fila central
    """
    return numeroInicial/(cantidadColumnas - 1)

def centroMatriz(filas):
    """
    retorna el centro de una matriz par o impar
    """
    centro = list()
    if(filas%2==0):
        #hay dos en el centro
        centro.append(int(filas/2))
        centro.append(int(filas/2 + 1))
    else:
        centro.append(int(filas/2) + 1)
    return centro

def ValIniCentroMatrizVelX(filas, columnas, MatrizVelocidadEjeX, velocidadInicial, cantidadDecimales):
    """
    funcion que llena los valores iniciales del 
    centro de una matriz 
    """
    centro = centroMatriz(filas)
    for i in range(filas-1):
        for j in range(columnas-1):
            # si el centro es par
            if(len(centro) == 2):
                if(i!=0 and i != (filas-1) and j != 0 and i == centro[0] - 1 or i == centro[1] - 1):
                    MatrizVelocidadEjeX[i, j] = round(velocidadInicial - decremento(velocidadInicial, columnas)*j, cantidadDecimales)
            # si el centro es una unica fila
            else:
                if(i!=0 and i != (filas-1) and j != 0 and i == centro[0] - 1):
                    MatrizVelocidadEjeX[i, j] = round(velocidadInicial - decremento(velocidadInicial, columnas)*j, cantidadDecimales)

def ValIniMatrizVelY(filas, columnas, MatrizVelocidadEjeY, velocidadInicial, cantidadDecimales):
    for j in range(columnas):
        for i in range(filas):
            if(i != 0 and i != (filas-1) and j != 0 and j != (columnas-1)):
                MatrizVelocidadEjeY[i, j] = 0.1

def decrementoEjeY(MatrizVelocidadEjeX, filas):
    """
    decremento para el eje y
    """
    decrementoValoresInicialesY = MatrizVelocidadEjeX[centroMatriz(filas)[0]-1, 1] /(int(filas/2))
    return decrementoValoresInicialesY

def distParabolica(filas, columnas, MatrizVelocidadEjeX, presicion):
    """
    establece los valores iniciales decrecientes en el eje y
    valores iniciales del eje con distribucion parabolica
    """

    def ValIniNueMatrizVelYParabolico(filas, MatrizVelocidadEjeX, presicion):
        """
        # llena la primera columna de los valores iniciales con 
        # valores simetricos y decrecientes respecto al centro
        """
        centro = centroMatriz(filas)
        for i in range(int(filas/2)):
            if(i!=0 and i!= centro[0] - 1):
                MatrizVelocidadEjeX[i, 1] = round(MatrizVelocidadEjeX[centro[0]-1, 1] - decrementoEjeY(MatrizVelocidadEjeX, filas)*(centro[0] - 1 - i), presicion)
        for i in range(int(filas/2), filas-1):
            if(i != (filas - 1) and i!= centro[0] - 1):
                MatrizVelocidadEjeX[i, 1] = MatrizVelocidadEjeX[centro[0] - (i - int(filas/2 - 1)), 1]

    def ValIniLMatrizVelX(filas, columnas, MatrizVelocidadEjeX, presicion):
        """
        funcion que llena las casillas faltantes de la matriz con valores 
        iniciales correspondientes, ya definidos en la funcion anterior
        """
        centro = centroMatriz(filas)
        for i in range(filas-1):
            for j in range(columnas-2):                       
                velocidadInicial = MatrizVelocidadEjeX[i, 1]
                decrementoX = decremento(velocidadInicial, columnas)
                #PARTE SUPERIOR MATRIZ
                if(i!=0 and i != (filas-1) and j != 0 and i < centro[0] - 1):
                    MatrizVelocidadEjeX[i, j+1] = round(velocidadInicial - decrementoX*(j + 1), presicion)
                # PARTE INFERIOR MATRIZ

                # MATRIZ PAR
                elif(len(centro) == 2):
                    if(i != (filas-1) and j != 0 and i > centro[1] - 1):
                        MatrizVelocidadEjeX[i, j+1] = round(velocidadInicial - decrementoX*(j + 1), presicion)
                # UNICA FILA CENTRAL
                else:
                    if(i != (filas-1) and j != 0 and i > centro[0] - 1):
                        MatrizVelocidadEjeX[i, j+1] = round(velocidadInicial - decrementoX*(j + 1), presicion)
    ValIniNueMatrizVelYParabolico(filas, MatrizVelocidadEjeX, presicion)
    ValIniLMatrizVelX(filas, columnas, MatrizVelocidadEjeX, presicion)

def distLineal(filas, columnas, MatrizVelocidadEjeX, velocidadInicial, presicion):
    """
    llena la matriz con los mismos valores iniciales en cada fila
    """
    for i in range(filas-1):
        for j in range(columnas-1):
            if(i!=0 and i!=(filas-1) and j!=0):
                MatrizVelocidadEjeX[i,j] = round(velocidadInicial - decremento(velocidadInicial, columnas)*j, presicion)

def distCompleta(matriz, valor):
    """
    Crea una matriz de tamaño m x n llena con un valor designado, 
    excepto las fronteras que quedan como 0.
    """
    matriz[1:-1, 1:-1] = valor       # Llena el interior con `valor`

def matrizIdentidad(filas, columnas, coeficiente):
    """
    Retorna una matriz identidad
    """
    matriz = np.zeros((filas, columnas), float)
    for i in range(filas):
        for j in range(columnas):
            if(i==j):
                matriz[i,j] = coeficiente
            else:
                matriz[i,j] = 0
    return matriz

def copiarDiagonal(matriz):
    """
    Retorna una matriz con la copia de la diagnonal de una matriz dada
    """
    filas, columnas = matriz.shape
    Q = np.zeros((filas, columnas), float)
    for i in range(filas):
        for j in range(columnas):
            if(i==j):
                Q[i, j] = matriz[i, j]
    return Q

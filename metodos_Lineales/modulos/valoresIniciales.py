import numpy as np

# Para llenar la matriz con los valores iniciales, podemos asignar valores 
# iniciales de acuerdo al comportamiento fisico o sentido fisico del problema
# o si desconocemos su comportamiento, asignar una distribucion lineal o 
# completa

# valores de frontera
def ValsFrontMatrizVel(MatrizVelocidadEjeX, velocidadInicial):
    """
    se llena la primera columna con los 
    valores iniciales correspondientes
    """
    filas = MatrizVelocidadEjeX.shape[0]
    for i in range(filas-1):
        if(i!=0 and i != (filas - 1)):
                MatrizVelocidadEjeX[i, 0] = velocidadInicial

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

def ValIniCentroMatrizVelX(MatrizVelocidadEjeX, velocidadInicial, DECIMALES):
    """
    funcion que llena los valores iniciales del 
    centro de una matriz
    """
    filas, columnas = MatrizVelocidadEjeX.shape
    centro = centroMatriz(filas)
    for i in range(filas-1):
        for j in range(columnas-1):
            # si el centro es par
            if(len(centro) == 2):
                if(i!=0 and i != (filas-1) and j != 0 and i == centro[0] - 1 or i == centro[1] - 1):
                    MatrizVelocidadEjeX[i, j] = round(velocidadInicial - decremento(velocidadInicial, columnas)*j, DECIMALES)
            # si el centro es una unica fila
            else:
                if(i!=0 and i != (filas-1) and j != 0 and i == centro[0] - 1):
                    MatrizVelocidadEjeX[i, j] = round(velocidadInicial - decremento(velocidadInicial, columnas)*j, DECIMALES)

def ValIniMatrizVelY(MatrizVelocidadEjeY, velocidadInicial):
    """
    asigna una vorticidad constante a la matriz y
    """
    filas, columnas = MatrizVelocidadEjeY.shape
    for j in range(columnas):
        for i in range(filas):
            if(i != 0 and i != (filas-1) and j != 0 and j != (columnas-1)):
                MatrizVelocidadEjeY[i, j] = velocidadInicial

def decrementoEjeY(MatrizVelocidadEjeX, filas: int):
    """
    decremento para el eje y
    """
    decrementoValoresInicialesY = MatrizVelocidadEjeX[centroMatriz(filas)[0]-1, 1] /(int(filas/2))
    return decrementoValoresInicialesY

def distParabolica(MatrizVelocidadEjeX, DECIMALES: float):
    """
    Establece los valores iniciales decrecientes en el eje y, luego los
    valores iniciales del resto de la matriz con distribucion parabolica
    """
    filas, columnas = MatrizVelocidadEjeX.shape

    def ValIniNueMatrizVelYParabolico():
        """
        Llena la primera columna de los valores iniciales con 
        valores simetricos y decrecientes respecto al centro
        """
        centro = centroMatriz(filas)
        for i in range(int(filas/2)):
            if(i!=0 and i!= centro[0] - 1):
                MatrizVelocidadEjeX[i, 1] = round(MatrizVelocidadEjeX[centro[0]-1, 1] - decrementoEjeY(MatrizVelocidadEjeX, filas)*(centro[0] - 1 - i), DECIMALES)
        for i in range(int(filas/2), filas-1):
            if(i != (filas - 1) and i!= centro[0] - 1):
                MatrizVelocidadEjeX[i, 1] = MatrizVelocidadEjeX[centro[0] - (i - int(filas/2 - 1)), 1]

    def ValIniLMatrizVelX():
        """
        funcion que llena las casillas faltantes de la matriz con valores 
        iniciales correspondientes, ya definidos en la funcion ValIniNueMatrizVelYParabolico
        """
        centro = centroMatriz(filas)
        for i in range(filas-1):
            for j in range(columnas-2):                       
                velocidadInicial = MatrizVelocidadEjeX[i, 1]
                decrementoX = decremento(velocidadInicial, columnas)
                #PARTE SUPERIOR MATRIZ
                if(i!=0 and i != (filas-1) and j != 0 and i < centro[0] - 1):
                    MatrizVelocidadEjeX[i, j+1] = round(velocidadInicial - decrementoX*(j + 1), DECIMALES)
                
                # PARTE INFERIOR MATRIZ
                # MATRIZ PAR
                elif(len(centro) == 2):
                    if(i != (filas-1) and j != 0 and i > centro[1] - 1):
                        MatrizVelocidadEjeX[i, j+1] = round(velocidadInicial - decrementoX*(j + 1), DECIMALES)
                # UNICA FILA CENTRAL
                else:
                    if(i != (filas-1) and j != 0 and i > centro[0] - 1):
                        MatrizVelocidadEjeX[i, j+1] = round(velocidadInicial - decrementoX*(j + 1), DECIMALES)

    ValIniNueMatrizVelYParabolico()
    ValIniLMatrizVelX()

def distLineal(MatrizVelocidadEjeX, velocidadInicial, DECIMALES: float):
    """
    llena la matriz con los mismos valores iniciales en cada fila
    """
    filas, columnas = MatrizVelocidadEjeX.shape
    for i in range(filas-1):
        for j in range(columnas-1):
            if(i!=0 and i!=(filas-1) and j!=0):
                MatrizVelocidadEjeX[i,j] = round(velocidadInicial - decremento(velocidadInicial, columnas)*j, DECIMALES)

def distCompleta(matriz, valor: float):
    """
    Crea una matriz de tamaño m x n llena con un valor designado, 
    excepto las fronteras que quedan como 0.
    """
    matriz[1:-1, 1:-1] = valor       # Llena el interior con `valor`

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
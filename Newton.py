from metodos_Lineales.metodos_lineales import *
from metodos_Lineales.modulos.valoresIniciales import *
from metodos_Lineales.modulos.matrices import *
from metodos_Lineales.modulos.grafico import *
from metodos_Lineales.Tensores import * # J y F

# Parametros de la simulacion
FILAS = 8; COLUMNAS = 50 
VELOCIDAD_INICIAL_X = 1; VELOCIDAD_INICIAL_Y = 0.1 # Vorticidad
DECIMALES = 10; ITERACIONES_MAX = 100; UMBRAL = pow(10, -5)
OMEGA = 1.2

# Configuracion de impresion de matrices
np.set_printoptions(suppress=True, precision=DECIMALES, threshold=np.inf, linewidth=np.inf)

def inicializar_matrices():
    # Inicializacion de matrices de velocidad
    vel_x = np.zeros((FILAS, COLUMNAS), float)
    vel_y = np.zeros((FILAS, COLUMNAS), float)

    # Asignacion de condiciones de frontera iniciales
    ValsFrontMatrizVel(vel_x, VELOCIDAD_INICIAL_X)
    ValsFrontMatrizVel(vel_y, VELOCIDAD_INICIAL_Y)

    # funcion del modulo valoresIniciales
    ValIniCentroMatrizVelX(vel_x, VELOCIDAD_INICIAL_X, DECIMALES)
    ValIniMatrizVelY(vel_y, VELOCIDAD_INICIAL_Y)

    # se llena el resto de la matriz con valores iniciales
    distParabolica(vel_x, DECIMALES)

    return vel_x, vel_y

def metodo_newton(vel_x, vel_y, metodo):
    for iteracion in range(ITERACIONES_MAX + 1):
        J = MatrizJacobiana(vel_x, vel_y, DECIMALES)
        
        if np.linalg.det(J) == 0:
            raise ValueError("La matriz Jacobiana es singular. Newton no puede continuar.")
        
        F = llenarMatrizF(vel_x, vel_y, DECIMALES)
        x0 = matrizCentral(vel_x)

        # Se elige el método de resolución del sistema J·H = -F
        match metodo:
            case 1:
                name = "Solución con el método de Newton"
                H = np.linalg.solve(J, F)
            case 2:
                name = "Solución con el método de Richardson"
                Q = np.eye(F.shape[0])
                H = Richardson(J, F, x0, Q, ITERACIONES_MAX, UMBRAL)
            case 3:
                name = "Solución con el método de Jacobi"
                H = metodo_jacobi(J, F, x0, ITERACIONES_MAX, UMBRAL)
            case 4:
                name = "Solución con el método de Gauss-Seidel"
                H = gauss_seidel(J, F, x0, ITERACIONES_MAX)
            case 5:
                name = "Solución con el método del Gradiente conjugado"
                H = gradiente_conjugado(J, F, x0, ITERACIONES_MAX, UMBRAL)
            case 6:
                name = "Solución con el método del Gradiente descendente"
                H = gradiente_descendente(J, F, x0, ITERACIONES_MAX, UMBRAL)
            case _:
                raise ValueError("Método no implementado.")

        H = np.round(H, DECIMALES)
        H = np.reshape(H, (FILAS-2, COLUMNAS-2))

        # se llena la matriz velocidad con los nuevos valores calculados
        # Xn+1 = Xn - OMEGA x J^(-1) x F
        Valores = matrizCentral(vel_x)
        vel_x[1:-1, 1:-1] -= np.round(OMEGA*H, DECIMALES)
        newValores = matrizCentral(vel_x)

        # abs((MatrizVelocidad[1,1] - H[0, 0]) - MatrizVelocidad[1,1]) < umbral
        # Se calcula la diferencia máxima entre la matriz anterior y la nueva
        # norma infinito de un vector
        delta = np.max(np.abs(newValores - Valores))

        print(f"Iteración {iteracion}, delta = {delta:.6f}")

        # Verificacion de convergencia
        if delta < UMBRAL:
            print(f"Convergencia alcanzada en la iteración {iteracion} con delta = {delta:.6f}")
            grafico_convergencia_velocidades(vel_x, name)
            break

    else:  
        print("\nNo se alcanzó convergencia dentro del número máximo de iteraciones.")

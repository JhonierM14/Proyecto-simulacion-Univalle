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
def metodo_newton_new(vel_x, vel_y, metodo):
    """
    Implementación corregida y robusta del método de Newton-Raphson.
    """
    for iteracion in range(ITERACIONES_MAX + 1):
        # Ensamblar la Jacobiana (J) y el vector de residuos (F)
        J = MatrizJacobiana(vel_x, vel_y, DECIMALES)
        F = llenarMatrizF(vel_x, vel_y, DECIMALES)

        # El sistema a resolver en Newton-Raphson es J * H = -F
        b = -F

        #  La conjetura inicial para el vector de actualización H debe ser un vector de ceros.
        h_inicial = np.zeros_like(b)

        try:
            # Se elige el método de resolución del sistema lineal J·H = b
            match metodo:
                case 1:
                    # Solución directa con el método de Newton (usando np.linalg.solve)
                    name = "Solución con el método de Newton"
                    H = np.linalg.solve(J, b)
                case 2:
                    # Solución con el método de Richardson
                    name = "Solución con el método de Richardson"
                    Q = np.eye(b.shape)
                    H = Richardson(J, b, h_inicial, Q, ITERACIONES_MAX, UMBRAL)
                case 3:
                    # Solución con el método de Jacobi
                    name = "Solución con el método de Jacobi"
                    H = metodo_jacobi(J, b, h_inicial, ITERACIONES_MAX, UMBRAL)
                case 4:
                    # Solución con el método de Gauss-Seidel
                    name = "Solución con el método de Gauss-Seidel"
                    H = gauss_seidel(J, b, h_inicial, ITERACIONES_MAX, UMBRAL)
                case 5:
                     # Solución con el método de Gradiente Conjugado
                    raise ValueError("Método  El Gradiente Conjugado")
                case 6:
                    # Solución con el método del Gradiente descendente
                    name = "Solución con el método del Gradiente descendente"
                    H = gradiente_descendente(J, b, h_inicial, ITERACIONES_MAX, UMBRAL)
                case _:
                    raise ValueError("Método no implementado.")

        #  Manejo de errores robusto en lugar de calcular el determinante.
        except np.linalg.LinAlgError:
            raise ValueError("La matriz Jacobiana es singular. Newton no puede continuar.")

        #  NO redondear en pasos intermedios para no perder precisión.
        H = np.reshape(H, (FILAS-2, COLUMNAS-2))

        # Se guarda una copia de los valores centrales antes de la actualización
        Valores = matrizCentral(vel_x).copy()

        #  La actualización es una suma (X_n+1 = X_n + H) porque H se calculó con -F.
        vel_x[1:-1, 1:-1] += OMEGA * H

        newValores = matrizCentral(vel_x)

        # Se calcula la diferencia máxima entre la matriz anterior y la nueva (norma infinito)
        delta = np.max(np.abs(newValores - Valores))

        print(f"Iteración {iteracion}, delta = {delta:.6f}")

        # Verificación de convergencia
        if delta < UMBRAL:
            print(f"Convergencia alcanzada en la iteración {iteracion} con delta = {delta:.6f}")
            grafico_convergencia_velocidades(vel_x, name)
            break
    else:
        # Este bloque se ejecuta si el bucle for termina sin un 'break'
        print("\nNo se alcanzó convergencia dentro del número máximo de iteraciones.")

    return vel_x
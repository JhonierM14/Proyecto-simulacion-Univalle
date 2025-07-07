from scipy.interpolate import RectBivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from metodos_Lineales.modulos.grafico import *

def aplicar_spline_2D(matriz_velocidad_x: np.ndarray, matriz_velocidad_y: np.ndarray, titulo="Spline 2D de Velocidad") -> np.ndarray:
    """
    Aplica un spline 2D sobre una matriz de velocidad y la grafica.

    Args:
        matriz_velocidad_x: np.ndarray (2D)
        matriz_velocidad_y: np.ndarray (2D)
        titulo: str, título para la gráfica
    """
    filas, columnas = matriz_velocidad_x.shape

    # Dominio original (rejilla de puntos)
    x = np.linspace(0, columnas - 1, columnas)
    y = np.linspace(0, filas - 1, filas)

    # Spline 2D
    spline = RectBivariateSpline(y, x, matriz_velocidad_x)

    # Nueva rejilla más fina (para suavizar)
    x_nuevo = np.linspace(0, columnas - 1, 200)
    y_nuevo = np.linspace(0, filas - 1, 100)

    # Evaluar spline en la nueva rejilla
    z_suavizado = spline(y_nuevo, x_nuevo)

    # Graficar
    grafico_convergencia_velocidades(z_suavizado, "Spline 2D")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import numpy as np
from matplotlib.colors import LogNorm

# ───────────────────────────────────────────────
# Funciones de gráficos
# ───────────────────────────────────────────────

# Colores que van de verde brillante a rojo oscuro (orden decreciente)
colores = [
    '#330000',  # Rojo oscuro
    '#4d0000',
    '#660000',
    '#800000',
    '#990000',
    '#b30000',
    '#cc0000',
    '#e60000',
    '#ff0000',
    '#ff3333',
    '#ff6666',
    '#ff9966',
    '#ffcc66',
    '#ffe066',
    '#ffff00',
    '#ccff00',
    '#99ff00',
    '#66ff00',
    '#33ff00',
    '#00ff00',  # Verde brillante
]

def graficoValoresIniciales(matriz):
    # Definir los valores límites (de 1 a 0.01 en 20 pasos)
    valores = np.linspace(1.0, 0.01, 20)

    # Crear el colormap y la normalización
    cmap = LinearSegmentedColormap.from_list("verde_a_rojo", colores, N=len(colores))
    norm = BoundaryNorm(boundaries=valores[::-1], ncolors=len(colores))

    plt.figure(figsize=(8, 6))
    plt.imshow(matriz, cmap=cmap, norm=norm, origin='upper', interpolation='nearest')
    plt.title("Mapa de Calor", fontsize=14)
    plt.colorbar(label='Valor')
    plt.xlabel("Columnas")
    plt.ylabel("Filas")
    plt.show()

def grafico_convergencia_velocidades(MatrizVelocidadEjeX):
    """
    Grafica una matriz de velocidades en escala de grises usando normalización logarítmica.
    Los valores deben estar entre 1e-8 y 1.
    
    Parámetros:
        MatrizVelocidadEjeX (ndarray): matriz 2D de velocidades.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(MatrizVelocidadEjeX, cmap='gray', norm=LogNorm(vmin=1e-8, vmax=1))
    plt.colorbar(label='Velocidad (log)')
    plt.title("Matriz de Velocidad (Escala Logarítmica en Grises)")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.show()

def grafico_vector_f(F_reshape):
    """Grafica la distribución del vector F usando el colormap 'coolwarm'."""
    plt.figure(figsize=(8, 4))
    plt.title("Distribución del Vector F")
    plt.imshow(F_reshape, cmap='coolwarm', origin='upper')
    plt.colorbar(label="Valor de F")
    plt.tight_layout()
    plt.show()

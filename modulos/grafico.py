import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np

# ───────────────────────────────────────────────
# Configuración del colormap personalizado
# ───────────────────────────────────────────────

# Límites reales de los datos (ajustables según la precisión de tus datos)
# BOUNDARIES = [
#     0.0,
#     1e-10,
#     1e-9,
#     1e-8,
#     1e-7,
#     1e-6,
#     1e-5,
#     1e-4,
#     1e-3,
#     1e-2,
#     1e-1,
#     1.0
# ]

# BOUNDARIES = [
#     1e-8,
#     5e-8,
#     1e-7,
#     5e-7,
#     1e-6,
#     5e-6,
#     1e-5,
#     5e-5,
#     1e-4,
#     5e-4,
#     1e-3,
#     1e-2,
#     1e-1,
#     1.0
# ]

# Nuevas divisiones, destacando la zona media
BOUNDARIES = [
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    1.0
]

# Paleta de colores: rojo oscuro → amarillo → verde
# COLOR_PALETTE = [
#     '#800000',  # rojo oscuro
#     '#990000',
#     '#b30000',
#     '#d73027',
#     '#fc8d59',
#     '#fee08b',
#     '#ffff00',
#     '#ccff66',
#     '#99ff66',
#     '#66ff66',
#     '#00ff00'   # verde brillante
# ]

# COLOR_PALETTE = [
#     '#2c003e',  # violeta oscuro
#     '#660066',  # púrpura
#     '#a30000',  # rojo oscuro
#     '#d73027',  # rojo intenso
#     '#fc8d59',  # naranja
#     '#fee08b',  # amarillo claro
#     '#ffffbf',  # amarillo pastel
#     '#e0f3f8',  # celeste muy claro
#     '#abd9e9',  # celeste
#     '#74add1',  # azul medio
#     '#4575b4',  # azul más intenso
#     '#313695',  # azul oscuro
#     '#253494',  # azul profundo
# ]

# Paleta de colores estilo "hotspot" (de frío a caliente)
COLOR_PALETTE = [
    '#ffffe0',  # amarillo muy claro
    '#fff7bc',
    '#fee391',
    '#fec44f',
    '#fe9929',
    '#ec7014',
    '#cc4c02',
    '#993404',
    '#662506',
    '#542788',
    '#2b2380',
    '#1b0c4e',
    '#0d0030',
    '#000022',
    '#000000'   # negro
]

# Crear colormap y normalizador asociado
custom_cmap = ListedColormap(COLOR_PALETTE, name="DeRojoAVerde")
custom_norm = BoundaryNorm(BOUNDARIES, ncolors=custom_cmap.N, clip=True)


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

def graficoValoresIniciales(matriz, eje):
    # Definir los valores límites (de 1 a 0.01 en 20 pasos)
    valores = np.linspace(1.0, 0.01, 20)

    # Crear el colormap y la normalización
    cmap = LinearSegmentedColormap.from_list("verde_a_rojo", colores, N=len(colores))
    norm = BoundaryNorm(boundaries=valores[::-1], ncolors=len(colores))

    plt.figure(figsize=(8, 6))
    plt.imshow(matriz, cmap=cmap, norm=norm, origin='upper', interpolation='nearest')
    plt.title("Valores Iniciales " + eje, fontsize=14)
    plt.colorbar(label='Valor')
    plt.xlabel("Columnas")
    plt.ylabel("Filas")
    plt.show()

# matriz final tras iteraciones
def grafico_matriz_velocidad(matriz):
    """Grafica la matriz de velocidad usando un mapa de color personalizado."""
    plt.figure(figsize=(10, 5))
    plt.title("Mapa de Calor - Velocidad en el Eje X", fontsize=14)
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    im = plt.imshow(matriz, cmap=custom_cmap, norm=custom_norm,
               interpolation="nearest", origin="upper")
    # plt.colorbar(label="Velocidad")
    # Usar notación científica en la colorbar
    cbar = plt.colorbar(im, format=LogFormatterSciNotation())
    cbar.set_label("Velocidad", rotation=270, labelpad=15)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def grafico_vector_f(F_reshape):
    """Grafica la distribución del vector F usando el colormap 'coolwarm'."""
    plt.figure(figsize=(8, 4))
    plt.title("Distribución del Vector F")
    plt.imshow(F_reshape, cmap='coolwarm', origin='upper')
    plt.colorbar(label="Valor de F")
    plt.tight_layout()
    plt.show()

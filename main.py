from Newton import *
from spline import *

vel_x, vel_y = inicializar_matrices()

graficoValoresIniciales(vel_x, "Valores iniciales vel_x")
graficoValoresIniciales(vel_y, "Valores iniciales vel_y (const 0.1)")

"METODOS "
"1 - Newton, "
"2 - Richardson, "
"3 - Jacobi, "
"4 - Gauss-Seidel, "
"5 - Gradiente conjugado, "
"6 - Gradiente descendente"

metodo_newton(vel_x, vel_y, 3)

aplicar_spline_2D(vel_x, vel_y, "Spline de matriz velocidad X final")
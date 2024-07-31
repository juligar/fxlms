import numpy as np
import matplotlib.pyplot as plt

# Parámetro para el número de iteraciones
n_iterations = 4000

# Vectores de entrada
x = np.random.randn(n_iterations)    # Señal de entrada aleatoria
h = np.array([0, 0, 0, 0, 1])        # Kernel para la convolución

# Convolución de los vectores
y = np.convolve(x, h)

# Crear una figura
plt.figure(figsize=(10, 6))

# Graficar la señal de entrada x
plt.subplot(3, 1, 1)
plt.plot(x, drawstyle='steps-post')
plt.title('Señal de entrada x[n]')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid()

# Graficar la señal de entrada h (con delay)
plt.subplot(3, 1, 2)
plt.plot(h, drawstyle='steps-post', marker='o')
plt.title('Kernel de convolución h[n]')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.grid()

# Graficar la señal de salida y
plt.subplot(3, 1, 3)
plt.plot(y, drawstyle='steps-post')
plt.title('Señal de salida y[n] = x[n] * h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid()

# Mostrar la gráfica
plt.tight_layout()
plt.show()

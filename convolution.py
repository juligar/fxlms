import numpy as np
import matplotlib.pyplot as plt

# Vectores de entrada
x = np.array([1, 2, 3, 4, 5])        # Señal de entrada
h = np.array([0, 0, 0, 0, 1])        # Kernel para la convolución

# Convolución de los vectores
y = np.convolve(x, h)

# Crear una figura
plt.figure(figsize=(10, 6))

# Graficar la señal de entrada x
plt.subplot(3, 1, 1)
plt.plot(x, drawstyle='steps-post', marker='o')
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
plt.plot(y, drawstyle='steps-post', marker='o')
plt.title('Señal de salida y[n] = x[n] * h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid()

# Mostrar la gráfica
plt.tight_layout()
plt.show()

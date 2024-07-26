import numpy as np
import matplotlib.pyplot as plt

# Parámetros del algoritmo LMS
mu = 0.01  # Stepsize
N = 1000   # Samples
L = 10     # Filter Lenght

# Generate White Noise as my input signal
x = np.random.normal(0, 1, N)

# Primary Path
h_deseado = np.random.normal(0, 1, L)
d = np.convolve(x, h_deseado)[:N]

# Initialize adaptive filter
w = np.zeros(L)
y = np.zeros(N)
e = np.zeros(N)

# Prepare the input matrix for the adaptive filter
X = np.array([x[i:i+L] for i in range(N-L)])

# Algoritmo LMS vectorizado
for n in range(L, N):
    x_n = X[n-L]
    y[n] = np.dot(w, x_n)
    e[n] = d[n] - y[n]
    w += 2 * mu * e[n] * x_n

# Graficar los resultados
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(d, label='Señal Deseada')
plt.plot(y, label='Salida del Filtro')
plt.title('Señal Deseada vs Salida del Filtro')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(e)
plt.title('Error')

plt.subplot(3, 1, 3)
plt.plot(w)
plt.title('Coeficientes')

plt.tight_layout()
plt.show()

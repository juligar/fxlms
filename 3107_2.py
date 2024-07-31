import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación
n_iterations = 4000  # Número de iteraciones
n_taps = 10          # Número de coeficientes del filtro adaptativo
mu = 0.001           # Tasa de aprendizaje ajustada

# Generar la señal de referencia x(n)
x = np.random.randn(n_iterations)

# Función de convolución manual ajustada
def convolucion_manual(x_timeHistory, h):
    len_h = len(h)
    y = 0.0  # Asegurar que y sea un float
    for j in range(len_h):
        y += x_timeHistory[j] * h[j]
    return y

# Modelo del camino primario P(z)
def primary_path(x, delay_primary=5):
    P = [0] * delay_primary + [1]
    x_timeHistory = [0.0] * len(P)  # Asegurar que el historial sea de floats
    y = []
    for x_n in x:
        x_timeHistory = np.roll(x_timeHistory, 1)
        x_timeHistory[0] = x_n
        y.append(convolucion_manual(x_timeHistory, P))
    return y

# Modelo del camino secundario S(z)
def secondary_path(y_timeHistory, delay_secondary=2):
    S = [0] * delay_secondary + [1]
    y = convolucion_manual(y_timeHistory, S)
    return y

# Estimación del camino secundario Ŝ(z)
def secondary_path_estimation(y_timeHistory, delay_secondary_est=2):
    S_hat = [0] * delay_secondary_est + [1]
    y = convolucion_manual(y_timeHistory, S_hat)
    return y

# Inicialización de variables
W = np.zeros(n_taps)        # Coeficientes del filtro adaptativo
e = np.zeros(n_iterations)  # Error
y = np.zeros(n_iterations)  # Salida del filtro adaptativo
z = np.zeros(n_iterations)  # Señal filtrada por el camino secundario
x_timeHistory = np.zeros(n_taps)  # Historial de la señal de referencia
y_timeHistory = np.zeros(n_taps)  # Historial de la salida del filtro adaptativo
v_timeHistory = np.zeros(n_taps)  # Historial de la señal estimada

# Generar la señal deseada d(n) utilizando el camino primario
d = primary_path(x)

# FxLMS Algorithm
for n in range(n_taps, n_iterations):
    # Señal de referencia retrasada
    x_timeHistory = np.roll(x_timeHistory, 1)
    x_timeHistory[0] = x[n]

    # Salida del filtro adaptativo
    y[n] = np.dot(W, x_timeHistory)

    # Actualizar historial de la salida del filtro adaptativo
    y_timeHistory = np.roll(y_timeHistory, 1)
    y_timeHistory[0] = y[n]

    # Filtrar la salida del filtro adaptativo a través del camino secundario
    z[n] = secondary_path(y_timeHistory, delay_secondary=2)

    # Error de salida
    e[n] = d[n] - z[n]

    # Estimación del camino secundario
    v = secondary_path_estimation(x_timeHistory, delay_secondary_est=2)

    # Actualizar historial de la señal estimada
    v_timeHistory = np.roll(v_timeHistory, 1)
    v_timeHistory[0] = v

    # Normalización de la señal de referencia
    norm_factor = np.dot(v_timeHistory, v_timeHistory) + 1e-6  # evitar división por cero
    v_timeHistory_norm = v_timeHistory / norm_factor

    # Actualización de los coeficientes del filtro adaptativo
    W += 2 * mu * e[n] * v_timeHistory_norm

# Graficar los resultados
plt.figure(figsize=(15, 10))

# Graficar la señal deseada
plt.subplot(3, 1, 1)
plt.plot(d, label='Señal deseada (ruido a cancelar)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()

# Graficar la salida del filtro adaptativo
plt.subplot(3, 1, 2)
plt.plot(y, label='Salida del filtro adaptativo')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()

# Graficar el error
plt.subplot(3, 1, 3)
plt.plot(e, label='Error')
plt.ylabel('Error')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Graficar la señal filtrada por el camino secundario en una figura por separado
plt.figure(figsize=(15, 5))
plt.plot(z, label='Señal filtrada por el camino secundario')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

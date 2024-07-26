import numpy as np

def convolucion_manual(x_timeHistory, h):
    # Longitudes de la señal de entrada y la respuesta al impulso
    len_x = len(x_timeHistory)
    len_h = len(h)
    
    # Longitud del resultado de la convolución
    len_y = len_x + len_h - 1
    
    # Inicializar el resultado con ceros
    y = 0
    
    # Realizar la convolución
    for j in range(len_h):
        y += x_timeHistory[j] * h[j]

    return y, x_timeHistory

# Ejemplo de uso
x = [1, 2, 3, 4, 5, 6, 7, 8]
h = [0, 0, 0, 1]
x_timeHistory = [0, 0, 0, 0]
result = []

for x_n in x:
    x_timeHistory = np.roll(x_timeHistory, 1) # Desplazar el vector x_timeHistory utilizando numpy.roll
    x_timeHistory[0] = x_n  # Actualizar el primer valor de x_timeHistory con el valor actual de x
    resultado, x_timeHistory = convolucion_manual(x_timeHistory, h)
    result.append(resultado)

print(result)

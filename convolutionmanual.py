def convolucion_manual(x, h):
    # Longitudes de la señal de entrada y la respuesta al impulso
    len_x = len(x)
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
x = [1, 2, 3, 4, 5, 6]
h = [0, 0, 0, 1]
x_timeHistory = [3, 2, 1, 0]
for x_n in x:
    
    resultado, x_timeHistory = convolucion_manual(x_timeHistory, h)
print(resultado)

# -*- coding: utf-8 -*-
"""
Genera el archivo datos_sinteticos_anillos.csv usado en el ejemplo
ejemplo_sintetico_pca_vs_kernelpca.ipynb.

El conjunto de datos tiene 1000 ejemplos y 10 variables (var_1 a var_10),
más una columna 'grupo' con la etiqueta de cada punto. La estructura es:
dos anillos concéntricos en 2D, 8 variables de ruido de poca varianza,
y una rotación aleatoria en 10 dimensiones que reparte la estructura
entre todas las variables. Finalmente, dos variables (var_3 y var_8) se
expresan en unidades distintas (x100 y x1000), de modo que el conjunto
requiere escalado antes de aplicar PCA o Kernel PCA.

Para regenerar o modificar los datos (por ejemplo, acercar los anillos
cambiando factor=0.3 por factor=0.7), edite los parámetros y ejecute:

    python generar_datos_sinteticos_anillos.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_circles

rng = np.random.RandomState(42)
n = 1000

# Paso 1: dos anillos concéntricos en 2D
# (grupo 0: anillo exterior de radio 1; grupo 1: anillo interior de radio 0.3)
X_anillos, y = make_circles(n_samples=n, factor=0.3, noise=0.05, random_state=42)

# Paso 2: 8 variables de ruido con poca varianza
# (matriz de 1000 x 8, distribución normal con media 0 y desviación estándar 0.02)
X_ruido = rng.normal(loc=0.0, scale=0.02, size=(n, 8))

# Paso 3: construimos una rotación aleatoria en 10 dimensiones.
# Generamos una matriz aleatoria de 10 x 10 y le aplicamos la descomposición QR:
# la matriz Q resultante es ortogonal, es decir, representa una rotación.
matriz_aleatoria = rng.normal(loc=0.0, scale=1.0, size=(10, 10))
Q, R = np.linalg.qr(matriz_aleatoria)   # R no se utiliza

# Unimos las 2 variables de los anillos con las 8 de ruido (1000 x 10)
# y aplicamos la rotación multiplicando por Q
X_sin_rotar = np.hstack([X_anillos, X_ruido])
X = X_sin_rotar @ Q

# Paso 4: simulamos que dos variables fueron medidas en unidades distintas
# (sus valores quedan cientos o miles de veces más grandes que los demás)
X[:, 2] *= 100     # var_3
X[:, 7] *= 1000    # var_8

# Guardamos todo como CSV: las 10 variables más la columna de grupo
datos = pd.DataFrame(X, columns=['var_%d' % (i + 1) for i in range(10)])
datos['grupo'] = y

ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'datos_sinteticos_anillos.csv')
datos.to_csv(ruta, index=False)
print("Archivo generado:", ruta)
print("Dimensiones:", datos.shape)

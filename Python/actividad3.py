
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = {
    'Ruta': ['R1', 'R1', 'R2', 'R2'],
    'Estación Origen': ['Estación A', 'Estación B', 'Estación A', 'Estación D'],
    'Estación Destino': ['Estación B', 'Estación C', 'Estación D', 'Estación E'],
    'Tiempo de Viaje': [10, 8, 15, 20],
    'Demanda de Pasajeros': [45, 60, 30, 20],
    'Día Semana': ['Lunes', 'Lunes', 'Martes', 'Martes'],
    'Hora Pico': [1, 0, 1, 0]  # 1 para Sí, 0 para No
}


df = pd.DataFrame(data)


X = df[['Demanda de Pasajeros', 'Hora Pico']]
y = df['Tiempo de Viaje']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


modelo = DecisionTreeRegressor()
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)

# Se evalua el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error Cuadrático Medio: {mse}')

# Aquí graficamos los resultados
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicción del Tiempo de Viaje')
plt.show()

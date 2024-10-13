import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


datos = {
    'Estacion': ['A', 'B', 'C', 'D', 'E'],
    'Pasajeros_por_dia': [200, 150, 300, 350, 400],
    'Distancia_a_centro': [5, 10, 3, 7, 2]  
}

df = pd.DataFrame(datos)


X = df[['Pasajeros_por_dia', 'Distancia_a_centro']]


kmeans = KMeans(n_clusters=2)
kmeans.fit(X)


df['Cluster'] = kmeans.labels_


print(df)


plt.scatter(df['Pasajeros_por_dia'], df['Distancia_a_centro'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Pasajeros por día')
plt.ylabel('Distancia al centro (km)')
plt.title('Agrupamiento de estaciones de transporte masivo')
plt.show()

import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style("darkgrid")
from sklearn import linear_model
import random

import warnings
warnings.filterwarnings('ignore')

# Levanto los datos
df = pd.read_csv('./data/worldTemperature.csv', ' ')
print("Decripcion de los datos")
print(df.describe())

# print("\nLas primeras 8 filas de los datos")
# print(df.head(8))

# Graficos los datos
ax = sns.tsplot(time=df['x'], data=df['y'], interpolate=False)
sns.plt.show()

# Limpio los outliers
def mascara_outliers(s):
    return abs(s - s.mean()) <= 2*s.std()
    
mascara = mascara_outliers(df['y'])
df = df[mascara]
sns.tsplot(time=df['x'], data=df['y'], interpolate=False)

# Para que se evidencien mejor los outliers que saque, grafico en el mismo rango
# Algunos me parecen que no deberian ser removidos
sns.plt.ylim((6.5, 10.5))
sns.plt.show()

# Uso cuadrados minimos para predecir
sns.tsplot(time=df['x'], data=df['y'], interpolate=False)

def armar_matriz_A(s):
    temp = []
    for t in s:
        temp.append([t**2, np.cos(t), np.sin(t), t, 1])
    return np.array(temp)

def entrenar_y_predecir_en_rangos(df, rango_entrenamiento, rango_prediccion):
    regr = linear_model.LinearRegression(fit_intercept=False)

    # Entreno el modelo
    df_e = df[df['x'].isin(rango_entrenamiento)]
    A_e = armar_matriz_A(df_e['x'])
    regr.fit(A_e, df_e['y'])

    # Me fijo la aproximacion que se realizo
    df_e['p'] = regr.predict(A_e)
    sns.tsplot(time=df_e['x'], data=df_e['p'], color='r')

    # Realizo predicciones
    df_p = df[df['x'].isin(rango_prediccion)]

    A_p = armar_matriz_A(df_p['x'])
    df_p['p'] = regr.predict(A_p)
    sns.tsplot(time=df_p['x'], data=df_p['p'], color='g')

    ECM = sum((df_p['p']-df_p['y'])**2)
    return ECM

def predecir(k):
    return entrenar_y_predecir_en_rangos(df, range(k-30,k), range(k,k+10))

ECM = predecir(1930)
print("Error de prediccion: {:.2f}".format(ECM))    

ECM = predecir(1960)
print("Error de prediccion: {:.2f}".format(ECM))

ECM = predecir(1990)
print("Error de prediccion: {:.2f}".format(ECM))

sns.plt.xlim((1880, 2012))
sns.plt.show()

# Un par de cosas mas de pandas
# years = [2004]*6 + [2005]*6 + [2006]*6
# delays = list(np.random.randn(6)+80) + list(np.random.randn(6)+5) + list(np.random.randn(6)+50)
# delays[2] = 5
# delays[7] = 80

# df1 = pd.DataFrame({
#     'year': years,
#     'delay': delays
# })

# print df1

# # Groupby y sacar outliers
# mascara = df1.groupby('year')['delay'].apply(mascara_outliers)
# df1 = df1[mascara]
# print df1

# # Groupby y calcular promedio
# promedio = df1.groupby('year').aggregate(['mean', 'std', 'count'])
# print promedio
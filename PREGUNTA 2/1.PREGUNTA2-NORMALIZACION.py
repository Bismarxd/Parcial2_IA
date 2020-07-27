# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:56:06 2020

@author: bisma
"""
#IMPORTAMOS LOS ARCHIVOS NECESARIOS CON EL PIPELINE SCIKITLEAR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

#LEEMOS EL DATA SET DESCARGADO DE UCI
datos = pd.read_csv('Datos.csv')
datos = datos.replace(np.nan,"0")
df = pd.DataFrame(datos)

#LEEMOS LOS VALORES DE LA MEDIA, DESVIACIÓN ESTANDAR, MAXIMO Y MINIMO
print("Media")
print(df['X'].mean())
print("Desviación Estandar")
print(df['X'].std())
print("Valor Maximo")
print(df['X'].max())
print("Valor Minimo")
print(df['X'].min())

#GRAFICA
'''fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,5))
ax1.set_title('Antes')
sns.kdeplot(df('X'), ax=ax1)
sns.kdeplot(df('Y'), ax=ax1)'''

#HACIENDO EL PROCESAMIENTO-NORMALIZACIÓN
scaler = preprocessing.Normalizer(norm='l2', copy=True)
df[['Y','X']]=scaler.fit_transform(df[['Y','X']])

print("*"*20)
print("Media")
print(df['X'].mean())
print("Desviación Estandar")
print(df['X'].std())
print("Valor Maximo")
print(df['X'].max())
print("Valor Minimo")
print(df['X'].min())
'''
df.to_csv('prepro2.csv', sep='\t')
ax2.set_title('Despues de escalar')
sns.kdeplot(df('X'), ax=ax1)
sns.kdeplot(df('Y'), ax=ax1)
plt.show()
´'''
print("*"*20)
print(df['X'].iloc[0])



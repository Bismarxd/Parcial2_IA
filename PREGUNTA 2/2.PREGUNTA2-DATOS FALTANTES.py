# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:10:53 2020

@author: bisma
"""
#PROCESAMIENTO DE LOS DATOS FALTANTES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cargamos el conjunto de datos
dataset = pd.read_csv('Datos.csv')

# Separamos las variables dependientes e independientes
x = dataset.iloc[:, :0].values
y = dataset.iloc[:, 1].values

#Imputacion de datos faltantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis = 0)
imputer = imputer.fit(x[:, 1:2])
x[:, 1:2] = imputer.transform(x[:, 1:2])

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[2,1] = labelencoder_x.fit_transform(x[2,1])





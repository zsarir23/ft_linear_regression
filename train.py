import pandas as pd
from sklearn.preprocessing import StandardScaler
from linearRegression import LinearRegression
import numpy as np


df = pd.read_csv('./data.csv')
X = df['km']. values
print(X)
Y = df['price']. values


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
#X_test_scaled = scaler.transform(X_test)

lreg = LinearRegression(lr = 0.01, n_iters=10000)
lreg.fit(X_train_scaled, Y)

lreg.save_parameters()





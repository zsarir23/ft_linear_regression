import pandas as pd
from sklearn.preprocessing import StandardScaler
from linearRegression import LinearRegression
import numpy as np
import matplotlib as plt


df = pd.read_csv('./data.csv')
X = df[['km']]. values
Y = df['price']. values


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X).flatten()
#X_test_scaled = scaler.transform(X_test)

lreg = LinearRegression(lr = 0.01, n_iters=10000)
lreg.fit(X_train_scaled, Y)

lreg.save_parameters()
prediction = lreg.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Training Data')
plt.plot(X, prediction, color='red', linewidth=1, label='Predictions')
plt.xlabel('Kilometers')
plt.ylabel('Price')
plt.legend()
plt.show()





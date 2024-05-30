from cmath import nan
import numpy  as np
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
import argparse

def mean(mylist: list):

	if len(mylist) == 0:
		raise Exception('Empty data')
	else:
		return np.sum(mylist)/len(mylist)


def std(mylist: list):
	if len(mylist) == 0:
		raise Exception('Empty data')
	else:
		n = len(mylist)
		_mean = mean(mylist)
		var = np.sum([(x - _mean)**2 for x in mylist])/n
		return sqrt(var)

class LinearRegression:
	def __init__(self, lr=0.01, n_iters = 1000) -> None:
		self.lr = lr
		self.n_iters = n_iters
		self.weight = 0
		self.bias = 0

	
	def fit(self, x, y):
		if len(x) == 0 or len(y) == 0:
			raise ValueError("Empty data provided for training")
		n_samples = len(x)
		self.weight = 0
		self.bias = 0
		x_scaled = (x - mean(x))/std(x)

		for n in range(self.n_iters):
			y_pred = self.predict(x_scaled)
			dw = np.sum( x_scaled * (y_pred - y))/n_samples
			db = (1/n_samples) * np.sum(y_pred - y)
	
			self.weight -= (self.lr * dw)
			self.bias -= self.lr * db
		self.unscaler(x)
	

	def predict(self, x):
		y_pred = x * self.weight + self.bias
		return y_pred

	def unscaler(self, x):
		self.bias = self.bias - (self.weight * mean(x)) / std(x)
		self.weight = self.weight / std(x)
	
	def save_parameters(self, file_path='parameters.txt'):
		with open(file_path, 'w') as f:
			f.write(f"{self.weight},{self.bias}")

	def load_parameters(self, file_path='parameters.txt'):
		try:
			with open(file_path, 'r') as f:
				weight, bias = f.read().split(',')
				self.weight = float(weight)
				self.bias = float(bias)
		except FileNotFoundError:
			self.weight = 0
			self.bias = 0

	@staticmethod
	def r2_score(y, y_pred):
		y_mean = mean(y)
		ss_res = sum((y - y_pred)**2)
		ss_tot = sum((y - y_mean)**2)
		return 1 - ss_res/ss_tot

	def plot(self, x, y):
		y_pred = self.predict(x)
		r2_score = LinearRegression.r2_score(y, y_pred)
		plt.figure(figsize=(8, 6))
		plt.scatter(x, y, color='blue', label='Training Data')
		plt.plot(x, y_pred, color='red', linewidth=1, label='Predictions')
		plt.title(f'Linear Regression precision: {r2_score * 100 :.2f}%')
		plt.xlabel('Kilometers')
		plt.ylabel('Price')
		plt.legend()
		plt.show()

def main():
    parser = argparse.ArgumentParser(description='Linear Regression Training')
    parser.add_argument('-v', '--visualize', action='store_true', help='Plot the results')
    args = parser.parse_args()
    df = pd.read_csv('./data.csv')
    X = df['km']. values
    Y = df['price']. values

    lreg = LinearRegression()
    lreg.fit(X, Y)
    lreg.save_parameters()
    if args.visualize:
        lreg.plot(X, Y)


if __name__ == '__main__':
    main()
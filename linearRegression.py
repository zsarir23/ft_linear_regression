from cmath import nan
import numpy  as np

class LinearRegression:
	def __init__(self, lr=0.001, n_iters = 1000) -> None:
		self.lr = lr
		self.n_iters = n_iters
		self.weight = 0
		self.bias = 0
	
	def fit(self, X, y):
		n_samples = len(X)
		self.weight = 0
		self.bias = 0

		for n in range(self.n_iters):
			y_pred = np.dot(X, self.weight) + self.bias
			dw = (1/n_samples) * np.dot(X, y_pred - y)
			db = (1/n_samples) * sum(y_pred - y)
	
			self.weight -= (self.lr * dw)
			self.bias -= self.lr * db

	def predict(self, X):
		y_pred = np.dot(X, self.weight) + self.bias
		return y_pred
	
	def save_parameters(self, file_path='parameters.txt'):
		with open(file_path, 'w') as f:
			f.write(f"{self.weight},{self.bias}")

	@staticmethod
	def estimate_price(mileage, theta0, theta1):
		return theta0 + theta1 * mileage

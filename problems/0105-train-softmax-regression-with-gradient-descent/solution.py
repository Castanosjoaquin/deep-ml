import numpy as np
def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	X = add_bias(X)      # n x m  (m = features + 1, bias primero)
	n, m = X.shape

	y = y.reshape(-1)
	K = len(np.unique(y))
	Y = one_hot(y, K)     # n x K

	coeffs = np.zeros((K, m))   # C x M, como pide el hint
	losses = []

	for i in range(iterations):
		y_pred = softmax(X @ coeffs.T)   # (n x m) @ (m x K) = n x K
		res = y_pred - Y                  # n x K

		loss = -np.sum(Y * np.log(y_pred + 1e-15)) 
		losses.append(loss)

		coeffs = coeffs - learning_rate * (res.T @ X)    # (K x n) @ (n x m) = K x m

	return coeffs, losses
def one_hot(y, K):
	n = len(y)
	Y = np.zeros((n, K))
	Y[np.arange(n), y] = 1
	return Y
def softmax(Z):
	Z = Z - np.max(Z, axis=1, keepdims=True)
	dividendo = np.exp(Z).sum(axis=1, keepdims=True)
	return np.exp(Z) / dividendo

def add_bias( X):
	n = X.shape[0]
	bias = np.ones((n, 1))
	return np.hstack([bias, X])

	
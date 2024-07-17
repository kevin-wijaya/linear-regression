import numpy as np

# Ordinary Least Squares (OLS)
class OLS:
    """
    Linear Regression with Ordinary Least Squares
    
    Attributes:
    ----------
    beta_ : np.ndarray
        The coefficients of the linear model 
    """
    
    def __init__(self) -> None:
        self.beta_ = None
        
    def fit(self, X:np.ndarray, Y:np.ndarray) -> None:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.beta_ = np.linalg.inv(X.T @ X) @ (X.T @ Y) # SSE
        
    def predict(self, X:np.ndarray) -> np.ndarray:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta_
    
# Gradient Descent (GD) ~ Batch Gradient Descent
class GD:
    """
    Linear Regression with Batch Gradient Descent
    
    Attributes:
    ----------
    beta_ : np.ndarray
        The coefficients of the linear model
    learning_rate : float
        Learning rate for the gradient descent algorithm
    max_iter : int
        Maximum number of iteration for the gradient descent algorithm
    tolerance : float
        The convergence threshold, optimization stops when changes
        are smaller than this value        
    """
    
    def __init__(self, learning_rate=1e-2, max_iter=1000, tolerance=1e-7):
        self.beta = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X:np.ndarray, Y:np.ndarray) -> None:
        self.beta_ = np.zeros((X.shape[1] + 1))
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        prev_beta = self.beta_.copy()
        for _ in range(int(self.max_iter)):
            d_beta = 2 / X.shape[0] * X.T @ (X @ self.beta_ - Y) # MSE
            self.beta_ -= self.learning_rate * d_beta   
            if np.linalg.norm(self.beta_ - prev_beta) < self.tolerance: break
            prev_beta = self.beta_.copy()
        
    def predict(self, X:np.ndarray) -> np.ndarray:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta_  
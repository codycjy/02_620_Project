import numpy as np

class CustomLASSO:
    """
    alpha : L1 penalty coefficient
    
    learning_rate : Initial learning rate for gradient descent, default=0.01
        
        
    learning_rate_schedule : 'constant' uses the same learning rate
        throughout training, 'optimal' uses 1.0 / (alpha * (t + t0)).
        
    random_state : Random seed for reproducibility, default=42
        
        
    warm_start : When set to True, reuse the solution of the previous call to fit as
        initialization for the next fit call, default=True
        
    """
    
    def __init__(self, alpha=0.01, learning_rate=0.01, learning_rate_schedule='optimal',
                 random_state=42, warm_start=True):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.random_state = random_state
        self.warm_start = warm_start
        
        # Model parameters
        self.weights = None
        self.intercept = 0.0
        
        # Internal tracking
        self.t = 0  # Global step counter for learning rate scheduling
        
    def _get_learning_rate(self):
        
        if self.learning_rate_schedule == 'constant':
            return self.learning_rate
        elif self.learning_rate_schedule == 'optimal':
            # t0 is determined automatically based on alpha
            t0 = 1.0 / self.alpha
            return self.learning_rate / (self.t + t0)
        else:
            raise ValueError("Unknown learning rate schedule")
    
    def _soft_threshold(self, x, threshold):
        
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        
        # Reset internal state if not warm starting
        if not self.warm_start or self.weights is None:
            _, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.intercept = 0.0
            self.t = 0
        
        self.partial_fit(X, y)
        
        return self
    
    def partial_fit(self, X, y):
        """
        Incrementally fit the model to batches of samples,
        used for mini-batch

        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        # Initialize weights if not already done
        if self.weights is None:
            self.weights = np.zeros(n_features)
            self.intercept = 0.0
        
        # Shuffle the data
        random_state = np.random.RandomState(self.random_state + self.t)
        indices = random_state.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        # Process each sample
        for i in range(n_samples):
            self.t += 1
            lr = self._get_learning_rate()
            
            xi = X[i]
            yi = y[i]
            
            # Predict and calculate error
            pred = np.dot(xi, self.weights) + self.intercept
            error = pred - yi
            
            # Calculate gradients
            grad_weights = error * xi
            grad_intercept = error
            
            # Gradient clipping to prevent explosion
            if np.max(np.abs(grad_weights)) > 10:
                grad_scale = 10 / np.max(np.abs(grad_weights))
                grad_weights *= grad_scale
                grad_intercept *= grad_scale
            
            # Update weights and intercept
            self.weights -= lr * grad_weights
            self.intercept -= lr * grad_intercept
            
            # Apply L1 penalty (soft thresholding)
            threshold = lr * self.alpha
            self.weights = self._soft_threshold(self.weights, threshold)
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model
        
        """
        X = np.asarray(X)
        return np.dot(X, self.weights) + self.intercept
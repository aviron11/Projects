import numpy as np

class Loss:
    def __init__(self):
        pass

    def softmax(self, X, W, b):
        """
        Compute the softmax of the input X with weights W and bias b.

        Parameters:
        X (numpy.ndarray): Input data, shape (n, m) where n is the number of features and m is the number of samples.
        W (numpy.ndarray): Weights, shape (n, l) where n is the number of features and l is the number of classes.
        b (numpy.ndarray): Bias, shape (l, 1) where l is the number of classes.

        Returns:
        numpy.ndarray: Softmax probabilities, shape (m, l)
        """
        XTW = np.dot(X.T, W) + b.T  # Transpose b to match the shape (1, l)
        eta = np.max(XTW, axis=1, keepdims=True)
        e_XTW = np.exp(XTW - eta)
        softmax_probs = e_XTW / np.sum(e_XTW, axis=1, keepdims=True)
        return softmax_probs
    
    def softmax_predictions(self, probabilities):
        """
        Compute the predicted class for each sample based on the softmax probabilities.

        Parameters:
        probabilities (numpy.ndarray): Softmax probabilities, shape (m, l) where m is the number of samples and l is the number of classes.

        Returns:
        numpy.ndarray: Predicted class indices for each sample, shape (m,)
        """
        return np.argmax(probabilities, axis=1).reshape(-1, 1)
    
    def cross_entropy_loss(self, predictions, C):
        """
        Compute the cross-entropy loss.

        Parameters:
        predictions (numpy.ndarray): Predicted probabilities, shape (m, l) where m is the number of samples and l is the number of classes.
        C (numpy.ndarray): True labels, shape (m, l) where m is the number of samples and l is the number of classes.

        Returns:
        float: Cross-entropy loss
        """
        m = C.shape[0]
        log_predictions = np.log(predictions + 1e-9)  # 1e-9 Adding a small value to avoid log(0) maybe not necessary
        loss = (-1/m) * np.sum(C * log_predictions)
        return loss
    
    def cross_entropy_gradient(self, predictions, C, X, W):
        """
        Compute the gradient of the cross-entropy loss with respect to the weights.

        Parameters:
        X (numpy.ndarray): Input data, shape (n, m) where n is the number of features and m is the number of samples.
        predictions (numpy.ndarray): Predicted probabilities, shape (m, l) where m is the number of samples and l is the number of classes.
        C (numpy.ndarray): True labels, shape (m, l) where m is the number of samples and l is the number of classes.

        Returns:
        tuple: Gradients of the loss with respect to the weights and biases, shapes (n, l) and (l, 1)
        """
        m = X.shape[1]
        gradient = (predictions - C) / m
        grad_W = np.dot(X, gradient)
        grad_b = np.sum(gradient, axis=0, keepdims=True).T  # Transpose to match the shape (l, 1)
        grad_X = np.dot(W, gradient.T)
        return grad_W, grad_b, grad_X
    
    def least_squares_loss(self, X, y, W, b):
        """
        Compute the least squares loss.

        Parameters:
        X (numpy.ndarray): Input data, shape (n, m) where n is the number of features and m is the number of samples.
        y (numpy.ndarray): True labels, shape (m, 1).
        W (numpy.ndarray): Weights, shape (n, l).
        b (numpy.ndarray): Biases, shape (l, 1).

        Returns:
        float: Least squares loss.
        """
        predictions = X.T @ W + b.T  # Compute predictions with bias
        y.reshape(-1, 1)  # Ensure y is of shape (m, 1)
        errors = predictions - y
        cost = (1 / X.shape[1]) * np.sum(errors ** 2)
        return cost

    def least_squares_gradient(self, X, y, W, b):
        """
        Compute the gradient of the least squares loss with respect to the weights and biases.

        Parameters:
        X (numpy.ndarray): Input data, shape (n, m) where n is the number of features and m is the number of samples.
        y (numpy.ndarray): True labels, shape (m, 1).
        W (numpy.ndarray): Weights, shape (n, l).
        b (numpy.ndarray): Biases, shape (l, 1).

        Returns:
        tuple: Gradients of the loss with respect to the weights and biases, shapes (n, l) and (l, 1).
        """
        m = X.shape[1]  # Number of samples
        l = W.shape[1]  # Number of output dimensions
        y = y.reshape(-1, 1)  # Ensure y is of shape (m, 1)
        predictions = X.T @ W + b.T  # Shape: (m, l)
        errors = predictions - y  # Shape: (m, l)
        grad_W = (2 / m) * (X @ errors)  # Shape: (n, l)
        grad_b = (2 / m) * np.sum(errors, axis=0, keepdims=True).T  # Shape: (l, 1)
        return grad_W, grad_b

    # Prediction function
    def least_squares_predictions(self, X, W, b):
        """
        Compute predictions for least squares.

        Parameters:
        X (numpy.ndarray): Input data, shape (n, m) where n is the number of features and m is the number of samples.
        W (numpy.ndarray): Weights, shape (n, l).
        b (numpy.ndarray): Biases, shape (l, 1).

        Returns:
        numpy.ndarray: Predictions, shape (m, l).
        """
        return X.T @ W + b.T

# according to notes:
# c is m x l
# x is n x m
# w is n x l

# according to print shapes
# c is l x m  need to transpose


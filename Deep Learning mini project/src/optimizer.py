import numpy as np
import matplotlib.pyplot as plt
from loss import Loss

class Optimizer:
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def create_mini_batches(self, indices, batch_size):
        """
        Create mini-batches from a list of indices.

        Parameters:
        indices (numpy.ndarray or list): Array or list of data indices to be split into mini-batches.
        batch_size (int): The number of indices in each mini-batch.

        Returns:
        list: A list of numpy arrays, where each array contains indices for one mini-batch.
        """
        mini_batches = [
            indices[i:i + batch_size] for i in range(0, len(indices), batch_size)
        ]
        return mini_batches
    
    def SGD(self, X, y, C, W, b, loss, epochs, X_val=None, y_val=None, C_val=None, plot=True, convergence_threshold=1e-6):
        """
        Perform Stochastic Gradient Descent (SGD) optimization.

        Parameters:
        X (numpy.ndarray): Input data, shape (n, m) where n is the number of features and m is the number of samples.
        y (numpy.ndarray): True labels, shape (m,).
        C (numpy.ndarray): One-hot encoded labels, shape (m, l).
        W (numpy.ndarray): Initial weights, shape (n, l).
        b (numpy.ndarray): Initial biases, shape (l,).
        loss (object): Loss object with methods to compute loss, gradient, and predictions.
        epochs (int): Number of full iterations over the dataset.
        X_val (numpy.ndarray, optional): Validation input data, shape (n, m_val).
        y_val (numpy.ndarray, optional): Validation true labels, shape (m_val,).
        C_val (numpy.ndarray, optional): Validation one-hot encoded labels, shape (m_val, l).
        convergence_threshold (float): Threshold for convergence based on the relative change in the residual norm.

        Returns:
        tuple: Updated weights and biases after optimization, training losses, training success percentages, validation losses, validation success percentages.
        """
        m = X.shape[1]  # Number of samples
        indices = np.arange(m)  # Indices for shuffling

        losses = []  # List to track training loss values
        success_percentages = []  # List to track training accuracy
        residual_norms = []  # List to track residual norms
        val_losses = []  # List to track validation loss values
        val_success_percentages = []  # List to track validation accuracy

        for k in range(epochs):
            
            np.random.shuffle(indices)  # Shuffle data indices
            mini_batches = self.create_mini_batches(indices, self.batch_size)

            for mini_batch in mini_batches:
                # Extract mini-batch data
                X_batch = X[:, mini_batch]
                y_batch = y[mini_batch]
                
                # Compute gradient
                if len(C) > 0:
                    predictions = loss.softmax(X_batch, W, b)
                    C_batch = C[mini_batch, :]  
                    grad_W, grad_b, _ = loss.cross_entropy_gradient(predictions, C_batch, X_batch, W)
                else:
                    # just for least squares example
                    predictions = loss.least_squares_predictions(X_batch, W, b)
                    grad_W, grad_b = loss.least_squares_gradient(X_batch, y_batch, W, b)

                # Update weights and biases
                W -= self.learning_rate * grad_W
                b -= self.learning_rate * grad_b

            # Compute training loss
            if len(C) > 0:
                predictions = loss.softmax(X, W, b)
                running_loss = loss.cross_entropy_loss(predictions, C)
                predictions = loss.softmax_predictions(predictions)
            else:
                predictions = loss.least_squares_predictions(X, W, b)
                running_loss = loss.least_squares_loss(X, y, W, b)

            # Compute training success percentage
            success_count = np.sum(predictions == y)
            success_percentage = (success_count / m) * 100
            
            # Compute residual norm
            residual = predictions - y
            residual_norm = np.linalg.norm(residual)
            
            # Track training metrics
            losses.append(running_loss)
            success_percentages.append(success_percentage)
            residual_norms.append(residual_norm)
            
            # Compute validation loss and success percentage
            if X_val is not None and y_val is not None:
                
                val_predictions = loss.softmax(X_val, W, b)
                val_running_loss = loss.cross_entropy_loss(val_predictions, C_val)
                val_predictions = loss.softmax_predictions(val_predictions)
                val_success_count = np.sum(val_predictions == y_val)
                
                
                val_success_percentage = (val_success_count / X_val.shape[1]) * 100
                val_losses.append(val_running_loss)
                val_success_percentages.append(val_success_percentage)
        
            # Check for convergence
            if k > 0 and residual_norms[-1] / residual_norms[-2] < convergence_threshold:
                print(f"Convergence reached at iteration {k}")
                break

        if plot:
            self.plot_sgd_results(losses, success_percentages, "Training Loss and Success Percentage")
            if X_val is not None and y_val is not None:
                self.plot_sgd_results(val_losses, val_success_percentages, "Validation Loss and Success Percentage")

        return W, b, losses, success_percentages, val_losses, val_success_percentages


    def plot_sgd_results(self, losses, success_percentages, title):
        
        """
        Plot the results of the SGD optimization.

        Parameters:
        losses (list): List of loss values per iteration.
        success_percentages (list): List of success percentages per iteration.
        title (str): Title for the entire figure.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Average F(W) per iteration', color='blue')
        plt.scatter(len(losses) - 1, losses[-1], color='red', marker='*', 
                    zorder=5, label=f'Final F(W): {losses[-1]:.3f}')  # Mark the final value
        plt.xlabel('Full Iterations')
        plt.ylabel('Value of F(W)')
        plt.title('Evolution of F(W) during SGD (per full iteration)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(success_percentages, label='Success Percentage', color='green')
        plt.scatter(len(success_percentages) - 1, success_percentages[-1], color='red', marker='*', 
                    zorder=5, label=f'Final Success: {success_percentages[-1]:.3f}')  # Mark the final value
        plt.xlabel('Full Iterations')
        plt.ylabel('Success Percentage')
        plt.title('Success Percentage during SGD (per full iteration)')
        plt.legend()

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
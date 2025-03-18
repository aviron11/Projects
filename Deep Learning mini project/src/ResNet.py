import numpy as np
from layer import ResNetLayer, NNLayer
from loss import Loss
from optimizer import Optimizer
import matplotlib.pyplot as plt

class ResNet:
    def __init__(self, layers_config, learning_rate, batch_size):
        """
        Initialize the ResNet with the given layers configuration.

        Parameters:
        layers_config (list of tuples): Each tuple contains (input_dim, output_dim, activation)
        learning_rate (float): Learning rate for the optimizer
        """
        self.num_layers = len(layers_config)
        self.layers = []
        self.lr = learning_rate
        self.optimizer = Optimizer(learning_rate=learning_rate, batch_size=batch_size)
        
        for i, (input_dim, output_dim, activation) in enumerate(layers_config):
            if i == self.num_layers - 1:
                self.layers.append(NNLayer(input_dim, output_dim, activation))
            else:
                self.layers.append(ResNetLayer(input_dim, output_dim, activation))

    def forward(self, X):
        """
        Perform forward pass through the network.

        Parameters:
        X (numpy.ndarray): Input data, shape (n, m) where n is the number of features and m is the number of samples.

        Returns:
        numpy.ndarray: Output of the network
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output.T

    def backward(self, pred, C):
        """
        Perform backward pass through the network and update weights.

        Parameters:
        pred (numpy.ndarray): Predicted probabilities from the forward pass.
        C (numpy.ndarray): True labels.

        Returns:
        dict: Dictionary containing gradients for each layer.
        """
        grad_propagate = pred, C    

        # Backward pass - all layers
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            grad_propagate = layer.backward(grad_propagate)[0]
            
        # Update parameters - all layers
        for layer in self.layers:
            layer.update(self.lr)
        
    
    # only for testing - backward with no update
    def backprop(self, pred, C):
       
        # Dictionary to store gradients for tests
        gradients = {}
        grad_propagate = pred, C    

        # Backward pass - all layers
        for i in reversed(range(self.num_layers)):
            
            layer = self.layers[i]
            grad_propagate = layer.backward(grad_propagate)[0]
            
            if isinstance(layer, NNLayer):
                # Store gradients for the layer
                gradients[f'layer_{i}'] = {
                    'grad_W': layer.grad_Weights,
                    'grad_b': layer.grad_Bias
                }
            else:
                # Store gradients for the layer
                gradients[f'layer_{i}'] = {
                    'grad_W1': layer.grad_W1,
                    'grad_W2': layer.grad_W2,
                    'grad_b': layer.grad_b
                }
    
        return gradients

    def train(self, X_train, y_train, C_train, epochs, batch_size, X_val=None, y_val=None, C_val=None):
        """
        Train the neural network and plot training progress.

        Parameters:
        X_train (numpy.ndarray): Training input data, shape (n, m) where n is the number of features and m is the number of samples.
        y_train (numpy.ndarray): Training true labels, shape (m,) where m is the number of samples.
        C_train (numpy.ndarray): Training indicators, shape (m, l) where m is the number of samples and l is the number of classes.
        epochs (int): Number of training epochs
        batch_size (int): Size of each mini-batch
        X_val (numpy.ndarray, optional): Validation input data, shape (n, m_val)
        y_val (numpy.ndarray, optional): Validation true labels, shape (m_val,)
        C_val (numpy.ndarray, optional): Validation indicators, shape (m_val, l)
        """
        loss_function = Loss()
        m = X_train.shape[1]

        metrics = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }

        for epoch in range(epochs):
            # Shuffle the training data
            indices = np.arange(m)
            np.random.shuffle(indices)
            mini_batches = self.optimizer.create_mini_batches(indices, batch_size)

            epoch_loss = 0

            # Mini-batch training
            for mini_batch in mini_batches:
                X_batch = X_train[:, mini_batch]
                y_batch = y_train[mini_batch]
                C_batch = C_train[mini_batch, :]

                # Forward pass
                predictions = self.forward(X_batch)

                # Compute loss
                loss = loss_function.cross_entropy_loss(predictions, C_batch)
                epoch_loss += loss

                # Backward pass
                self.backward(predictions, C_batch)

            # Average loss for the epoch
            epoch_loss /= len(mini_batches)

            # Compute accuracy for the entire training set
            train_predictions = self.forward(X_train)
            train_predicted_classes = loss_function.softmax_predictions(train_predictions)
            train_accuracy = np.mean(train_predicted_classes == y_train)

            metrics['train_losses'].append(epoch_loss)
            metrics['train_accuracies'].append(train_accuracy)

            # Validation loss and accuracy
            if X_val is not None and y_val is not None and C_val is not None:
                val_predictions = self.forward(X_val)
                val_loss = loss_function.cross_entropy_loss(val_predictions, C_val)
                val_predicted_classes = loss_function.softmax_predictions(val_predictions)
                val_accuracy = np.mean(val_predicted_classes == y_val)

                metrics['val_losses'].append(val_loss)
                metrics['val_accuracies'].append(val_accuracy)

                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, "
                    f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Plot Training & Validation Loss
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_losses'], label="Training Loss", color="blue", linestyle="-")
        if X_val is not None:
            plt.plot(metrics['val_losses'], label="Validation Loss", color="red", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.grid(True)

        # Plot Training & Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(metrics['train_accuracies'], label="Training Accuracy", color="green", linestyle="-")
        if X_val is not None:
            plt.plot(metrics['val_accuracies'], label="Validation Accuracy", color="purple", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return metrics
    
    def get_params(self):
        # Return all layers parameters as a single vector
        return np.concatenate([layer.get_params() for layer in self.layers])

    def set_params(self, flat_params):
        # Set all layer parameters from a single vector
        current_idx = 0
        for layer in self.layers:
            layer_params = layer.get_params()
            params_size = layer_params.size
            layer.set_params(flat_params[current_idx:current_idx + params_size])
            current_idx += params_size
    
    
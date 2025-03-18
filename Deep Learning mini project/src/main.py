# FILE: /my-python-project/my-python-project/src/main.py
import matplotlib.pyplot as plt
from loss import Loss
from optimizer import Optimizer
import numpy as np
from Data import Data
from NeuralNetwork import NeuralNetwork
from ResNet import ResNet
from tests import jacobian_test_network, gradient_test_network, grad_test


def least_example():
    # Generate synthetic data
    np.random.seed(42)
    
    # Features, Samples
    n, m = 3, 100  
    
    X = np.random.rand(n, m)
    true_W = np.array([[1.0], [2.0], [3.0]])
    true_b = np.array([[0.5]])  
    y = X.T @ true_W + true_b.T  

    # Reshape y to (m,1) for compatibility
    y = y.reshape(-1,1)

    # Initialize weights and bias
    initial_W = np.random.randn(n, 1)
    initial_b = np.random.randn(1, 1)

    # define optimizer
    optimizer = Optimizer(0.1, 100)
    
    updated_W, updated_b, losses, success_percentages, val_losses, val_success_percentages = optimizer.SGD(
        X=X,
        y=y,
        C=np.empty((0, m)),  # C not used in least squares
        W=initial_W,
        b=initial_b,
        loss = Loss(),
        epochs=200
    )

    print("True Weights:", true_W.flatten())
    print("Learned Weights:", updated_W.flatten())
    print("True Bias:", true_b.flatten())
    print("Learned Bias:", updated_b.flatten())

def find_best_hyperparameters_sgd(Data, epochs,plot=True, learning_rates=None, batch_sizes=None):
    """
    Find the best hyperparameters for a given dataset.
    
    Parameters:
    Data (Data): Data object.
    epochs (int): Number of training epochs.
    learning_rates (list, optional): List of learning rates to test.
    batch_sizes (list, optional): List of batch sizes to test.
    
    Returns:
    tuple: Best hyperparameters (learning rate, batch size, training losses, training success percentages, validation losses, validation success percentages).
    """
    if learning_rates is None:
        learning_rates = [0.001, 0.01, 0.1, 0.5]
    if batch_sizes is None:
        batch_sizes = [32, 64, 128, 200, 256] 
    
    best_accuracy = 0
    best_lr = None
    best_batch_size = None
    
    results = []
    
    for lr in learning_rates:
        for bs in batch_sizes:
            optimizer = Optimizer(lr, bs)
            print(f"Testing hyperparameters: Learning Rate = {lr}, Batch Size = {bs}")
            W, b, train_losses, train_success_percentages, val_losses, val_success_percentages = optimizer.SGD(
                X=Data.train_data,
                y=Data.train_labels,
                C=Data.train_indicators,
                W=Data.weights_train,
                b=Data.bias_train,
                loss=Loss(),
                epochs=epochs,
                X_val=Data.test_data,
                y_val=Data.test_labels,
                C_val=Data.test_indicators,
                plot=False
            )
            
            if val_success_percentages[-1] > best_accuracy:
                best_accuracy = val_success_percentages[-1]
                best_lr = lr
                best_batch_size = bs

            result = {
                "learning_rate": lr,
                "batch_size": bs,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accuracies": train_success_percentages,
                "val_accuracies": val_success_percentages,
            }
            results.append(result)
    
    print(f"Best hyperparameters: Learning Rate = {best_lr}, Batch Size = {best_batch_size}")
    
    if plot:
        for result in results:
                if result["learning_rate"] == best_lr and result["batch_size"] == best_batch_size:
                    
                    best_train_losses = result["train_losses"]
                    best_train_success_percentages = result["train_accuracies"]
                    best_val_losses = result["val_losses"]
                    best_val_success_percentages = result["val_accuracies"]
                    
                    
                    optimizer.plot_sgd_results(best_train_losses, best_train_success_percentages, "Training Loss and Success Percentage")
                    optimizer.plot_sgd_results(best_val_losses, best_val_success_percentages, "Validation Loss and Success Percentage")
        
        # Plot the bar graph for final validation accuracy
        labels = [f"lr={res['learning_rate']}, bs={res['batch_size']}" for res in results]
        final_val_accuracies = [res['val_accuracies'][-1] for res in results]
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, final_val_accuracies)
        plt.xlabel("Learning Rate and Batch Size")
        plt.ylabel("Final Validation Accuracy")
        plt.title("Final Validation Accuracy for Different Hyperparameters")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        
                 
    best_hyperparameters = (best_lr, best_batch_size)
    return best_hyperparameters, best_train_losses, best_train_success_percentages, best_val_losses, best_val_success_percentages
    

def main():
    
    # least_example()
    
    # Load data
    Swiss_Roll = Data("SwissRollData.mat", "Swiss Roll")
    Peaks = Data("PeaksData.mat", "Peaks")
    GMM = Data("GMMData.mat", "GMM")
    
    # find_best_hyperparameters_sgd(Peaks, epochs=50)
    
    # grad test for the classifier
    # grad_test(GMM)
    
    # optimizer = Optimizer(0.01, 200)
    # W, b, losses, success_percentages, val_losses, val_success_percentages = optimizer.SGD(
    #     X=Peaks.train_data,
    #     y=Peaks.train_labels,
    #     C=Peaks.train_indicators,
    #     W=Peaks.weights_train,
    #     b=Peaks.bias_train,
    #     loss=Loss(),
    #     epochs=200,
    #     X_val=Peaks.test_data,
    #     y_val=Peaks.test_labels,
    #     C_val=Peaks.test_indicators,
    #     plot=True
    # )
    
    
    # optimizer.SGD(
    #     X=GMM.train_data,
    #     y=GMM.train_labels,
    #     C=GMM.train_indicators,
    #     W=GMM.weights_train,
    #     b=GMM.bias_train,
    #     loss=Loss(),
    #     epochs=200,
    #     X_val=GMM.test_data,
    #     y_val=GMM.test_labels,
    #     C_val=GMM.test_indicators,
    #     plot=True
    # )
    
    
    
    # Find the best hyperparameters for the Swiss Roll dataset
    # find_best_hyperparameters_sgd(Peaks, epochs=50)
    
    
    # model = NeuralNetwork([(2, 10, 'relu'), (10, 2, 'softmax')], 0.01, 32)
    # model = NeuralNetwork([(2, 10, 'relu'), (10,10,'relu'), (10, 2, 'softmax')], 0.01, 32)
    # model = ResNet([(2, 64, 'relu'), (2,64,'relu'),(2, 2, 'softmax')], 0.01, 32)
    # model = ResNet([(2, 10, 'tanh'), (2,10,'tanh'),(2, 2, 'softmax')], 0.01, 32)
    # model = NeuralNetwork([(2, 10, 'tanh'), (10,10,'tanh'), (10, 2, 'softmax')], 0.01, 32)
    # model = NeuralNetwork([(2, 2, 'softmax')], 0.01, 32)
    # model = NeuralNetwork([(2, 2, 'tanh'),(2, 2, 'softmax')], 0.01, 32)
    # model1 = NeuralNetwork([((2, 2, 'softmax'))], 0.01, 64)
    # model2 = NeuralNetwork([(2, 10, 'tanh'), (10, 10, 'tanh'), (10, 2, 'softmax')], 0.01, 64)
    # model3 = NeuralNetwork([(2, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 2, 'softmax')], 0.01, 64)
    # model4 = NeuralNetwork([(2, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'relu'), (10, 2, 'softmax')], 0.01, 64)
    # model5 = NeuralNetwork([(2, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 10, 'tanh'), (10, 2, 'softmax')], 0.01, 64)
    model1 = NeuralNetwork([(2, 25, 'tanh'), (25, 2, 'softmax')], 0.01, 64)
    model2 = NeuralNetwork([(2, 10, 'tanh'), (10, 12, 'tanh'), (12, 7, 'tanh'), (7, 2, 'softmax')], 0.01, 64)
    model3 = NeuralNetwork([(2, 7, 'tanh'), (7, 8, 'tanh'), (8, 8, 'tanh'), (8, 6, 'tanh'), (6, 5, 'tanh'), (5, 2, 'softmax')], 0.01, 64)
    model4 = NeuralNetwork([(2, 6, 'tanh'), (6, 6, 'tanh'), (6, 5, 'tanh'), (5, 5, 'tanh'), (5, 5, 'tanh'), (5, 4, 'tanh'), (4, 3, 'tanh'), (3, 2, 'softmax')], 0.01, 64)
    model5 = NeuralNetwork([(2, 5, 'tanh'), (5, 5, 'tanh'), (5, 4, 'tanh'), (4, 4, 'tanh'), (4, 3, 'tanh'), (3, 3, 'tanh'), (3, 3, 'tanh'), (3, 3, 'tanh'), (3, 3, 'tanh'), (3, 2, 'softmax')], 0.01, 64)
    # model.train(X_train=Swiss_Roll.train_data, 
    #             y_train=Swiss_Roll.train_labels, 
    #             C_train=Swiss_Roll.train_indicators, 
    #             epochs=200, 
    #             batch_size=32, 
    #             X_val=Swiss_Roll.test_data, 
    #             y_val=Swiss_Roll.test_labels, 
    #             C_val=Swiss_Roll.test_indicators)
    model2.train(X_train=Swiss_Roll.train_data, 
                y_train=Swiss_Roll.train_labels, 
                C_train=Swiss_Roll.train_indicators, 
                epochs=200, 
                batch_size=64, 
                X_val=Swiss_Roll.test_data, 
                y_val=Swiss_Roll.test_labels, 
                C_val=Swiss_Roll.test_indicators)
    
    # Jacobian test
    # jacobian_test_network(model, Swiss_Roll.train_data, 10)
    
    # gradient_test_network(model, Swiss_Roll.train_data, Swiss_Roll.train_indicators, 10)
    
    

if __name__ == "__main__":
    main()
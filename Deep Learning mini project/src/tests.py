import numpy as np
import matplotlib.pyplot as plt
from loss import Loss
from layer import NNLayer, ResNetLayer

#gradient test
def grad_test1(train_data, indicators, W, data_name):
    """
    Perform a gradient test to validate the implementation of gradients for a softmax function.

    Parameters:
    train_data (numpy.ndarray): Input data, shape (n+1, m) where n is the number of features and m is the number of samples.
    train_labels (numpy.ndarray): One-hot encoded true labels, shape (l, m) where l is the number of classes and m is the number of samples.
    indicators (numpy.ndarray): Additional indicator matrix (not used in this function but kept for consistency).
    W (numpy.ndarray): Weights matrix, shape (n+1, l), where n+1 is the number of features (including bias) 
                       and l is the number of classes.
    data_name (str): Name of the dataset being tested, used for labeling the output.

    This function compares the zero-order and first-order approximations of a function's value
    using gradients to validate the correctness of gradient calculations.
    """

    # Randomly initialize weights (n + 1 to include bias term) for softmax

    # Compute the initial softmax loss and gradient
    F0 = cross_entropy_loss(train_data, indicators, W)
    G0 = softmax_gradient(train_data, indicators, W)

    # Randomly initialize a direction matrix D (same shape as W)
    D = np.random.rand(W.shape[0], W.shape[1])

    # Set initial epsilon value for the test
    epsilon = 0.1

    # Arrays to store zero-order and first-order approximations for comparison
    zero_order = np.zeros(10)  # Zero-order approximation errors
    first_order = np.zeros(10)  # First-order approximation errors

    # Print table header for clarity in output
    print("k\terror order 1 \t\t\terror order 2")

    # Loop over decreasing epsilon values to test the approximation
    for k in range(10):
        # Reduce epsilon geometrically
        epsk = epsilon * (0.5 ** k)

        # Perturbed weights
        W_prime = W + epsk * D

        # Compute softmax loss for perturbed weights
        Fk = cross_entropy_loss(train_data, indicators, W_prime)

        # First-order approximation of Fk using gradients
        F1 = F0 + epsk * np.sum(G0 * D)

        # Compute the absolute errors for zero-order and first-order approximations
        zero_order[k] = abs(Fk - F0)
        first_order[k] = abs(Fk - F1)

        # Print current k, zero-order error, and first-order error
        print(f"{k+1}\t{zero_order[k]}\t\t{first_order[k]}")

    # Plot the errors in a semilogarithmic plot
    plt.semilogy(range(1, 11), zero_order, label="Zero order approx")
    plt.semilogy(range(1, 11), first_order, label="First order approx")
    plt.legend()
    plt.title("Successful Grad Test in Semilogarithmic Plot- " + data_name)
    plt.xlabel("k (iterations with decreasing epsilon)")
    plt.ylabel("Error")
    plt.show()
    
 
def grad_test(Data):
    """
    Perform a gradient test to validate the implementation of gradients for a softmax function.

    Parameters:
    Data (object): Data object containing train_data (X), train_labels, train_indicators (C), weights_train, and bias_train.
    data_name (str): Name of the dataset being tested, used for labeling the output.

    This function compares the zero-order and first-order approximations of a function's value
    using gradients to validate the correctness of gradient calculations.
    """
    # Extract data from the Data object
    X = Data.train_data
    C = Data.train_indicators
    W = Data.weights_train
    b = Data.bias_train
    data_name = Data.name

    # Initialize the loss object
    loss = Loss()

    # Compute the initial softmax loss and gradient
    prediction = loss.softmax(X, W, b)  
    F0 = loss.cross_entropy_loss(prediction, C)
    G0_W, G0_b, G0_X = loss.cross_entropy_gradient(prediction, C, X, W)

    # Randomly initialize direction matrices D_W (same shape as W), d_b (same shape as b), and d_X (same shape as X)
    D_W = np.random.rand(W.shape[0], W.shape[1])
    d_b = np.random.rand(b.shape[0], b.shape[1])
    d_X = np.random.rand(X.shape[0], X.shape[1])

    # Set initial epsilon value for the test
    epsilon = 0.1

    # Arrays to store zero-order and first-order approximations for comparison
    zero_order_W = np.zeros(10)  # Zero-order approximation errors for W
    first_order_W = np.zeros(10)  # First-order approximation errors for W
    zero_order_b = np.zeros(10)  # Zero-order approximation errors for b
    first_order_b = np.zeros(10)  # First-order approximation errors for b
    zero_order_X = np.zeros(10)  # Zero-order approximation errors for X
    first_order_X = np.zeros(10)  # First-order approximation errors for X

    # Print table header for clarity in output
    # print("k\terror order 1 (W)\terror order 2 (W)\terror order 1 (b)\terror order 2 (b)\terror order 1 (X)\terror order 2 (X)")

    # Loop over decreasing epsilon values to test the approximation
    for k in range(10):
        # Reduce epsilon geometrically
        epsk = epsilon * (0.5 ** k)

        # Perturbed weights, biases, and input data
        W_prime = W + epsk * D_W
        b_prime = b + epsk * d_b
        X_prime = X + epsk * d_X

        # Compute softmax loss for perturbed weights, biases, and input data
        prediction_W = loss.softmax(X, W_prime, b)
        prediction_b = loss.softmax(X, W, b_prime)
        prediction_X = loss.softmax(X_prime, W, b)
        
        Fk_W = loss.cross_entropy_loss(prediction_W, C)
        Fk_b = loss.cross_entropy_loss(prediction_b, C)
        Fk_X = loss.cross_entropy_loss(prediction_X, C)

        # First-order approximation of Fk using gradients
        F1_W = F0 + epsk * np.sum(G0_W * D_W)
        F1_b = F0 + epsk * np.sum(G0_b * d_b)
        F1_X = F0 + epsk * np.sum(G0_X * d_X)

        # Compute the absolute errors for zero-order and first-order approximations
        zero_order_W[k] = abs(Fk_W - F0)
        first_order_W[k] = abs(Fk_W - F1_W)
        zero_order_b[k] = abs(Fk_b - F0)
        first_order_b[k] = abs(Fk_b - F1_b)
        zero_order_X[k] = abs(Fk_X - F0)
        first_order_X[k] = abs(Fk_X - F1_X)

        # Print current k, zero-order error, and first-order error for W, b, and X
        # print(f"{k+1}\t{zero_order_W[k]}\t{first_order_W[k]}\t{zero_order_b[k]}\t{first_order_b[k]}\t{zero_order_X[k]}\t{first_order_X[k]}")

    # Plot the errors in a semilogarithmic plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.semilogy(range(1, 11), zero_order_W, label="Zero order approx (W)")
    plt.semilogy(range(1, 11), first_order_W, label="First order approx (W)")
    plt.legend()
    plt.title(f"Gradient Test for Weights (W) - {data_name}")
    plt.xlabel("k")
    plt.xticks(range(1, 11))
    plt.ylabel("Error")

    plt.subplot(1, 3, 2)
    plt.semilogy(range(1, 11), zero_order_b, label="Zero order approx (b)")
    plt.semilogy(range(1, 11), first_order_b, label="First order approx (b)")
    plt.legend()
    plt.title(f"Gradient Test for Biases (b) - {data_name}")
    plt.xlabel("k")
    plt.xticks(range(1, 11))
    plt.ylabel("Error")

    plt.subplot(1, 3, 3)
    plt.semilogy(range(1, 11), zero_order_X, label="Zero order approx (X)")
    plt.semilogy(range(1, 11), first_order_X, label="First order approx (X)")
    plt.legend()
    plt.title(f"Gradient Test for Input Data (X) - {data_name}")
    plt.xlabel("k")
    plt.xticks(range(1, 11))
    plt.ylabel("Error")

    plt.suptitle(f"Gradient Test Results - {data_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
 
 
    
def jacobian_test_network(model, X, sample_num=1, plot=True):
    
    model.forward(X)
    layers_num = model.num_layers
    
    # Loop over all layers - except the softmax layer
    for i in range(layers_num-1):
        curr_layer = model.layers[i]
        
        v = np.random.rand(curr_layer.output.shape[0], curr_layer.input.shape[1])
        v /= np.linalg.norm(v) if np.linalg.norm(v) != 0 else 1
        X = curr_layer.input.astype(np.float64)  # Avoid overflow
        
        base_forward = np.vdot(v, curr_layer.output)
                
        if isinstance(curr_layer, ResNetLayer): 
            grad_x, grad_w1, grad_w2, grad_b = curr_layer.backward(v)
            
            # Testing W1
            zero_order_w1, first_order_w1 = test_gradients(curr_layer.W1, grad_w1, sample_num=sample_num, 
                                                           X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing W2
            zero_order_w2, first_order_w2 = test_gradients(curr_layer.W2, grad_w2, sample_num=sample_num, 
                                                           X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing b
            zero_order_b, first_order_b = test_gradients(curr_layer.b, grad_b, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing X
            zero_order_x, first_order_x = test_gradients(curr_layer.input, grad_x, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            
            if plot:
                gradients = [
                    ("W1", zero_order_w1, first_order_w1),
                    ("W2", zero_order_w2, first_order_w2),
                    ("b", zero_order_b, first_order_b),
                    ("X", zero_order_x, first_order_x)
                ]
                plot_gradients(gradients, "ResNetLayer", i+1)
        else:   
            grad_x, grad_w, grad_b = curr_layer.backward(v)
             
            # Testing W
            zero_order_w, first_order_w = test_gradients(curr_layer.Weights, grad_w, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing b
            zero_order_b, first_order_b = test_gradients(curr_layer.Bias, grad_b, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            # Testing X
            zero_order_x, first_order_x = test_gradients(curr_layer.input, grad_x, sample_num=sample_num, 
                                                         X=X, v=v, curr_layer=curr_layer, base_forward=base_forward)
            
            if plot:
                gradients = [
                    ("W", zero_order_w, first_order_w),
                    ("b", zero_order_b, first_order_b),
                    ("X", zero_order_x, first_order_x)
                ]
                plot_gradients(gradients, "NNLayer", i+1)
            
def test_gradients(parameter, grad_param, sample_num, X, v, curr_layer, base_forward):
    
    epsilon_iterator = [0.5 ** i for i in range(1, 11)]
    
    # Initialize accumulators for differences
    zero_order = np.zeros(len(epsilon_iterator))
    first_order = np.zeros(len(epsilon_iterator))
                    
    for i in range(sample_num):
        # Generate a random perturbation
        perturbations = np.random.randn(*parameter.shape)
        perturbations /= np.linalg.norm(perturbations) if np.linalg.norm(perturbations) != 0 else 1
                        
        original_param = parameter.copy()
                        
        for idx, eps in enumerate(epsilon_iterator):
            
            # Perturb the parameter
            parameter += perturbations * eps
            # Forward pass after perturbation
            forward_after_eps = np.vdot(v, curr_layer.forward(X))
            # Revert the parameter to original
            parameter[:] = original_param
                            
            # Compute differences
            diff = np.abs(forward_after_eps - base_forward)
            grad_diff = np.abs(forward_after_eps - base_forward - np.vdot(grad_param, perturbations * eps))
                            
            # Accumulate differences
            zero_order[idx] += diff
            first_order[idx] += grad_diff
                    
    # Compute average over samples
    avg_zero_order = zero_order / sample_num
    avg_first_order = first_order / sample_num
                    
    return avg_zero_order, avg_first_order

def plot_gradients(gradients, layer_type, layer_num):
    epsilon_iterator = [0.5 ** i for i in range(1, 11)]
    x_labels = list(range(1, 11))  # Labels from 1 to 10
    
    num_plots = len(gradients)
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    for idx, (param_name, avg_grad_diffs, avg_grad_diffs_grad) in enumerate(gradients):
        axs[idx].plot(x_labels, avg_grad_diffs, label=f"Zero-order approximation ({param_name})")
        axs[idx].plot(x_labels, avg_grad_diffs_grad, label=f"First-order approximation ({param_name})")
        axs[idx].set_yscale('log')
        axs[idx].set_ylabel('Error')
        axs[idx].set_xlabel('k')
        axs[idx].set_title(f'Error vs. k for {param_name}')
        axs[idx].legend()
        axs[idx].set_aspect('auto')
        axs[idx].set_xticks(x_labels)
    
    plt.suptitle(f'Gradient Test for {layer_type}, Layer: {layer_num}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0)
    plt.show()


  
def gradient_test_network(model, X, C, num_samples=1):
    
    epsilon_iterator=[0.5 ** i for i in range(1, 11)]
    is_resNet = False
    
    # Initialize arrays to store errors
    zero_order_errors = np.zeros(len(epsilon_iterator))
    first_order_errors = np.zeros(len(epsilon_iterator))

    prediction = model.forward(X)                   # Forward pass
    F0 = Loss().cross_entropy_loss(prediction, C)   # Loss without perturbation
    grads = model.backprop(prediction, C)           # Backward pass (compute gradients) - no update
    
    # Flatten gradients
    flat_grads = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, ResNetLayer):
            is_resNet = True
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_W1', np.zeros_like(layer.W1)).flatten())
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_W2', np.zeros_like(layer.W2)).flatten())
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_b', np.zeros_like(layer.b)).flatten())
        else:
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_W', np.zeros_like(layer.Weights)).flatten())
            flat_grads.append(grads.get(f'layer_{i}', {}).get('grad_b', np.zeros_like(layer.Bias)).flatten())
    flat_grads = np.concatenate(flat_grads)

    original_params = model.get_params().copy()
    
    for idx in range(num_samples):
        perturbations = np.random.randn(len(flat_grads))
        perturbations /= np.linalg.norm(perturbations) if np.linalg.norm(perturbations) != 0 else 1.0
        
        for k, eps in enumerate(epsilon_iterator):
            flat_params_plus = original_params + eps * perturbations
            
            model.set_params(flat_params_plus)
            
            prediction_perturbation = model.forward(X)
            F_plus = Loss().cross_entropy_loss(prediction_perturbation, C) # Loss with perturbation
            model.set_params(original_params)
            
            zero_order_error = np.abs(F_plus - F0)
            first_order_error = np.abs(F_plus - F0 - np.vdot(flat_grads, eps * perturbations))

            zero_order_errors[k] += zero_order_error / num_samples
            first_order_errors[k] += first_order_error / num_samples
            
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,11), zero_order_errors, label="Zero order error")
    plt.plot(range(1,11), first_order_errors, label="First order error")
    plt.xticks(range(1,11))
    plt.yscale('log')
    plt.xlabel("k")
    plt.ylabel("Error")
    if is_resNet:
        plt.title("Gradient test - Residual Network")
    else:
        plt.title("Gradient test - Neural Network")
    plt.legend()
    plt.show()

    return zero_order_errors, first_order_errors
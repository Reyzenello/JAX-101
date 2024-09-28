# JAX-101

Testing out JAX:

Requirements of the module needs to be installed:
pip install jax jaxlib
python testing-out-JAX.py

![image](https://github.com/Reyzenello/JAX-101/assets/43668563/4e6be246-37e5-459e-8ae2-471275bca57b)


This code implements a simple feedforward neural network using JAX, a high-performance numerical computing library optimized for machine learning tasks. The network is trained to approximate a linear function 
𝑦
=
2
𝑥
+
1
y=2x+1 based on generated data. Below is a step-by-step explanation of the code:

Imports and Setup
python
Copy code
import jax
import jax.numpy as jnp
from jax import grad, jit, random
jax: The main JAX library for high-performance numerical computing.
jax.numpy as jnp: A NumPy-compatible API provided by JAX for array operations.
grad: Automatic differentiation function to compute gradients.
jit: Just-In-Time compilation decorator to optimize functions.
random: Module for random number generation in JAX.
Parameter Initialization
python
Copy code
def init_params(layer_sizes, key):
    keys = random.split(key, len(layer_sizes))
    return [(random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), jnp.zeros(n))
            for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
Purpose: Initializes the weights and biases for each layer of the neural network.
Parameters:
layer_sizes: A list specifying the number of neurons in each layer (e.g., [1, 10, 1]).
key: A PRNGKey for random number generation.
Process:
random.split(key, len(layer_sizes)): Splits the key into multiple subkeys for reproducibility.
Weight Initialization:
random.normal(k, (m, n)): Generates a matrix of random values with a normal distribution for weights.
* jnp.sqrt(2.0 / m): Scales the weights using He initialization to maintain signal variance through layers.
Bias Initialization:
jnp.zeros(n): Initializes biases to zero for each neuron in the layer.
Neural Network Prediction Function
python
Copy code
def predict(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, w) + b)
    final_w, final_b = params[-1]
    return jnp.dot(x, final_w) + final_b
Purpose: Performs a forward pass through the neural network to make predictions.
Process:
Hidden Layers:
Loops through each pair of weights (w) and biases (b) except for the last layer.
jnp.dot(x, w) + b: Computes the affine transformation.
jnp.tanh(...): Applies the hyperbolic tangent activation function.
Output Layer:
Retrieves the weights and biases for the final layer.
jnp.dot(x, final_w) + final_b: Computes the output without an activation function (suitable for regression tasks).
Loss Function
python
Copy code
def loss(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)
Purpose: Calculates the mean squared error (MSE) between the predictions and the true values.
Parameters:
params: Current weights and biases of the network.
x: Input features.
y: True target values.
Process:
predict(params, x): Generates predictions using the current parameters.
jnp.mean((pred - y) ** 2): Computes the average of the squared differences.
Parameter Update Function
python
Copy code
@jit
def update(params, x, y, lr):
    grads = grad(loss)(params, x, y)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
Purpose: Updates the network's parameters using gradient descent.
Decorators:
@jit: Compiles the function for faster execution.
Process:
grad(loss)(params, x, y): Computes gradients of the loss function with respect to each parameter.
Parameter Update:
w - lr * dw: Updates weights by moving against the gradient.
b - lr * db: Updates biases similarly.
Parameters:
params: Current parameters.
x: Input features.
y: True target values.
lr: Learning rate.
Data Generation
python
Copy code
key = random.PRNGKey(0)
x = random.normal(key, (100, 1))
y = 2.0 * x + 1.0 + 0.1 * random.normal(key, (100, 1))
Purpose: Generates synthetic data for training.
Process:
Input Features (x):
random.normal(key, (100, 1)): Generates 100 random samples from a standard normal distribution.
Target Values (y):
2.0 * x + 1.0: Computes the linear relationship.
+ 0.1 * random.normal(key, (100, 1)): Adds Gaussian noise to simulate real-world data.
Parameter Initialization for the Network
python
Copy code
layer_sizes = [1, 10, 1]
params = init_params(layer_sizes, key)
Purpose: Sets up the neural network architecture and initializes its parameters.
Parameters:
layer_sizes: Defines a network with:
Input Layer: 1 neuron.
Hidden Layer: 10 neurons.
Output Layer: 1 neuron.
Process:
init_params(layer_sizes, key): Calls the initialization function with the specified architecture.
Training Loop
python
Copy code
lr = 0.01
for i in range(1000):
    params = update(params, x, y, lr)
    if i % 100 == 0:
        current_loss = loss(params, x, y)
        print(f"Step {i}, Loss: {current_loss}")
Purpose: Trains the neural network using gradient descent.
Parameters:
lr: Learning rate set to 0.01.
Iterations: Runs for 1,000 epochs.
Process:
Parameter Update:
params = update(params, x, y, lr): Updates the parameters in each iteration.
Monitoring:
Every 100 steps, computes and prints the current loss to monitor training progress.
Testing the Trained Network
python
Copy code
x_test = jnp.array([[1.0], [2.0], [3.0]])
preds = predict(params, x_test)
print(preds)
Purpose: Evaluates the trained network on new data.
Process:
Test Inputs (x_test): An array of values for which we want predictions.
Predictions (preds):
predict(params, x_test): Generates predictions using the trained parameters.
Output:
Prints the predicted values for the test inputs.
Summary
Objective: Train a neural network to learn the relationship 
𝑦
=
2
𝑥
+
1
y=2x+1 using synthetic data.
Architecture:
A simple feedforward network with one hidden layer of 10 neurons.
Training:
Uses mean squared error as the loss function.
Employs gradient descent for optimization.
Results:
The network learns to approximate the linear function.
Predictions on test data should be close to the true values 
𝑦
=
2
𝑥
+
1
y=2x+1.
Key Concepts
JAX: Provides high-performance computations with automatic differentiation and JIT compilation.
He Initialization: Weight initialization technique that maintains variance in the forward pass.
Gradient Descent: Optimization algorithm that updates parameters to minimize the loss function.
Automatic Differentiation: Computes gradients efficiently for complex functions.
Sample Output
During training, you might see output like:

vbnet
Copy code
Step 0, Loss: 5.4321
Step 100, Loss: 0.1234
Step 200, Loss: 0.0456
...
Step 900, Loss: 0.0012
After testing, the predictions might be:

lua
Copy code
[[2.99]
 [4.99]
 [6.99]]
Which are close to the expected values:

For 
𝑥
=
1.0
x=1.0: 
𝑦
=
2
×
1.0
+
1
=
3.0
y=2×1.0+1=3.0
For 
𝑥
=
2.0
x=2.0: 
𝑦
=
2
×
2.0
+
1
=
5.0
y=2×2.0+1=5.0
For 
𝑥
=
3.0
x=3.0: 
𝑦
=
2
×
3.0
+
1
=
7.0
y=2×3.0+1=7.0

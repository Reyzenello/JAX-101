# JAX-101

Testing out JAX:

Requirements of the module needs to be installed:
pip install jax jaxlib
python testing-out-JAX.py

![image](https://github.com/Reyzenello/JAX-101/assets/43668563/4e6be246-37e5-459e-8ae2-471275bca57b)

This code implements a simple neural network for linear regression using JAX, a high-performance numerical computation library in Python. Let's break down the code step by step:

**1. Importing Libraries:**

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, random
```

* `jax`: The core JAX library.
* `jax.numpy as jnp`: JAX's NumPy-like array library.  It's used almost like regular NumPy, but it allows JAX to perform its automatic differentiation and just-in-time compilation magic.
* `grad`:  A JAX function for automatic differentiation (finding gradients).
* `jit`:  The Just-In-Time (JIT) compilation decorator. This significantly speeds up JAX code by compiling it to optimized machine code.
* `random`: JAX's pseudo-random number generator (PRNG). It's important to use JAX's PRNG for reproducibility.

**2. Initializing Parameters (`init_params`):**

```python
def init_params(layer_sizes, key):
    keys = random.split(key, len(layer_sizes))
    return [(random.normal(k, (m, n)) * jnp.sqrt(2.0/m), jnp.zeros(n))
            for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
```

* `layer_sizes`: A list defining the number of neurons in each layer.  `[1, 10, 1]` means an input layer with 1 neuron, a hidden layer with 10 neurons, and an output layer with 1 neuron.
* `key`:  The PRNG key used for generating random numbers.  It's important to split keys for different operations to avoid correlations.
* `random.split(key, len(layer_sizes))`: Splits the initial PRNG key into subkeys, one for each layer.
* `random.normal(k, (m, n)) * jnp.sqrt(2.0/m)`: Initializes the weights (W) with normally distributed random numbers, scaled by `jnp.sqrt(2.0/m)` (Xavier initialization). This scaling can help with training.
* `jnp.zeros(n)`: Initializes the biases (b) to zero.
* The function returns a list of tuples `(w, b)` for each layer.

**3. Defining the Neural Network (`predict`):**

```python
def predict(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, w) + b)
    final_w, final_b = params[-1]
    return jnp.dot(x, final_w) + final_b
```

* `params`: The list of weights and biases from `init_params`.
* `x`: The input data.
* The code loops through each layer except the last one.
* `jnp.tanh(...)`: Applies the hyperbolic tangent activation function element-wise.
* `jnp.dot(x, w) + b`: Performs the linear transformation (matrix multiplication with weights and adding bias).
* For the final layer, a linear transformation (no activation function) is used.

**4. Defining the Loss Function (`loss`):**

```python
def loss(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)
```

* `params`, `x`, `y`: As before.
* `pred`: The model's prediction using the `predict` function.
* `jnp.mean((pred - y) ** 2)`: Calculates the mean squared error (MSE) between predictions and true values.

**5. Defining the Update Function (`update`):**

```python
@jit
def update(params, x, y, lr):
    grads = grad(loss)(params, x, y)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
```

* `@jit`: JIT-compiles the function for speed.
* `grad(loss)(params, x, y)`: Calculates the gradients of the `loss` function with respect to `params`. This is the heart of JAX's automatic differentiation.
* `lr`: The learning rate.
* `(w - lr * dw, b - lr * db)`: Updates the weights and biases using gradient descent.

**6. Generating Data:**

```python
key = random.PRNGKey(0)
x = random.normal(key, (100, 1))
y = 2.0 * x + 1.0 + 0.1 * random.normal(key, (100, 1))
```

* Creates some synthetic linear data with some noise.

**7. Initializing Parameters and Training Loop:**

```python
layer_sizes = [1, 10, 1]
params = init_params(layer_sizes, key)

lr = 0.01
for i in range(1000):
    params = update(params, x, y, lr)
    if i % 100 == 0:
        current_loss = loss(params, x, y)
        print(f"Step {i}, Loss: {current_loss}")
```

* Initializes the network parameters.
* Runs the training loop for 1000 steps, updating the parameters in each step.
* Prints the loss every 100 steps.


**8. Testing the Trained Network:**

```python
x_test = jnp.array([[1.0], [2.0], [3.0]])
preds = predict(params, x_test)
print(preds)
```

* Makes predictions on some test data.


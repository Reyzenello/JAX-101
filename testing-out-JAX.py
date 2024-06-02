import jax
import jax.numpy as jnp
from jax import grad, jit, random

# Initialize parameters
def init_params(layer_sizes, key):
    keys = random.split(key, len(layer_sizes))
    return [(random.normal(k, (m, n)) * jnp.sqrt(2.0/m), jnp.zeros(n))
            for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]

# Define the neural network
def predict(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, w) + b)
    final_w, final_b = params[-1]
    return jnp.dot(x, final_w) + final_b

# Define the loss function
def loss(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

# Define the update function
@jit
def update(params, x, y, lr):
    grads = grad(loss)(params, x, y)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]

# Generate some data
key = random.PRNGKey(0)
x = random.normal(key, (100, 1))
y = 2.0 * x + 1.0 + 0.1 * random.normal(key, (100, 1))

# Initialize parameters
layer_sizes = [1, 10, 1]
params = init_params(layer_sizes, key)

# Training loop
lr = 0.01
for i in range(1000):
    params = update(params, x, y, lr)
    if i % 100 == 0:
        current_loss = loss(params, x, y)
        print(f"Step {i}, Loss: {current_loss}")

# Test the trained network
x_test = jnp.array([[1.0], [2.0], [3.0]])
preds = predict(params, x_test)
print(preds)

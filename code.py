import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate a simple dataset
# y = 2 * x + 3 with some added noise
n_samples = 100
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 2 * X + 3 + np.random.randn(n_samples, 1) * 2  # Add noise

# Add a bias term (ones column) to X for the intercept
X_b = np.c_[np.ones((n_samples, 1)), X]

# Linear model: y = theta_0 + theta_1 * x
def predict(X, theta):
    return X.dot(theta)

# Mean Squared Error loss
def compute_loss(y, y_pred):
    return np.mean((y_pred - y) ** 2)

# Gradient Descent Optimizer
def gradient_descent(X, y, theta, lr, epochs):
    m = len(y)
    loss_history = []
    for epoch in range(epochs):
        gradients = 2 / m * X.T.dot(predict(X, theta) - y)
        theta -= lr * gradients
        loss = compute_loss(y, predict(X, theta))
        loss_history.append(loss)
    return theta, loss_history

# Stochastic Gradient Descent Optimizer
def stochastic_gradient_descent(X, y, theta, lr, epochs):
    m = len(y)
    loss_history = []
    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index + 1]
            yi = y[rand_index:rand_index + 1]
            gradients = 2 * xi.T.dot(predict(xi, theta) - yi)
            theta -= lr * gradients
        loss = compute_loss(y, predict(X, theta))
        loss_history.append(loss)
    return theta, loss_history

# SGD with Momentum Optimizer
def sgd_with_momentum(X, y, theta, lr, epochs, momentum=0.9):
    m = len(y)
    velocity = np.zeros(theta.shape)
    loss_history = []
    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index + 1]
            yi = y[rand_index:rand_index + 1]
            gradients = 2 * xi.T.dot(predict(xi, theta) - yi)
            velocity = momentum * velocity - lr * gradients
            theta += velocity
        loss = compute_loss(y, predict(X, theta))
        loss_history.append(loss)
    return theta, loss_history

# Adam Optimizer
def adam(X, y, theta, lr, epochs, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = len(y)
    mt = np.zeros(theta.shape)  # First moment
    vt = np.zeros(theta.shape)  # Second moment
    loss_history = []
    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index + 1]
            yi = y[rand_index:rand_index + 1]
            gradients = 2 * xi.T.dot(predict(xi, theta) - yi)
            mt = beta1 * mt + (1 - beta1) * gradients
            vt = beta2 * vt + (1 - beta2) * (gradients ** 2)
            mt_hat = mt / (1 - beta1 ** (epoch + 1))
            vt_hat = vt / (1 - beta2 ** (epoch + 1))
            theta -= lr * mt_hat / (np.sqrt(vt_hat) + epsilon)
        loss = compute_loss(y, predict(X, theta))
        loss_history.append(loss)
    return theta, loss_history


#Nesterov Accelerated Gradient (NAG)
def nag(X, y, theta, lr, epochs, momentum=0.9):
    m = len(y)
    velocity = np.zeros(theta.shape)
    loss_history = []
    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index + 1]
            yi = y[rand_index:rand_index + 1]
            theta_lookahead = theta + momentum * velocity
            gradients = 2 * xi.T.dot(predict(xi, theta_lookahead) - yi)
            velocity = momentum * velocity - lr * gradients
            theta += velocity
        loss = compute_loss(y, predict(X, theta))
        loss_history.append(loss)
    return theta, loss_history


# Initialize model parameters
theta_initial = np.random.randn(2, 1)
epochs = 300
lr = 0.01

# Train models with different optimizers
theta_gd, loss_gd = gradient_descent(X_b.copy(), y, theta_initial.copy(), lr, epochs)
theta_sgd, loss_sgd = stochastic_gradient_descent(X_b.copy(), y, theta_initial.copy(), lr, epochs)
theta_sgd_momentum, loss_sgd_momentum = sgd_with_momentum(X_b.copy(), y, theta_initial.copy(), lr, epochs)
theta_adam, loss_adam = adam(X_b.copy(), y, theta_initial.copy(), lr, epochs)
theta_nag, loss_nag = nag(X_b.copy(), y, theta_initial.copy(), lr, epochs)

# Create subplots
fig, axs = plt.subplots(1, 5, figsize=(20, 5))  # 1 row, 4 columns

# Gradient Descent
axs[0].plot(loss_gd, label="Gradient Descent")
axs[0].set_title("Gradient Descent")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")

# SGD
axs[1].plot(loss_sgd, label="SGD")
axs[1].set_title("SGD")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss")

# SGD with Momentum
axs[2].plot(loss_sgd_momentum, label="SGD with Momentum")
axs[2].set_title("SGD with Momentum")
axs[2].set_xlabel("Epochs")
axs[2].set_ylabel("Loss")

# Adam
axs[3].plot(loss_adam, label="Adam")
axs[3].set_title("Adam")
axs[3].set_xlabel("Epochs")
axs[3].set_ylabel("Loss")


# Nag
axs[4].plot(loss_nag, label="Nesterov Accelerated Gradient")
axs[4].set_title("Nesterov Accelerated Gradient")
axs[4].set_xlabel("Epochs")
axs[4].set_ylabel("Loss")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

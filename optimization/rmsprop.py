import numpy as np


def RMSProp(X, w, t, grad, v, epoch, beta=0.9, eta=0.1, epsilon=0.01):
    """Computes RMSProp updates on the weight."""
    for i in range(epoch):
        # momentum update
        v = beta * (v) + (1 - beta) * np.square(grad(X, w, t))
        # weight update
        w = w - (eta / (np.sqrt(v) + epsilon)) * grad(X, w, t)
        
        if i <= 10 or i % 10 == 0: 
            print(f"iter {i} weight: {w}")
    
    return w


def grad(X, w, t):
    """Computes the gradient of MSE loss w.r.t. weight, w"""
    return 2 * X.T * (np.dot(X, w) - t)


X = np.array([2., 1.])
w = np.array([0., 0.])
v = np.array([0.1, 0.1])
t = np.array([2])

RMSProp(X, w, t, grad, v, 200)



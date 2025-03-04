import numpy as np

def armijo_line_search(f, grad_f, x_k, d_k, alpha=1, rho=0.5, c=1e-4):
    """
    Armijo backtracking line search.
    
    Parameters:
        f (function): Objective function.
        grad_f (function): Gradient of the objective function.
        x_k (numpy array): Current point.
        d_k (numpy array): Search direction.
        alpha (float): Initial step size.
        rho (float): Reduction factor for step size (default: 0.5).
        c (float): Armijo constant (default: 1e-4).
    
    Returns:
        alpha (float): Optimal step size satisfying Armijo condition.
    """
    while f(x_k + alpha * d_k) > f(x_k) + c * alpha * np.dot(grad_f(x_k), d_k):
        alpha *= rho  # Reduce step size

    return alpha

# Example Usage
def f(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2  # A simple quadratic function

def grad_f(x):
    return np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])  # Gradient

x_k = np.array([0.0, 0.0])  # Initial point
d_k = -grad_f(x_k)  # Steepest descent direction

optimal_alpha = armijo_line_search(f, grad_f, x_k, d_k)

print("Optimal step size:", optimal_alpha)

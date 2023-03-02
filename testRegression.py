import numpy as np

# Generate some random data with a linear trend
x = np.linspace(0, 10, 20)
y = 2 * x + 1 + np.random.randn(20)

# Stack the arrays to create the design matrix
A = np.vstack([x, np.ones(len(x))]).T

# Solve the least-squares problem
result = np.linalg.lstsq(A, y, rcond=None)

# Get the optimal values of the unknowns
m, c = result[0]

# Print the results
print(f"The estimated line is y = {m:.2f}x + {c:.2f}")
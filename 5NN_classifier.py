# Re-import necessary packages due to code environment reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Redefine the dataset points and labels
points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 0.5],   # Class A
                   [0.5, 1.5], [1.5, 0.5], [2.5, 1.5]])  # Class B
labels = np.array([0, 0, 0, 1, 1, 1])  # 0 = Class A, 1 = Class B

# Create a 5-NN classifier and fit it with the training points
knn_5nn = KNeighborsClassifier(n_neighbors=5)
knn_5nn.fit(points, labels)

# Create a mesh grid over the 2D space (0 ≤ x ≤ 3, 0 ≤ y ≤ 2)
x_min, x_max = 0, 3
y_min, y_max = 0, 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 200))

# Predict the class for each point on the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
predictions_5nn = knn_5nn.predict(grid_points).reshape(xx.shape)

# Plotting the results for 5-NN
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the decision boundary (regions classified as Class B)
ax.contourf(xx, yy, predictions_5nn, levels=[0.5, 1.5], colors=['lightblue'], alpha=0.5)

# Plot the Class A points
ax.scatter(points[:3, 0], points[:3, 1], color='red', marker='o', label='Class A')

# Plot the Class B points
ax.scatter(points[3:, 0], points[3:, 1], color='blue', marker='x', label='Class B')

# Formatting the plot
ax.set_xlim(0, 3)
ax.set_ylim(0, 2)
ax.set_xticks(np.arange(0, 4, 0.5))
ax.set_yticks(np.arange(0, 3, 0.5))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
ax.set_title('5-Nearest Neighbor Classification Regions')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()

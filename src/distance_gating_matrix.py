
import numpy as np

# Three predicted object positions
predicted_positions = np.array([
    [5.0, 2.0, 1.0],
    [0.0, 5.0, 0.0],
    [2.0, 3.0, 5.0]
])

# Three detected positions
detections = np.array([
    [5.2, 2.1, 1.0],
    [0.2, 5.1, 0.1],
    [50.0, 40.0, 30.0]
])

max_distance = 2.0

# Calculate the distance matrix
distance_matrix = np.linalg.norm(
    predicted_positions[:, np.newaxis, :] -
    detections[np.newaxis, :, :],
    axis=2
)

print("Original Distance Matrix:")
print(np.round(distance_matrix, 3))

# Apply distance gating
gated_matrix = distance_matrix.copy()

gated_matrix[gated_matrix > max_distance] = np.inf

print("\nGated Distance Matrix:")
print(gated_matrix)

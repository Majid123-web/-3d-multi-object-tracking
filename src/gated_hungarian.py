
import numpy as np
from scipy.optimize import linear_sum_assignment

# Predicted positions
predicted_positions = np.array([
    [5.0, 2.0, 1.0],
    [0.0, 5.0, 0.0],
    [2.0, 3.0, 5.0]
])

# Current detections
detections = np.array([
    [5.2, 2.1, 1.0],
    [0.2, 5.1, 0.1],
    [50.0, 40.0, 30.0]
])

max_distance = 2.0

# Calculate distance matrix
distance_matrix = np.linalg.norm(
    predicted_positions[:, np.newaxis, :] -
    detections[np.newaxis, :, :],
    axis=2
)

# Replace invalid distances with a large penalty
cost_matrix = distance_matrix.copy()
cost_matrix[cost_matrix > max_distance] = 1e6

# Hungarian assignment
row_indices, col_indices = linear_sum_assignment(cost_matrix)

print("Gated Hungarian Association\n")

# Accept only valid assignments
for track_index, detection_index in zip(row_indices, col_indices):

    distance = distance_matrix[track_index, detection_index]

    if distance <= max_distance:

        print(
            f"Track {track_index + 1} matched "
            f"Detection {detection_index + 1}, "
            f"Distance = {distance:.3f}"
        )

    else:

        print(
            f"Track {track_index + 1}: "
            f"No valid detection"
        )

print("\nGated Hungarian test completed!")

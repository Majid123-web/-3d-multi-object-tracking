
import numpy as np
from scipy.optimize import linear_sum_assignment

# Step 1: Load centroid data
centroids = np.load("data/all_centroids.npy")

num_frames = centroids.shape[0]
num_objects = centroids.shape[1]

# Store positions using permanent tracking IDs
tracked_positions = np.zeros_like(centroids)

# Step 2: Assign initial IDs in Frame 0
tracked_positions[0] = centroids[0]

print("Frame 0: Initial object IDs assigned")

for obj in range(num_objects):
    print(f"Object ID {obj + 1}: {centroids[0, obj]}")

# Step 3: Match objects in subsequent frames
for frame in range(1, num_frames):

    previous_positions = tracked_positions[frame - 1]
    current_positions = centroids[frame]

    # Calculate pairwise Euclidean distances
    distance_matrix = np.linalg.norm(
        previous_positions[:, np.newaxis, :] -
        current_positions[np.newaxis, :, :],
        axis=2
    )

    # Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(
        distance_matrix
    )

    # Preserve the IDs from the previous frame
    for previous_index, current_index in zip(
        row_indices, col_indices
    ):

        tracked_positions[frame, previous_index] = (
            current_positions[current_index]
        )

        distance = distance_matrix[
            previous_index, current_index
        ]

        print(
            f"Frame {frame}: "
            f"Object ID {previous_index + 1} matched "
            f"with Cluster {current_index + 1}, "
            f"Distance = {distance:.3f}"
        )

# Step 4: Save tracked positions
np.save("data/tracked_positions.npy", tracked_positions)

print("\nObject matching completed!")
print("Tracked positions saved successfully!")


import numpy as np
from scipy.optimize import linear_sum_assignment

# Step 1: Load shuffled centroid data
centroids = np.load("data/shuffled_centroids.npy")

num_frames = centroids.shape[0]
num_objects = centroids.shape[1]

# Step 2: Initialize object tracks
tracked_positions = np.zeros(
    (num_frames, num_objects, 3)
)

# Frame 0 has the original detection order
tracked_positions[0] = centroids[0]

print("\nInitial object tracks created.")

for obj in range(num_objects):
    print(
        f"Object {obj + 1} initial position:",
        np.round(tracked_positions[0, obj], 3)
    )

# Step 3: Basic Hungarian data association

for frame in range(1, num_frames):

    previous_positions = tracked_positions[frame - 1]

    current_positions = centroids[frame]

    # Calculate distance matrix
    distance_matrix = np.linalg.norm(
        previous_positions[:, np.newaxis, :] -
        current_positions[np.newaxis, :, :],
        axis=2
    )

    # Apply Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(
        distance_matrix
    )

    # Assign detections to existing tracks
    for track_index, detection_index in zip(
        row_indices, col_indices
    ):

        tracked_positions[frame, track_index] = (
            current_positions[detection_index]
        )

        print(
            f"Frame {frame}: "
            f"Object {track_index + 1}, "
            f"Matched Cluster {detection_index + 1}, "
            f"Distance = "
            f"{distance_matrix[track_index, detection_index]:.3f}"
        )

# Step 4: Save basic association results

np.save(
    "data/basic_associated_positions.npy",
    tracked_positions
)

print("\nBasic Hungarian association completed!")
print("Tracked positions shape:", tracked_positions.shape)
print("Saved: data/basic_associated_positions.npy")


print("Number of frames:", num_frames)
print("Number of objects:", num_objects)
print("Data shape:", centroids.shape)

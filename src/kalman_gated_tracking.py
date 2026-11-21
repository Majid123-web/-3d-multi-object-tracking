
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter3D

# Step 1: Load shuffled detections
centroids = np.load("data/shuffled_centroids.npy")

num_frames = centroids.shape[0]
num_objects = centroids.shape[1]

max_distance = 2.0
large_cost = 1e6

# Step 2: Initialize Kalman filters
filters = []

for obj in range(num_objects):

    kf = KalmanFilter3D(dt=1.0)

    kf.initialize(
        position=centroids[0, obj],
        velocity=[0, 0, 0]
    )

    filters.append(kf)

# Step 3: Store tracking results
tracked_positions = np.zeros(
    (num_frames, num_objects, 3)
)

tracked_positions[0] = centroids[0]

# Step 4: Kalman prediction and gated association
for frame in range(1, num_frames):

    predicted_positions = []

    for kf in filters:
        predicted_positions.append(kf.predict())

    predicted_positions = np.array(predicted_positions)

    current_detections = centroids[frame]

    # Calculate distance matrix
    distance_matrix = np.linalg.norm(
        predicted_positions[:, np.newaxis, :] -
        current_detections[np.newaxis, :, :],
        axis=2
    )

    # Apply gating
    cost_matrix = distance_matrix.copy()

    cost_matrix[cost_matrix > max_distance] = large_cost

    # Hungarian assignment
    row_indices, col_indices = linear_sum_assignment(
        cost_matrix
    )

    # Store predicted positions initially
    tracked_positions[frame] = predicted_positions

    # Update only valid matches
    for track_index, detection_index in zip(
        row_indices, col_indices
    ):

        distance = distance_matrix[
            track_index, detection_index
        ]

        if distance <= max_distance:

            measurement = current_detections[detection_index]

            updated_position = filters[track_index].update(
                measurement
            )

            tracked_positions[frame, track_index] = updated_position

            print(
                f"Frame {frame}: "
                f"Track {track_index + 1}, "
                f"Detection {detection_index + 1}, "
                f"Distance = {distance:.3f}, MATCHED"
            )

        else:

            print(
                f"Frame {frame}: "
                f"Track {track_index + 1}, "
                f"NO VALID DETECTION"
            )

# Step 5: Save results
np.save(
    "data/kalman_gated_positions.npy",
    tracked_positions
)

print("\nKalman gated tracking completed!")
print("Tracked positions shape:", tracked_positions.shape)
print("Saved: data/kalman_gated_positions.npy")

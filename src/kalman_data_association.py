import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter3D
# Step 1: Load centroid data
centroids = np.load("data/shuffled_centroids.npy")

num_frames = centroids.shape[0]
num_objects = centroids.shape[1]

# Step 2: Initialize Kalman filters

filters = []

for obj in range(num_objects):

    kf = KalmanFilter3D(dt=1.0)

    initial_position = centroids[0, obj]

    kf.initialize(
        position=initial_position,
        velocity=[0, 0, 0]
    )

    filters.append(kf)
    # Step 3: Store tracked positions

tracked_positions = np.zeros(
    (num_frames, num_objects, 3)
)

tracked_positions[0] = centroids[0]

# Step 4: Kalman prediction and data association

for frame in range(1, num_frames):

    # Predict positions
    predicted_positions = []

    for obj in range(num_objects):
        predicted_position = filters[obj].predict()
        predicted_positions.append(predicted_position)

    predicted_positions = np.array(predicted_positions)

    # Get current measurements
    current_positions = centroids[frame]

    # Calculate distance matrix
    distance_matrix = np.linalg.norm(
        predicted_positions[:, np.newaxis, :] -
        current_positions[np.newaxis, :, :],
        axis=2
    )

    # Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(
        distance_matrix
    )

    # Update Kalman filters
    for track_index, detection_index in zip(
        row_indices, col_indices
    ):

        measurement = current_positions[detection_index]

        updated_position = filters[track_index].update(
            measurement
        )

        tracked_positions[frame, track_index] = updated_position

        print(
            f"Frame {frame}: Object {track_index + 1}, "
            f"Matched Cluster {detection_index + 1}, "
            f"Distance = {distance_matrix[track_index, detection_index]:.3f}"
        )
        # Step 5: Save tracking results

np.save(
    "data/kalman_associated_positions.npy",
    tracked_positions
)

print("\nKalman-based data association completed!")
print("Tracked positions shape:", tracked_positions.shape)
print("Saved: data/kalman_associated_positions.npy")
print("\nInitial object positions stored.")
print("Tracked positions shape:", tracked_positions.shape)

print(f"Kalman Filter initialized for Object {obj + 1}")
print("Number of frames:", num_frames)
print("Number of objects:", num_objects)
print("Centroid data shape:", centroids.shape)
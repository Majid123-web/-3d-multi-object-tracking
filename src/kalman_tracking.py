
import numpy as np
from kalman_filter import KalmanFilter3D

# Load detected object centroids
centroids = np.load("data/all_centroids.npy")

num_frames = centroids.shape[0]
num_objects = centroids.shape[1]

# Create one independent Kalman Filter per object
filters = []

for obj in range(num_objects):

    kf = KalmanFilter3D(dt=1.0)

    # Initialize using the first detected position
    initial_position = centroids[0, obj]

    # Initial velocity is set to zero
    kf.initialize(
        position=initial_position,
        velocity=[0, 0, 0]
    )

    filters.append(kf)

# Store filtered positions
filtered_positions = np.zeros_like(centroids)

# First frame
filtered_positions[0] = centroids[0]

print("3D Kalman Tracking Started")

# Process remaining frames
for frame in range(1, num_frames):

    print(f"\nFrame {frame}:")

    for obj in range(num_objects):

        # Predict object position
        predicted_position = filters[obj].predict()

        # Get actual centroid measurement
        measurement = centroids[frame, obj]

        # Correct prediction using measurement
        updated_position = filters[obj].update(measurement)

        filtered_positions[frame, obj] = updated_position

        print(f"Object {obj + 1}:")
        print("Predicted:", np.round(predicted_position, 3))
        print("Measured: ", np.round(measurement, 3))
        print("Filtered: ", np.round(updated_position, 3))

# Save filtered trajectories
np.save("data/kalman_filtered_positions.npy", filtered_positions)

print("\nKalman Tracking Completed!")
print("Filtered positions shape:", filtered_positions.shape)
print("Saved: data/kalman_filtered_positions.npy")

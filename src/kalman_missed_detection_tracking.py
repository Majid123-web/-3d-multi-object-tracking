
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter3D

# Load data containing missed detections
detections = np.load("data/missed_detection_centroids.npy")

num_frames = detections.shape[0]
num_objects = 3

# Initialize Kalman filters
filters = []

for obj in range(num_objects):
    kf = KalmanFilter3D(dt=1.0)
    kf.initialize(detections[0, obj])
    filters.append(kf)

# Store tracked positions
tracked_positions = np.zeros((num_frames, num_objects, 3))

# Store initial positions
for obj in range(num_objects):
    tracked_positions[0, obj] = detections[0, obj]

max_distance = 2.0

print("Kalman Tracking with Missed Detections Started\n")

for frame in range(1, num_frames):

    # Predict every track
    predictions = []

    for kf in filters:
        predictions.append(kf.predict())

    predictions = np.array(predictions)

    # Keep predicted positions as default
    tracked_positions[frame] = predictions

    # Get current detections
    current_detections = detections[frame]

    # Remove missing detections (NaN values)
    valid_mask = np.all(np.isfinite(current_detections), axis=1)

    valid_detections = current_detections[valid_mask]

    print(f"Frame {frame}:")

    missing_count = np.sum(~valid_mask)

    if missing_count > 0:
        print(f"  Missing detections: {missing_count}")

    # Associate only valid detections
    if len(valid_detections) > 0:

        distance_matrix = np.zeros(
            (num_objects, len(valid_detections))
        )

        for i in range(num_objects):
            for j in range(len(valid_detections)):
                distance_matrix[i, j] = np.linalg.norm(
                    predictions[i] - valid_detections[j]
                )

        # Apply distance gating
        cost_matrix = distance_matrix.copy()
        cost_matrix[cost_matrix > max_distance] = 1e6

        rows, cols = linear_sum_assignment(cost_matrix)

        for row, col in zip(rows, cols):

            distance = distance_matrix[row, col]

            if distance <= max_distance:

                updated_position = filters[row].update(
                    valid_detections[col]
                )

                tracked_positions[frame, row] = updated_position

                print(
                    f"  Track {row + 1}: "
                    f"Measurement received, distance = {distance:.3f}"
                )

            else:
                print(
                    f"  Track {row + 1}: "
                    "No valid measurement; prediction retained"
                )

    else:
        print("  No valid detections in this frame")

    print()

# Save results
np.save(
    "data/kalman_missed_detection_positions.npy",
    tracked_positions
)

print("Tracking completed!")
print("Tracked positions shape:", tracked_positions.shape)
print("Saved: data/kalman_missed_detection_positions.npy")

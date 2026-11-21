
import numpy as np

# Load shuffled centroid data
centroids = np.load("data/shuffled_centroids.npy")

# Create a copy
missed_data = centroids.copy()

# Simulate missing detections
# Each entry represents (frame, detection index)

missing_cases = [
    (3, 0),
    (5, 1),
    (7, 2)
]

print("Simulating missed detections...\n")

for frame, detection in missing_cases:

    missed_data[frame, detection] = np.nan

    print(
        f"Frame {frame}: "
        f"Detection {detection + 1} removed"
    )

# Save modified data
np.save(
    "data/missed_detection_centroids.npy",
    missed_data
)

print("\nMissed detection simulation completed!")
print("Data shape:", missed_data.shape)
print("Saved: data/missed_detection_centroids.npy")

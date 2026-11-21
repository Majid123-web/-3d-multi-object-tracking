
import numpy as np

# Load original centroid data
centroids = np.load("data/all_centroids.npy")

# Make a copy
long_missing_data = centroids.copy()

# Object 1 disappears from frames 3 to 7
missing_object = 0
missing_frames = [3, 4, 5, 6, 7]

print("Simulating long object disappearance...\n")

for frame in missing_frames:

    long_missing_data[frame, missing_object] = np.nan

    print(
        f"Frame {frame}: "
        "Object 1 detection removed"
    )

print("\nFrame 8: Object 1 reappears")

# Save modified dataset
np.save(
    "data/long_missing_detections.npy",
    long_missing_data
)

print("\nSimulation completed!")
print("Data shape:", long_missing_data.shape)
print("Saved: data/long_missing_detections.npy")

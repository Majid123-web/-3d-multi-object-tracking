
import numpy as np

# Load original centroid data
centroids = np.load("data/all_centroids.npy")

num_frames = centroids.shape[0]
num_objects = centroids.shape[1]

# Create a copy of the original data
shuffled_centroids = centroids.copy()

# Random generator for reproducible results
rng = np.random.default_rng(seed=42)

print("Shuffling detection order...\n")

# Keep frame 0 unchanged for track initialization
for frame in range(1, num_frames):

    # Generate a random permutation of object indices
    permutation = rng.permutation(num_objects)

    # Shuffle detections in this frame
    shuffled_centroids[frame] = centroids[frame, permutation]

    print(
        f"Frame {frame}: "
        f"Detection order = {permutation + 1}"
    )

# Save shuffled detections
np.save(
    "data/shuffled_centroids.npy",
    shuffled_centroids
)

print("\nDetection shuffling completed!")
print("Saved: data/shuffled_centroids.npy")
print("Data shape:", shuffled_centroids.shape)

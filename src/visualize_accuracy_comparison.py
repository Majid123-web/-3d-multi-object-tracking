
import numpy as np
import matplotlib.pyplot as plt

# Load ground-truth and estimated positions
ground_truth = np.load("data/ground_truth_positions.npy")
measured = np.load("data/all_centroids.npy")
filtered = np.load("data/kalman_filtered_positions.npy")

# Calculate Euclidean error for every object in every frame
measured_errors = np.linalg.norm(
    measured - ground_truth, axis=2
)

filtered_errors = np.linalg.norm(
    filtered - ground_truth, axis=2
)

# Average error across all three objects in each frame
measured_frame_error = np.mean(measured_errors, axis=1)
filtered_frame_error = np.mean(filtered_errors, axis=1)

# Frame numbers
frames = np.arange(len(ground_truth))

# Plot comparison
plt.figure(figsize=(10, 6))

plt.plot(
    frames,
    measured_frame_error,
    marker="o",
    linestyle="--",
    label="Original Measurements"
)

plt.plot(
    frames,
    filtered_frame_error,
    marker="x",
    linestyle="-",
    label="Kalman Filter"
)

plt.title("Frame-by-Frame Tracking Error Comparison")
plt.xlabel("Frame Number")
plt.ylabel("Mean Euclidean Error")
plt.xticks(frames)
plt.grid(True)
plt.legend()

plt.savefig(
    "results/framewise_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Frame-by-frame accuracy visualization completed!")
print("Graph saved in results/framewise_accuracy_comparison.png")

print("\nFrame-wise Error Values:")
for frame in frames:
    print(
        f"Frame {frame}: "
        f"Measured = {measured_frame_error[frame]:.5f}, "
        f"Kalman = {filtered_frame_error[frame]:.5f}"
    )

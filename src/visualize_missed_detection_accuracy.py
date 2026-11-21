
import numpy as np
import matplotlib.pyplot as plt

# Load data
ground_truth = np.load("data/ground_truth_positions.npy")

original = np.load("data/kalman_associated_positions.npy")

missed = np.load(
    "data/kalman_missed_detection_positions.npy"
)

# Calculate 3D position errors
original_error = np.linalg.norm(
    original - ground_truth, axis=2
)

missed_error = np.linalg.norm(
    missed - ground_truth, axis=2
)

# Average error across all three objects for each frame
original_frame_error = np.mean(original_error, axis=1)

missed_frame_error = np.mean(missed_error, axis=1)

frames = np.arange(len(ground_truth))

# Create comparison graph
plt.figure(figsize=(11, 6))

plt.plot(
    frames,
    original_frame_error,
    marker="o",
    linewidth=2,
    label="Original Kalman Tracking"
)

plt.plot(
    frames,
    missed_frame_error,
    marker="s",
    linewidth=2,
    label="Tracking with Missed Detections"
)

# Highlight frames with missing detections
for frame in [3, 5, 7]:
    plt.axvspan(
        frame - 0.3,
        frame + 0.3,
        alpha=0.15
    )

plt.xlabel("Frame Number")
plt.ylabel("Mean 3D Position Error")

plt.title(
    "Kalman Tracking Accuracy with Missed Detections"
)

plt.xticks(frames)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/missed_detection_accuracy.png",
    dpi=300
)

plt.show()

print("Visualization completed!")
print("Saved: results/missed_detection_accuracy.png")

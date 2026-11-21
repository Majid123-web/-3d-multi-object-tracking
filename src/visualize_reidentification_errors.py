
import numpy as np
import matplotlib.pyplot as plt

estimated = np.load("data/reidentified_positions.npy")
history = np.load("data/track_identity_history.npy")

# Ground truth
initial_positions = np.array([
    [0, 0, 0],
    [0, 5, 0],
    [0, 0, 5]
], dtype=float)

velocities = np.array([
    [1, 0, 0],
    [0, 0.5, 0],
    [0.5, 0.5, 0]
], dtype=float)

num_frames = estimated.shape[0]

ground_truth = np.zeros_like(estimated)

for frame in range(num_frames):
    ground_truth[frame] = (
        initial_positions + velocities * frame
    )

# Calculate 3D position error
errors = np.linalg.norm(
    estimated - ground_truth,
    axis=2
)

frames = np.arange(num_frames)

plt.figure(figsize=(11, 6))

for track in range(3):

    plt.plot(
        frames,
        errors[:, track],
        marker="o",
        linewidth=2,
        label=f"Track {track + 1}"
    )

# Highlight disappearance period
plt.axvspan(
    3, 7,
    alpha=0.15,
    label="Disappearance period"
)

plt.xlabel("Frame Number")
plt.ylabel("3D Position Error")
plt.title("Position Estimation Error During Track Re-identification")

plt.xticks(frames)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/reidentification_position_errors.png",
    dpi=300
)

plt.show()

print("Re-identification error graph saved!")
print("File: results/reidentification_position_errors.png")

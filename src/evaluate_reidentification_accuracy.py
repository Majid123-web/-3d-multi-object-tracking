
import numpy as np

# Load estimated positions
estimated = np.load("data/reidentified_positions.npy")

# Generate ground-truth positions
num_frames = 10

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

ground_truth = np.zeros((num_frames, 3, 3))

for frame in range(num_frames):
    ground_truth[frame] = (
        initial_positions + velocities * frame
    )

# Compare only positions that were recorded
valid_mask = np.all(np.isfinite(estimated), axis=2)

errors = np.linalg.norm(
    estimated - ground_truth,
    axis=2
)

valid_errors = errors[valid_mask]

# Calculate metrics
mae = np.mean(np.abs(
    estimated[valid_mask] - ground_truth[valid_mask]
))

rmse = np.sqrt(np.mean(
    (estimated[valid_mask] - ground_truth[valid_mask]) ** 2
))

mean_3d_error = np.mean(valid_errors)

print("RE-IDENTIFICATION POSITION EVALUATION")
print("=" * 45)

print("Valid position estimates:", len(valid_errors))
print("MAE:", round(mae, 6))
print("RMSE:", round(rmse, 6))
print("Mean 3D Position Error:", round(mean_3d_error, 6))

print("\nTrack 1 position errors:")

for frame in range(num_frames):

    if valid_mask[frame, 0]:
        print(
            f"Frame {frame}: "
            f"{errors[frame, 0]:.6f}"
        )

print("\nEvaluation completed!")


import numpy as np

# Load ground truth
ground_truth = np.load("data/ground_truth_positions.npy")

# Load original Kalman tracking
original = np.load("data/kalman_associated_positions.npy")

# Load tracking with missed detections
missed = np.load(
    "data/kalman_missed_detection_positions.npy"
)

# Calculate errors
original_error = original - ground_truth
missed_error = missed - ground_truth

# Mean Absolute Error
original_mae = np.mean(np.abs(original_error))
missed_mae = np.mean(np.abs(missed_error))

# Root Mean Squared Error
original_rmse = np.sqrt(np.mean(original_error ** 2))
missed_rmse = np.sqrt(np.mean(missed_error ** 2))

# Mean 3D position error
original_3d = np.linalg.norm(original_error, axis=2)
missed_3d = np.linalg.norm(missed_error, axis=2)

original_mean_3d = np.mean(original_3d)
missed_mean_3d = np.mean(missed_3d)

print("TRACKING ACCURACY COMPARISON")
print("--------------------------------")

print("\nOriginal Kalman Tracking:")
print(f"MAE: {original_mae:.6f}")
print(f"RMSE: {original_rmse:.6f}")
print(f"Mean 3D Error: {original_mean_3d:.6f}")

print("\nTracking with Missed Detections:")
print(f"MAE: {missed_mae:.6f}")
print(f"RMSE: {missed_rmse:.6f}")
print(f"Mean 3D Error: {missed_mean_3d:.6f}")

print("\nAccuracy Difference:")
print(f"MAE increase: {missed_mae - original_mae:.6f}")
print(f"RMSE increase: {missed_rmse - original_rmse:.6f}")
print(f"3D error increase: {missed_mean_3d - original_mean_3d:.6f}")

# Evaluate only frames with missing detections
missing_frames = [3, 5, 7]

print("\nERROR ON MISSING-DETECTION FRAMES")
print("--------------------------------")

for frame in missing_frames:
    print(f"\nFrame {frame}:")

    for obj in range(3):
        print(
            f"Track {obj + 1}: "
            f"3D error = {missed_3d[frame, obj]:.6f}"
        )

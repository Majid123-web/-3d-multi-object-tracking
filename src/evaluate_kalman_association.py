
import numpy as np

# Load ground truth and associated tracking results
ground_truth = np.load("data/ground_truth_positions.npy")
associated = np.load("data/kalman_associated_positions.npy")

# Calculate tracking errors
errors = associated - ground_truth

# Overall metrics
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))

# Euclidean position error
euclidean_errors = np.linalg.norm(errors, axis=2)
mean_position_error = np.mean(euclidean_errors)

print("KALMAN DATA ASSOCIATION EVALUATION")
print("----------------------------------")

print(f"Overall MAE: {mae:.6f}")
print(f"Overall RMSE: {rmse:.6f}")
print(f"Mean 3D Position Error: {mean_position_error:.6f}")

# Evaluate each object separately
for obj in range(associated.shape[1]):

    object_error = errors[:, obj, :]

    object_mae = np.mean(np.abs(object_error))
    object_rmse = np.sqrt(np.mean(object_error ** 2))

    print(f"\nObject {obj + 1}:")
    print(f"MAE: {object_mae:.6f}")
    print(f"RMSE: {object_rmse:.6f}")

print("\nEvaluation completed!")

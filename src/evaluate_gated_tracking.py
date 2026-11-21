
import numpy as np

# Load ground truth and tracking results
ground_truth = np.load("data/ground_truth_positions.npy")

gated = np.load("data/kalman_gated_positions.npy")

# Calculate errors
errors = gated - ground_truth

mae = np.mean(np.abs(errors))

rmse = np.sqrt(np.mean(errors ** 2))

euclidean_errors = np.linalg.norm(errors, axis=2)

mean_position_error = np.mean(euclidean_errors)

print("KALMAN GATED TRACKING EVALUATION")
print("--------------------------------")

print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"Mean 3D Position Error: {mean_position_error:.6f}")

print("\nIndividual Object Errors")

for obj in range(gated.shape[1]):

    object_error = errors[:, obj, :]

    object_mae = np.mean(np.abs(object_error))

    object_rmse = np.sqrt(np.mean(object_error ** 2))

    print(f"\nObject {obj + 1}:")
    print(f"MAE: {object_mae:.6f}")
    print(f"RMSE: {object_rmse:.6f}")

print("\nGated tracking evaluation completed!")

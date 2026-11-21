
import numpy as np

# Simulation settings
num_frames = 10
num_objects = 3

# Initial positions
initial_positions = np.array([
    [0, 0, 0],
    [0, 5, 0],
    [0, 0, 5]
], dtype=float)

# Actual velocities
velocities = np.array([
    [1, 0, 0],
    [0, 0.5, 0],
    [0.5, 0.5, 0]
], dtype=float)

# Generate ground-truth positions
ground_truth = np.zeros((num_frames, num_objects, 3))

for frame in range(num_frames):
    for obj in range(num_objects):
        ground_truth[frame, obj] = (
            initial_positions[obj] +
            velocities[obj] * frame
        )

# Save ground-truth positions
np.save("data/ground_truth_positions.npy", ground_truth)


# Load measured and Kalman-filtered positions
measured = np.load("data/all_centroids.npy")
filtered = np.load("data/kalman_filtered_positions.npy")

# Calculate errors
measured_errors = measured - ground_truth
filtered_errors = filtered - ground_truth

# Mean Absolute Error (MAE)
measured_mae = np.mean(np.abs(measured_errors))
filtered_mae = np.mean(np.abs(filtered_errors))

# Root Mean Squared Error (RMSE)
measured_rmse = np.sqrt(np.mean(measured_errors ** 2))
filtered_rmse = np.sqrt(np.mean(filtered_errors ** 2))

# Display results
print("\n========== ACCURACY COMPARISON ==========")

print("\nOriginal Measurements:")
print("MAE:", measured_mae)
print("RMSE:", measured_rmse)

print("\nKalman Filter:")
print("MAE:", filtered_mae)
print("RMSE:", filtered_rmse)

print("\n========== IMPROVEMENT ==========")

mae_improvement = (
    (measured_mae - filtered_mae) / measured_mae
) * 100

rmse_improvement = (
    (measured_rmse - filtered_rmse) / measured_rmse
) * 100


# Estimate measurement noise from ground truth
measurement_noise = measured - ground_truth

# Reshape residuals into individual 3D measurements
noise_samples = measurement_noise.reshape(-1, 3)

# Calculate measurement noise statistics
noise_mean = np.mean(noise_samples, axis=0)
noise_std = np.std(noise_samples, axis=0)
noise_covariance = np.cov(noise_samples, rowvar=False)

print("\n========== MEASUREMENT NOISE ANALYSIS ==========")

print("\nMean Noise [X, Y, Z]:")
print(noise_mean)

print("\nNoise Standard Deviation [X, Y, Z]:")
print(noise_std)

print("\nEstimated Measurement Noise Covariance:")
print(noise_covariance)

print("\nCurrent Kalman R Matrix:")
print(np.eye(3) * 0.5)


print("Ground-truth positions generated!")
print("Shape:", ground_truth.shape)
print("\nFrame 0:")
print(ground_truth[0])

print("\nFrame 9:")
print(ground_truth[9])

print("MAE Improvement:", round(mae_improvement, 2), "%")
print("RMSE Improvement:", round(rmse_improvement, 2), "%")

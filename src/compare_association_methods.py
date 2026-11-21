
import numpy as np

# Load ground truth and tracking results
ground_truth = np.load("data/ground_truth_positions.npy")

basic = np.load("data/basic_associated_positions.npy")

kalman = np.load("data/kalman_associated_positions.npy")


def evaluate_tracking(predicted, actual):

    errors = predicted - actual

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(errors ** 2))

    euclidean_errors = np.linalg.norm(errors, axis=2)

    mean_position_error = np.mean(euclidean_errors)

    return mae, rmse, mean_position_error


# Evaluate both methods
basic_results = evaluate_tracking(basic, ground_truth)

kalman_results = evaluate_tracking(kalman, ground_truth)


print("\nTRACKING ALGORITHM COMPARISON")
print("--------------------------------")

print("\nMethod A: Basic Hungarian Association")

print(f"MAE: {basic_results[0]:.6f}")
print(f"RMSE: {basic_results[1]:.6f}")
print(f"Mean 3D Position Error: {basic_results[2]:.6f}")


print("\nMethod B: Kalman + Hungarian Association")

print(f"MAE: {kalman_results[0]:.6f}")
print(f"RMSE: {kalman_results[1]:.6f}")
print(f"Mean 3D Position Error: {kalman_results[2]:.6f}")


print("\nCOMPARISON")
print("--------------------------------")

if kalman_results[2] < basic_results[2]:

    print("Kalman-based association has lower mean position error.")

elif kalman_results[2] > basic_results[2]:

    print("Basic Hungarian association has lower mean position error.")

else:

    print("Both methods have equal mean position error.")

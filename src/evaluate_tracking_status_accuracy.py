
import numpy as np

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

# Calculate 3D position errors
errors = np.linalg.norm(
    estimated - ground_truth,
    axis=2
)

print("TRACKING ACCURACY BY STATUS")
print("=" * 45)

statuses = {
    1: "Active",
    2: "Dormant",
    3: "Re-identified"
}

for status, name in statuses.items():

    mask = (history == status)

    selected_errors = errors[mask]

    print(f"\n{name}:")

    if len(selected_errors) > 0:

        print("Number of estimates:", len(selected_errors))
        print("Mean 3D error:", round(
            np.mean(selected_errors), 6
        ))
        print("Maximum 3D error:", round(
            np.max(selected_errors), 6
        ))

    else:
        print("No estimates available")

print("\nEvaluation completed!")


import numpy as np
import matplotlib.pyplot as plt

# Load actual and predicted positions
actual = np.load("data/tracked_positions.npy")
predicted = np.load("data/predicted_positions.npy")

colors = ["red", "blue", "green"]

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

for obj in range(3):

    # Actual positions corresponding to predictions
    actual_part = actual[2:, obj, :]

    # Predicted positions
    predicted_part = predicted[:, obj, :]

    # Plot actual trajectory
    ax.plot(
        actual_part[:, 0],
        actual_part[:, 1],
        actual_part[:, 2],
        color=colors[obj],
        marker="o",
        label=f"Object {obj + 1} Actual"
    )

    # Plot predicted trajectory
    ax.plot(
        predicted_part[:, 0],
        predicted_part[:, 1],
        predicted_part[:, 2],
        color=colors[obj],
        linestyle="--",
        marker="x",
        label=f"Object {obj + 1} Predicted"
    )

ax.set_title("Actual vs Predicted 3D Trajectories")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=True)

print("Prediction visualization completed!")

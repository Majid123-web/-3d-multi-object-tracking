
import numpy as np
import matplotlib.pyplot as plt

# Load original and filtered positions
original = np.load("data/all_centroids.npy")
filtered = np.load("data/kalman_filtered_positions.npy")

colors = ["red", "blue", "green"]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

for obj in range(3):

    # Original measured trajectory
    ax.plot(
        original[:, obj, 0],
        original[:, obj, 1],
        original[:, obj, 2],
        color=colors[obj],
        linestyle=":",
        marker="o",
        label=f"Object {obj + 1} Measured"
    )

    # Kalman-filtered trajectory
    ax.plot(
        filtered[:, obj, 0],
        filtered[:, obj, 1],
        filtered[:, obj, 2],
        color=colors[obj],
        linestyle="-",
        marker="x",
        label=f"Object {obj + 1} Kalman Filtered"
    )

ax.set_title("3D Object Tracking: Measured vs Kalman Filtered")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend()

plt.savefig(
    "results/kalman_tracking_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=True)

print("Kalman tracking visualization completed!")
print("Graph saved in results/kalman_tracking_comparison.png")

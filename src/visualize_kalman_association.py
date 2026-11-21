
import numpy as np
import matplotlib.pyplot as plt

# Load tracking results
original = np.load("data/all_centroids.npy")
filtered = np.load("data/kalman_filtered_positions.npy")
associated = np.load("data/kalman_associated_positions.npy")

colors = ["red", "blue", "green"]

fig = plt.figure(figsize=(13, 9))
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

    # Previous Kalman-filtered trajectory
    ax.plot(
        filtered[:, obj, 0],
        filtered[:, obj, 1],
        filtered[:, obj, 2],
        color=colors[obj],
        linestyle="--",
        marker="x",
        label=f"Object {obj + 1} Kalman Filtered"
    )

    # New Kalman-associated trajectory
    ax.plot(
        associated[:, obj, 0],
        associated[:, obj, 1],
        associated[:, obj, 2],
        color=colors[obj],
        linestyle="-",
        marker="^",
        label=f"Object {obj + 1} Kalman Associated"
    )

ax.set_title("3D Tracking: Measured vs Kalman Filtering vs Association")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.legend(fontsize=8)

plt.savefig(
    "results/kalman_association_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=True)

print("Kalman association visualization completed!")
print("Graph saved in results/kalman_association_comparison.png")

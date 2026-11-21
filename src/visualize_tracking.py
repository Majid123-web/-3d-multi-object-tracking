
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load tracked positions
tracked_positions = np.load("data/tracked_positions.npy")

print("Tracked data loaded successfully!")
print("Dataset shape:", tracked_positions.shape)

# Step 2: Create a 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Step 3: Define object colors
colors = ["red", "blue", "green"]

# Step 4: Draw each object's trajectory
for obj in range(tracked_positions.shape[1]):

    x = tracked_positions[:, obj, 0]
    y = tracked_positions[:, obj, 1]
    z = tracked_positions[:, obj, 2]

    ax.plot(
        x, y, z,
        marker="o",
        color=colors[obj],
        label=f"Object ID {obj + 1}"
    )

    # Mark the starting position
    ax.scatter(x[0], y[0], z[0], marker="s", s=70, color=colors[obj])
    # Mark the ending position 
    ax.scatter(x[-1], y[-1], z[-1], marker="^", s=70, color=colors[obj])

    # Mark the ending position
    ax.scatter(x[-1], y[-1], z[-1], marker="^", s=70)

# Step 5: Add labels and title
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")

ax.set_title("3D Multi-Object Tracking Trajectories")
ax.legend()

plt.savefig("tracking_visualization.png", dpi=300, bbox_inches="tight")

plt.show(block=True)

print("3D tracking visualization completed!")
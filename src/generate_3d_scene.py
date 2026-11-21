
import numpy as np
import matplotlib.pyplot as plt

# Time steps
frames = np.arange(0, 10)

# Initial positions of three objects
initial_positions = np.array([
    [0, 0, 0],
    [0, 5, 0],
    [0, 0, 5]
])

# Velocity vectors
velocities = np.array([
    [1, 0, 0],
    [0, 0.5, 0],
    [0.5, 0.5, 0]
])

# Calculate positions over time
positions = []

for t in frames:
    current_positions = initial_positions + velocities * t
    positions.append(current_positions)

positions = np.array(positions)

# Display 3D trajectories
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

for obj in range(3):
    ax.plot(
        positions[:, obj, 0],
        positions[:, obj, 1],
        positions[:, obj, 2],
        marker="o",
        label=f"Object {obj + 1}"
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Object Movement Across Frames")
ax.legend()

plt.show()

# Print positions
print("Object positions across frames:")
print(positions)

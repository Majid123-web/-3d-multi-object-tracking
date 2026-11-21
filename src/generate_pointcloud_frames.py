
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Simulation settings
num_frames = 10
points_per_object = 100

# Initial positions (X, Y, Z)
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

# Generate one point cloud around an object center
def generate_object_points(center, num_points):

    # Random points around the center
    points = np.random.normal(
        loc=center,
        scale=0.4,
        size=(num_points, 3)
    )

    return points


# Store all frames
all_frames = []

for frame in range(num_frames):

    current_time = frame

    frame_objects = []

    for obj in range(3):

        # Calculate object center
        center = (
            initial_positions[obj]
            + velocities[obj] * current_time
        )

        # Generate 100 points around the center
        object_points = generate_object_points(
            center,
            points_per_object
        )

        frame_objects.append(object_points)

    # Combine all three objects
    frame_pointcloud = np.vstack(frame_objects)

    all_frames.append(frame_pointcloud)


# Convert to NumPy array
all_frames = np.array(all_frames)

np.save("data/pointcloud_frames.npy", all_frames)
print("Dataset saved successfully!")

print("Point cloud generation completed!")
print("Number of frames:", all_frames.shape[0])
print("Points per frame:", all_frames.shape[1])
print("Coordinates per point:", all_frames.shape[2])
print("Dataset shape:", all_frames.shape)


# Visualize the first frame
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

colors = ["red", "blue", "green"]

for obj in range(3):

    start = obj * points_per_object
    end = (obj + 1) * points_per_object

    object_points = all_frames[0, start:end]

    ax.scatter(
        object_points[:, 0],
        object_points[:, 1],
        object_points[:, 2],
        color=colors[obj],
        label=f"Object {obj + 1}",
        s=10
    )

ax.set_title("3D Point Clouds - Frame 0")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()

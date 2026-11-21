
import numpy as np

# Step 1: Load centroid data
centroids = np.load("data/all_centroids.npy")

# Time between consecutive frames (seconds)
dt = 1.0

# Step 2: Calculate velocity
# Difference between consecutive positions / time interval
velocities = np.diff(centroids, axis=0) / dt

print("Velocity calculation completed!")

# Step 3: Display velocities
for frame in range(len(velocities)):

    print(f"\nFrame {frame} to Frame {frame + 1}:")

    for obj in range(velocities.shape[1]):

        vx, vy, vz = velocities[frame, obj]

        speed = np.linalg.norm(velocities[frame, obj])

        print(
            f"Object {obj + 1}: "
            f"Vx={vx:.3f}, "
            f"Vy={vy:.3f}, "
            f"Vz={vz:.3f}, "
            f"Speed={speed:.3f}"
        )

# Step 4: Calculate average velocity
average_velocities = np.mean(velocities, axis=0)

print("\nAverage velocity of each object:")

for obj, velocity in enumerate(average_velocities):

    print(
        f"Object {obj + 1}: "
        f"Vx={velocity[0]:.3f}, "
        f"Vy={velocity[1]:.3f}, "
        f"Vz={velocity[2]:.3f}"
    )

# Step 5: Save velocity data
np.save("data/object_velocities.npy", velocities)

print("\nVelocity data saved successfully!")

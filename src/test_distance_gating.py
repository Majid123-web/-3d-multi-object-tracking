
import numpy as np

# Predicted position of an object
predicted_position = np.array([5.0, 2.0, 1.0])

# Three example detections
detections = np.array([
    [5.2, 2.1, 1.0],
    [5.8, 2.5, 1.2],
    [50.0, 40.0, 30.0]
])

# Maximum permitted distance
max_distance = 2.0

print("Distance Gating Test\n")

for i, detection in enumerate(detections):

    distance = np.linalg.norm(
        predicted_position - detection
    )

    if distance <= max_distance:
        print(
            f"Detection {i + 1}: "
            f"Distance = {distance:.3f} "
            f"-> ACCEPTED"
        )
    else:
        print(
            f"Detection {i + 1}: "
            f"Distance = {distance:.3f} "
            f"-> REJECTED"
        )

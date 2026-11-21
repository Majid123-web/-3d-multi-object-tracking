
import numpy as np

# Step 1: Load tracked positions
positions = np.load("data/tracked_positions.npy")

num_frames = positions.shape[0]
num_objects = positions.shape[1]

dt = 1.0

predicted_positions = []

print("Constant Velocity Motion Prediction")
print("------------------------------------")

# Step 2: Predict future positions
for frame in range(num_frames - 2):

    previous_positions = positions[frame]
    current_positions = positions[frame + 1]

    # Step 3: Calculate velocity
    velocities = (
        current_positions - previous_positions
    ) / dt

    # Step 4: Predict the next frame
    predictions = current_positions + velocities * dt

    predicted_positions.append(predictions)

    print(
        f"\nUsing Frames {frame + 1} and {frame + 2}, "
        f"predicting Frame {frame + 3}:"
    )

    for obj in range(num_objects):

        actual_next = positions[frame + 2, obj]
        predicted = predictions[obj]

        error = np.linalg.norm(actual_next - predicted)

        print(f"Object ID {obj + 1}:")
        print("Predicted Position:", np.round(predicted, 3))
        print("Actual Position:   ", np.round(actual_next, 3))
        print("Prediction Error:  ", round(error, 3))

# Step 5: Save predictions
predicted_positions = np.array(predicted_positions)

# Step 6: Evaluate prediction errors

actual_positions = positions[2:]

errors = np.linalg.norm(
    predicted_positions - actual_positions,
    axis=2
)

average_errors = np.mean(errors, axis=0)

print("\nAverage Prediction Error:")
for obj in range(num_objects):
    print(
        f"Object ID {obj + 1}: "
        f"{average_errors[obj]:.4f}"
    )

overall_error = np.mean(errors)

print("\nOverall Average Prediction Error:")
print(round(overall_error, 4))


np.save(
    "data/predicted_positions.npy",
    predicted_positions
)

print("\nMotion prediction completed!")
print("Predictions saved successfully!")
print("Prediction dataset shape:", predicted_positions.shape)

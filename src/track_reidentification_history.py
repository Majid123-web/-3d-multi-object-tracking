
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter3D

# Load dataset
detections = np.load("data/long_missing_detections.npy")

num_frames = len(detections)

max_age = 2
max_distance = 2.0
reid_distance = 2.5

# Track status:
# 0 = Not created
# 1 = Active
# 2 = Dormant / Missing
# 3 = Re-identified

history = np.zeros((num_frames, 3), dtype=int)
positions = np.full((num_frames, 3, 3), np.nan)

active_tracks = []
dormant_tracks = []

next_track_id = 1

# Initialize tracks using Frame 0
for detection in detections[0]:

    kf = KalmanFilter3D(dt=1.0)
    kf.initialize(detection)

    active_tracks.append({
        "id": next_track_id,
        "filter": kf,
        "missed_count": 0
    })

    history[0, next_track_id - 1] = 1
    positions[0, next_track_id - 1] = detection

    next_track_id += 1


# Process remaining frames
for frame in range(1, num_frames):

    current = detections[frame]

    valid_mask = np.all(np.isfinite(current), axis=1)
    valid = current[valid_mask]

    # Predict active tracks
    active_predictions = [
        track["filter"].predict()
        for track in active_tracks
    ]

    # Predict dormant tracks
    dormant_predictions = [
        track["filter"].predict()
        for track in dormant_tracks
    ]

    # Record dormant status in every frame
    for i, track in enumerate(dormant_tracks):

        track_index = track["id"] - 1

        if track_index < 3:

            history[frame, track_index] = 2
            positions[frame, track_index] = dormant_predictions[i]

    matched_active = set()
    matched_detections = set()

    # Associate active tracks with detections
    if active_tracks and len(valid) > 0:

        distances = np.array([
            [
                np.linalg.norm(pred - det)
                for det in valid
            ]
            for pred in active_predictions
        ])

        cost = distances.copy()
        cost[cost > max_distance] = 1e6

        rows, cols = linear_sum_assignment(cost)

        for row, col in zip(rows, cols):

            if distances[row, col] <= max_distance:

                track = active_tracks[row]

                updated = track["filter"].update(valid[col])
                track["missed_count"] = 0

                matched_active.add(row)
                matched_detections.add(col)

                track_index = track["id"] - 1

                if track_index < 3:
                    history[frame, track_index] = 1
                    positions[frame, track_index] = updated

    # Handle missed detections and expired tracks
    expired = []

    for i, track in enumerate(active_tracks):

        if i not in matched_active:

            track["missed_count"] += 1

            track_index = track["id"] - 1

            if track["missed_count"] > max_age:

                expired.append(i)

            else:

                if track_index < 3:
                    history[frame, track_index] = 2
                    positions[frame, track_index] = active_predictions[i]

    # Move expired tracks into dormant archive
    for i in reversed(expired):

        track = active_tracks.pop(i)
        dormant_tracks.append(track)

        track_index = track["id"] - 1

        if track_index < 3:
            history[frame, track_index] = 2
            positions[frame, track_index] = active_predictions[i]

    # Find detections not assigned to active tracks
    unmatched = [
        j for j in range(len(valid))
        if j not in matched_detections
    ]

    # Attempt to re-identify dormant tracks
    if dormant_tracks and unmatched:

        distances = np.array([
            [
                np.linalg.norm(
                    dormant_predictions[i] - valid[j]
                )
                for j in unmatched
            ]
            for i in range(len(dormant_tracks))
        ])

        cost = distances.copy()
        cost[cost > reid_distance] = 1e6

        rows, cols = linear_sum_assignment(cost)

        restored = set()

        for row, col in zip(rows, cols):

            if distances[row, col] <= reid_distance:

                track = dormant_tracks[row]
                detection_index = unmatched[col]

                updated = track["filter"].update(
                    valid[detection_index]
                )

                track["missed_count"] = 0

                active_tracks.append(track)

                track_index = track["id"] - 1

                if track_index < 3:
                    history[frame, track_index] = 3
                    positions[frame, track_index] = updated

                restored.add(row)
                matched_detections.add(detection_index)

        dormant_tracks = [
            track
            for i, track in enumerate(dormant_tracks)
            if i not in restored
        ]

    # Create new tracks for unmatched detections
    for j, detection in enumerate(valid):

        if j not in matched_detections:

            kf = KalmanFilter3D(dt=1.0)
            kf.initialize(detection)

            active_tracks.append({
                "id": next_track_id,
                "filter": kf,
                "missed_count": 0
            })

            if next_track_id <= 3:
                history[frame, next_track_id - 1] = 1
                positions[frame, next_track_id - 1] = detection

            next_track_id += 1


# Save results
np.save("data/track_identity_history.npy", history)
np.save("data/reidentified_positions.npy", positions)

print("Track identity history saved!")
print("History shape:", history.shape)
print("Position shape:", positions.shape)

print("\nTrack 1 status by frame:")
print(history[:, 0])

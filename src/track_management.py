
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter3D

# Load long disappearance dataset
detections = np.load("data/long_missing_detections.npy")

num_frames = detections.shape[0]

max_age = 2
max_distance = 2.0

# Active tracks
tracks = []

next_track_id = 1

# Initialize tracks from frame 0
for detection in detections[0]:

    kf = KalmanFilter3D(dt=1.0)
    kf.initialize(detection)

    tracks.append({
        "id": next_track_id,
        "filter": kf,
        "missed_count": 0
    })

    print(f"Created Track {next_track_id}")
    next_track_id += 1

print("\nTRACK MANAGEMENT STARTED\n")

# Process remaining frames
for frame in range(1, num_frames):

    print(f"Frame {frame}:")

    current_detections = detections[frame]

    valid_mask = np.all(
        np.isfinite(current_detections), axis=1
    )

    valid_detections = current_detections[valid_mask]

    # Predict active tracks
    predictions = []

    for track in tracks:
        prediction = track["filter"].predict()
        predictions.append(prediction)

    predictions = np.array(predictions)

    # Track and detection assignment
    matched_tracks = set()
    matched_detections = set()

    if len(tracks) > 0 and len(valid_detections) > 0:

        distances = np.zeros(
            (len(tracks), len(valid_detections))
        )

        for i in range(len(tracks)):
            for j in range(len(valid_detections)):
                distances[i, j] = np.linalg.norm(
                    predictions[i] - valid_detections[j]
                )

        cost = distances.copy()
        cost[cost > max_distance] = 1e6

        rows, cols = linear_sum_assignment(cost)

        for row, col in zip(rows, cols):

            if distances[row, col] <= max_distance:

                tracks[row]["filter"].update(
                    valid_detections[col]
                )

                tracks[row]["missed_count"] = 0

                matched_tracks.add(row)
                matched_detections.add(col)

                print(
                    f"  Track {tracks[row]['id']}: "
                    "Measurement received"
                )

    # Handle unmatched tracks
    deleted_ids = []

    for i, track in enumerate(tracks):

        if i not in matched_tracks:

            track["missed_count"] += 1

            print(
                f"  Track {track['id']}: "
                f"Missed {track['missed_count']} frame(s)"
            )

            if track["missed_count"] > max_age:

                deleted_ids.append(track["id"])

    # Delete expired tracks
    tracks = [
        track for track in tracks
        if track["id"] not in deleted_ids
    ]

    for track_id in deleted_ids:
        print(f"  Track {track_id} DELETED")

    # Create tracks for unmatched detections
    for j, detection in enumerate(valid_detections):

        if j not in matched_detections:

            kf = KalmanFilter3D(dt=1.0)
            kf.initialize(detection)

            tracks.append({
                "id": next_track_id,
                "filter": kf,
                "missed_count": 0
            })

            print(
                f"  NEW Track {next_track_id} CREATED"
            )

            next_track_id += 1

    print(
        "  Active track IDs:",
        [track["id"] for track in tracks]
    )

    print("\nTrack management completed for this frame.")

print("\nFINAL ACTIVE TRACKS:")
for track in tracks:
    print(f"Track {track['id']}")

print("\nTrack management experiment finished!")


import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter3D

# Load long disappearance dataset
detections = np.load("data/long_missing_detections.npy")

num_frames = detections.shape[0]

max_age = 2
max_distance = 2.0
reid_distance = 2.5

# Active and dormant tracks
active_tracks = []
dormant_tracks = []

next_track_id = 1

# Initialize tracks from frame 0
for detection in detections[0]:

    kf = KalmanFilter3D(dt=1.0)
    kf.initialize(detection)

    active_tracks.append({
        "id": next_track_id,
        "filter": kf,
        "missed_count": 0
    })

    print(f"Created Track {next_track_id}")
    next_track_id += 1

print("\nTRACK RE-IDENTIFICATION STARTED\n")

for frame in range(1, num_frames):

    print(f"Frame {frame}:")

    current_detections = detections[frame]

    valid_mask = np.all(
        np.isfinite(current_detections), axis=1
    )

    valid_detections = current_detections[valid_mask]

    # Predict active tracks
    active_predictions = []

    for track in active_tracks:
        active_predictions.append(
            track["filter"].predict()
        )

    active_predictions = np.array(active_predictions)

    # Predict dormant tracks
    dormant_predictions = []

    for track in dormant_tracks:
        dormant_predictions.append(
            track["filter"].predict()
        )

    dormant_predictions = np.array(dormant_predictions)

    matched_active = set()
    matched_detections = set()

    # Associate detections with active tracks
    if len(active_tracks) > 0 and len(valid_detections) > 0:

        distances = np.zeros(
            (len(active_tracks), len(valid_detections))
        )

        for i in range(len(active_tracks)):
            for j in range(len(valid_detections)):
                distances[i, j] = np.linalg.norm(
                    active_predictions[i] - valid_detections[j]
                )

        cost = distances.copy()
        cost[cost > max_distance] = 1e6

        rows, cols = linear_sum_assignment(cost)

        for row, col in zip(rows, cols):

            if distances[row, col] <= max_distance:

                active_tracks[row]["filter"].update(
                    valid_detections[col]
                )

                active_tracks[row]["missed_count"] = 0

                matched_active.add(row)
                matched_detections.add(col)

                print(
                    f"  Track {active_tracks[row]['id']}: "
                    "Measurement received"
                )

    # Update missed counts and move expired tracks to archive
    expired_tracks = []

    for i, track in enumerate(active_tracks):

        if i not in matched_active:

            track["missed_count"] += 1

            print(
                f"  Track {track['id']}: "
                f"Missed {track['missed_count']} frame(s)"
            )

            if track["missed_count"] > max_age:
                expired_tracks.append(i)

    for i in reversed(expired_tracks):

        track = active_tracks.pop(i)
        dormant_tracks.append(track)

        print(
            f"  Track {track['id']} moved to dormant archive"
        )

    # Re-identify unmatched detections using dormant tracks
    unmatched_detections = [
        j for j in range(len(valid_detections))
        if j not in matched_detections
    ]

    if len(dormant_tracks) > 0 and len(unmatched_detections) > 0:

        reid_distances = np.zeros(
            (len(dormant_tracks), len(unmatched_detections))
        )

        for i in range(len(dormant_tracks)):
            for j, detection_index in enumerate(unmatched_detections):

                reid_distances[i, j] = np.linalg.norm(
                    dormant_predictions[i]
                    - valid_detections[detection_index]
                )

        reid_cost = reid_distances.copy()
        reid_cost[reid_cost > reid_distance] = 1e6

        rows, cols = linear_sum_assignment(reid_cost)

        matched_dormant = set()

        for row, col in zip(rows, cols):

            if reid_distances[row, col] <= reid_distance:

                track = dormant_tracks[row]
                detection_index = unmatched_detections[col]

                track["filter"].update(
                    valid_detections[detection_index]
                )

                track["missed_count"] = 0

                active_tracks.append(track)
                matched_dormant.add(row)
                matched_detections.add(detection_index)

                print(
                    f"  RE-IDENTIFIED: Original Track "
                    f"{track['id']} restored!"
                )

        # Remove restored tracks from dormant archive
        dormant_tracks = [
            track for i, track in enumerate(dormant_tracks)
            if i not in matched_dormant
        ]

    # Create new tracks for detections not matched anywhere
    for j, detection in enumerate(valid_detections):

        if j not in matched_detections:

            kf = KalmanFilter3D(dt=1.0)
            kf.initialize(detection)

            active_tracks.append({
                "id": next_track_id,
                "filter": kf,
                "missed_count": 0
            })

            print(f"  NEW Track {next_track_id} CREATED")
            next_track_id += 1

    print(
        "  Active IDs:",
        [track["id"] for track in active_tracks]
    )

    print(
        "  Dormant IDs:",
        [track["id"] for track in dormant_tracks]
    )

    print()

print("FINAL ACTIVE TRACKS:")
for track in active_tracks:
    print(f"Track {track['id']}")

print("\nFINAL DORMANT TRACKS:")
for track in dormant_tracks:
    print(f"Track {track['id']}")

print("\nTrack re-identification experiment completed!")

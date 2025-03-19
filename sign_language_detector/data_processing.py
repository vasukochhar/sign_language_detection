import cv2
import numpy as np
import mediapipe as mp
import os
from typing import Optional, List

# Initialize MediaPipe Hands globally to avoid repeated instantiation
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_hand_features(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract hand landmarks from a frame using MediaPipe.

    Args:
        frame: Input frame in BGR format.

    Returns:
        Flattened array of hand landmark coordinates (x, y, z) for up to 2 hands,
        padded with zeros if fewer hands detected. Returns None if no hands detected.
    """
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        all_features: List[float] = []
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Corrected list comprehension
            features = [coord for landmark in hand_landmarks.landmark
                       for coord in (landmark.x, landmark.y, landmark.z)]
            all_features.extend(features)

        if len(results.multi_hand_landmarks) == 1:
            all_features.extend([0.0] * 21 * 3)  # Pad for second hand

        return np.array(all_features)
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def collect_data(label: str, dataset_path: str, sequence_length: int, num_sequences: int = 30) -> None:
    """
    Collect training data for a given label.

    Args:
        label: The sign language gesture label.
        dataset_path: Directory to save collected data.
        sequence_length: Number of frames per sequence.
        num_sequences: Number of sequences to collect.
    """
    label_dir = os.path.join(dataset_path, label)
    os.makedirs(label_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    print(f"Starting data collection for '{label}' with {num_sequences} sequences...")
    try:
        for seq in range(num_sequences):
            # Countdown phase
            for countdown in range(5, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Webcam feed interrupted during countdown.")
                cv2.putText(frame, f"Collecting {label}: {seq+1}/{num_sequences}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Starting in {countdown}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Collecting Data", frame)
                cv2.waitKey(1000)

            # Collection phase
            sequence_data: List[np.ndarray] = []
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Webcam feed interrupted during collection.")
                features = extract_hand_features(frame)
                if features is not None:
                    sequence_data.append(features)
                else:
                    print(f"Warning: No hands detected in frame {frame_num+1}/{sequence_length}")
                cv2.putText(frame, f"Collecting {label}: {seq+1}/{num_sequences}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame {frame_num+1}/{sequence_length}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Collecting Data", frame)
                cv2.waitKey(1)

            # Save sequence if complete, warn if incomplete
            if len(sequence_data) == sequence_length:
                np.save(os.path.join(label_dir, f"{seq}.npy"), np.array(sequence_data))
                print(f"Saved sequence {seq+1}/{num_sequences}")
            else:
                print(f"Warning: Sequence {seq+1} incomplete ({len(sequence_data)}/{sequence_length} frames)")

    except Exception as e:
        print(f"Error during data collection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Data collection completed.")

if __name__ == "__main__":
    # Example usage
    collect_data("hello", "dataset", sequence_length=30, num_sequences=2)
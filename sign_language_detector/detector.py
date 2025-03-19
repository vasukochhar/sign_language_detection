import cv2
import numpy as np
from typing import Optional, Tuple
import threading
import time
from collections import Counter, deque
from tensorflow.keras.models import load_model
from .data_processing import extract_hand_features
from .utils import load_labels

class SignLanguageDetector:
    """Real-time sign language detection using MediaPipe and LSTM."""

    def __init__(self, model_path: Optional[str] = None, dataset_path: str = "dataset",
                 labels_path: str = "labels.json", sequence_length: int = 30, threshold: float = 0.7):
        """
        Initialize the detector.

        Args:
            model_path: Path to pre-trained model file. If None, requires training.
            dataset_path: Directory containing training data.
            labels_path: Path to JSON file with label mappings.
            sequence_length: Number of frames per gesture sequence.
            threshold: Confidence threshold for predictions.
        """
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.sequence = deque(maxlen=sequence_length)
        self.last_predictions = deque(maxlen=10)
        self.is_running = True
        self.current_prediction: Optional[Tuple[str, float]] = None
        self.prediction_lock = threading.Lock()
        self.prev_time = 0

        # Load labels and model
        self.labels = load_labels(labels_path)
        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = load_model(model_path) if model_path else None

    def prediction_worker(self) -> None:
        """Run predictions in a separate thread."""
        while self.is_running:
            if len(self.sequence) == self.sequence_length and self.model:
                sequence_array = np.array([list(self.sequence)])
                prediction = self.model.predict(sequence_array, verbose=0)[0]
                pred_idx = np.argmax(prediction)
                confidence = prediction[pred_idx]

                if confidence > self.threshold:
                    label = list(self.labels.keys())[pred_idx]
                    with self.prediction_lock:
                        self.current_prediction = (label, confidence)
            time.sleep(0.01)  # Prevent CPU overload

    def run(self) -> None:
        """Run real-time detection with webcam input."""
        if not self.model:
            raise ValueError("No model loaded. Train or load a model first.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam.")

        prediction_thread = threading.Thread(target=self.prediction_worker, daemon=True)
        prediction_thread.start()

        try:
            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract features and update sequence
                features = extract_hand_features(frame)
                if features is not None:
                    self.sequence.append(features)

                # Display prediction
                with self.prediction_lock:
                    pred = self.current_prediction
                if pred:
                    label, confidence = pred
                    self.last_predictions.append(label)
                    smoothed_label = Counter(self.last_predictions).most_common(1)[0][0]
                    cv2.putText(frame, f"{smoothed_label} ({confidence:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # FPS calculation
                curr_time = time.time()
                fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
                self.prev_time = curr_time
                cv2.putText(frame, f"FPS: {fps:.1f}",
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Sign Language Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.is_running = False
            prediction_thread.join()
            cap.release()
            cv2.destroyAllWindows()
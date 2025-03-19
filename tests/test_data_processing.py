import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from sign_language_detector.data_processing import extract_hand_features, collect_data

class TestDataProcessing(unittest.TestCase):
    @patch("sign_language_detector.data_processing.hands")
    def test_extract_hand_features_no_hands(self, mock_hands):
        """Test feature extraction returns None when no hands are detected."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_hands.process.return_value = MagicMock(multi_hand_landmarks=None)
        result = extract_hand_features(mock_frame)
        self.assertIsNone(result)

    @patch("sign_language_detector.data_processing.hands")
    @patch("sign_language_detector.data_processing.mp_drawing")
    def test_extract_hand_features_one_hand(self, mock_drawing, mock_hands):
        """Test feature extraction with one hand detected."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_landmark = MagicMock(x=0.1, y=0.2, z=0.3)
        mock_hands.process.return_value = MagicMock(multi_hand_landmarks=[MagicMock(landmark=[mock_landmark] * 21)])
        result = extract_hand_features(mock_frame)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 21 * 3 * 2)  # 21 landmarks * 3 coords * 2 hands (padded)
        self.assertTrue(np.all(result[63:] == 0))  # Second hand padded with zeros

    @patch("sign_language_detector.data_processing.cv2.VideoCapture")
    @patch("sign_language_detector.data_processing.extract_hand_features")
    def test_collect_data(self, mock_extract, mock_capture):
        """Test data collection saves sequences correctly."""
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8))] * 35  # 5 countdown + 30 frames
        mock_extract.return_value = np.ones(126)  # Mock features

        with patch("sign_language_detector.data_processing.np.save") as mock_save:
            collect_data("test_label", "test_dataset", sequence_length=5, num_sequences=1)
            mock_save.assert_called_once()

if __name__ == "__main__":
    unittest.main()
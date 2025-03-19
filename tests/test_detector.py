import unittest
from unittest.mock import patch, MagicMock
from sign_language_detector.detector import SignLanguageDetector

class TestSignLanguageDetector(unittest.TestCase):
    def test_init_no_model(self):
        """Test initialization without a model."""
        detector = SignLanguageDetector(model_path=None)
        self.assertIsNone(detector.model)
        self.assertEqual(detector.sequence_length, 30)
        self.assertEqual(detector.threshold, 0.7)

    def test_init_invalid_model_path(self):
        """Test initialization with a non-existent model path raises an error."""
        with self.assertRaises(FileNotFoundError):
            SignLanguageDetector(model_path="nonexistent.h5")

    @patch("sign_language_detector.detector.load_model")
    def test_init_with_model(self, mock_load_model):
        """Test initialization with a valid model path."""
        mock_load_model.return_value = MagicMock()  # Mock a loaded model
        detector = SignLanguageDetector(model_path="fake_model.h5")
        self.assertIsNotNone(detector.model)
        mock_load_model.assert_called_once_with("fake_model.h5")

    @patch("sign_language_detector.detector.cv2.VideoCapture")
    def test_run_no_model(self, mock_capture):
        """Test run() raises error if no model is loaded."""
        detector = SignLanguageDetector(model_path=None)
        with self.assertRaises(ValueError):
            detector.run()

if __name__ == "__main__":
    unittest.main()
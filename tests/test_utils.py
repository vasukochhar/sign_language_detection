import unittest
from unittest.mock import patch
import os
from sign_language_detector.utils import load_labels, save_labels

class TestUtils(unittest.TestCase):
    def test_load_labels_empty(self):
        """Test loading labels from non-existent file returns empty dict."""
        with patch("os.path.exists", return_value=False):
            labels = load_labels("nonexistent.json")
            self.assertEqual(labels, {})

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"hello": 0}')
    def test_load_labels_valid(self, mock_open):
        """Test loading valid labels from JSON."""
        with patch("os.path.exists", return_value=True):
            labels = load_labels("labels.json")
            self.assertEqual(labels, {"hello": 0})

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_save_labels(self, mock_open):
        """Test saving labels to JSON."""
        labels = {"hello": 0, "goodbye": 1}
        save_labels(labels, "labels.json")
        mock_open.assert_called_once_with("labels.json", "w")
        handle = mock_open()
        handle.write.assert_called_once()

if __name__ == "__main__":
    unittest.main()
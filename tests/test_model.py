import unittest
from sign_language_detector.model import create_model

class TestModel(unittest.TestCase):
    def test_create_model(self):
        """Test model creation with correct input/output shapes."""
        model = create_model(sequence_length=30, num_features=126, num_classes=10)
        self.assertEqual(model.input_shape, (None, 30, 126))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(len(model.layers), 7)  # LSTM, Dropout, LSTM, Dropout, Dense, Dropout, Dense

    def test_model_compilation(self):
        """Test model is compiled with correct loss and metrics."""
        model = create_model(sequence_length=30, num_features=126, num_classes=5)
        self.assertEqual(model.loss, "categorical_crossentropy")
        self.assertIn("categorical_accuracy", model.metrics_names)

if __name__ == "__main__":
    unittest.main()
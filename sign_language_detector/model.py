import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from .utils import load_labels

def create_model(sequence_length: int, num_features: int, num_classes: int) -> Sequential:
    """Create an LSTM model for sign language detection."""
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=(sequence_length, num_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def train_model(dataset_path: str, labels_path: str, sequence_length: int,
                epochs: int = 50, batch_size: int = 32, model_save_path: str = "model.h5") -> Sequential:
    """Train the model on collected data."""
    # Placeholder for loading dataset (move this to data_processing if complex)
    labels = load_labels(labels_path)
    num_classes = len(labels)

    # Mock dataset loading (replace with actual implementation)
    X = []  # Load from dataset_path
    y = []  # Load labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(sequence_length, 126, num_classes)
    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=10, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=callbacks)
    return model
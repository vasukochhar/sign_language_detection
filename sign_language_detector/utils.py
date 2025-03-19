import json
import os
from typing import Dict

def load_labels(labels_path: str) -> Dict[str, int]:
    """Load label mappings from a JSON file."""
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            return json.load(f)
    return {}

def save_labels(labels: Dict[str, int], labels_path: str) -> None:
    """Save label mappings to a JSON file."""
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=4)
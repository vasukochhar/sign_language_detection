import argparse
from sign_language_detector.detector import SignLanguageDetector
from sign_language_detector.data_processing import collect_data
from sign_language_detector.model import train_model

def main():
    """Command-line interface for the Sign Language Detection system."""
    parser = argparse.ArgumentParser(description="Sign Language Detection System")
    parser.add_argument('--mode', choices=['detect', 'collect', 'train'], default='detect')
    parser.add_argument('--model', default='model.h5', help="Path to model file")
    parser.add_argument('--dataset', default='dataset', help="Path to dataset directory")
    parser.add_argument('--labels', default='labels.json', help="Path to labels file")
    parser.add_argument('--label', help="Label for data collection")
    parser.add_argument('--sequences', type=int, default=30, help="Number of sequences to collect")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--threshold', type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()

    detector = SignLanguageDetector(
        model_path=args.model if args.mode == 'detect' else None,
        dataset_path=args.dataset,
        labels_path=args.labels,
        threshold=args.threshold
    )

    if args.mode == 'detect':
        detector.run()
    elif args.mode == 'collect':
        if not args.label:
            raise ValueError("Label required for data collection.")
        collect_data(args.label, args.dataset, detector.sequence_length, args.sequences)
    elif args.mode == 'train':
        train_model(args.dataset, args.labels, detector.sequence_length,
                    args.epochs, args.batch_size, args.model)

if __name__ == "__main__":
    main()
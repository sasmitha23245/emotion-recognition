
import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_model.h5')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    print(f"[INFO] Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    # Load test data
    X_test = np.load(os.path.join(args.output_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.output_dir, 'y_test.npy'))

    # Predict
    print("[INFO] Running predictions...")
    y_pred_probs = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Overall Accuracy: {acc * 100:.2f}%")



if __name__ == "__main__":
    main()

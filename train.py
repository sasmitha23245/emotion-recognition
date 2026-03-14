
import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import load_fer2013, split_dataset
from model import build_cnn_model, compile_model, get_callbacks, get_data_augmentation


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train FER CNN Model')
    parser.add_argument('--csv', type=str, default='fer2013.csv',
                        help='Path to fer2013.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='models')
    return parser.parse_args()



# Main Training Loop
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[INFO] GPUs available: {len(gpus)}")

    # ── Load & preprocess data ────────────────
    X, y = load_fer2013(args.csv)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # Add channel dimension for CNN (48, 48) → (48, 48, 1)
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # Save test set for evaluation
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)

    # ── Class weights (handle FER2013 imbalance) ─
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[INFO] Class weights: {class_weight_dict}")

    # ── Build & compile model ─────────────────
    model = build_cnn_model(input_shape=(48, 48, 1), num_classes=7)
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    # ── Data augmentation ────────────────────
    augmentation = get_data_augmentation()

    # Build augmented training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = (train_ds
                .shuffle(buffer_size=len(X_train))
                .batch(args.batch_size)
                .map(lambda x, y: (augmentation(x, training=True), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # ── Train ────────────────────────────────
    model_save_path = os.path.join(args.output_dir, 'best_model.h5')
    callbacks = get_callbacks(model_save_path)

    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # ── Final evaluation on test set ──────────
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(args.batch_size)

    print("\n[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n✅ Test Accuracy : {test_acc * 100:.2f}%")
    print(f"✅ Test Loss     : {test_loss:.4f}")

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.h5')
    model.save(final_path)
    print(f"[INFO] Final model saved to: {final_path}")


if __name__ == "__main__":
    main()

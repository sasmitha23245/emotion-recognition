

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


#  Show Preprocessing Pipeline Steps
def visualize_preprocessing_pipeline(raw_image, save_path=None):
    import cv2
    steps = {}

    # Step 0: Original
    img = raw_image.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps['1. Original'] = img

    # Step 1: Resize
    resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    steps['2. Resized (48x48)'] = resized

    # Step 2: Gaussian blur
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    steps['3. Gaussian Blur'] = blurred

    # Step 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)
    steps['4. CLAHE Equalized'] = equalized

    # Step 4: Gamma correction
    gamma = 1.2
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)], dtype=np.uint8)
    gamma_img = cv2.LUT(equalized, table)
    steps['5. Gamma Corrected'] = gamma_img

    # Step 5: Edge enhancement
    laplacian = cv2.Laplacian(gamma_img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    enhanced = cv2.addWeighted(gamma_img, 1.0, laplacian, 0.3, 0)
    steps['6. Edge Enhanced'] = enhanced

    # Step 6: Normalized
    normalized = (enhanced.astype(np.float32) / 255.0 * 255).astype(np.uint8)
    steps['7. Normalized'] = normalized

    # Plot
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    fig.suptitle('Image Preprocessing Pipeline', fontsize=14, fontweight='bold')

    for ax, (title, img_step) in zip(axes, steps.items()):
        ax.imshow(img_step, cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Preprocessing pipeline saved: {save_path}")
    else:
        plt.show()
    plt.close()


#  Dataset Class Distribution
def plot_class_distribution(y, title='FER2013 Emotion Class Distribution', save_path=None):
    counts = [np.sum(y == i) for i in range(7)]
    colors = plt.cm.Set3(np.linspace(0, 1, 7))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = axes[0].bar(EMOTION_LABELS, counts, color=colors, edgecolor='black')
    axes[0].set_title(title, fontsize=13)
    axes[0].set_ylabel('Number of Images')
    axes[0].set_xlabel('Emotion')
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     str(cnt), ha='center', fontsize=9)

    # Pie chart
    axes[1].pie(counts, labels=EMOTION_LABELS, colors=colors,
                autopct='%1.1f%%', startangle=140)
    axes[1].set_title('Proportion of Each Emotion Class')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Class distribution saved: {save_path}")
    else:
        plt.show()
    plt.close()


#  Sample Prediction Grid
def plot_sample_predictions(X_test, y_test, y_pred, n_samples=16, save_path=None):
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    cols = 4
    rows = n_samples // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    fig.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=13)

    for i, idx in enumerate(indices):
        ax = axes[i // cols][i % cols]
        img = X_test[idx].squeeze()
        true_label = EMOTION_LABELS[y_test[idx]]
        pred_label = EMOTION_LABELS[y_pred[idx]]
        correct = y_test[idx] == y_pred[idx]

        ax.imshow(img, cmap='gray')
        color = 'green' if correct else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                     color=color, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Sample predictions saved: {save_path}")
    else:
        plt.show()
    plt.close()


#  Grad-CAM Heatmap
def generate_gradcam(model, image, last_conv_layer_name='conv2d_6'):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    input_tensor = image[np.newaxis, ..., np.newaxis]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap, EMOTION_LABELS[pred_index.numpy()]


def plot_gradcam(model, image, save_path=None):
    try:
        last_conv = [l.name for l in model.layers if 'conv2d' in l.name][-1]
        heatmap, predicted_emotion = generate_gradcam(model, image, last_conv)
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        return

    # Resize heatmap to 48x48
    heatmap_resized = cv2.resize(heatmap, (48, 48))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    face_rgb = np.uint8(image * 255)
    face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(face_rgb, 0.5, heatmap_color, 0.5, 0)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle(f'Grad-CAM Visualization — Predicted: {predicted_emotion}', fontsize=13)

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Superimposed')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Grad-CAM saved: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Test with a synthetic image
    dummy = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    visualize_preprocessing_pipeline(dummy, save_path='preprocessing_pipeline.png')
    print("[PASS] Visualization module works.")

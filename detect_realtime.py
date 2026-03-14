
import argparse
import cv2
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion → color mapping (BGR)
EMOTION_COLORS = {
    'Angry':    (0,   0,   255),
    'Disgust':  (0,   128, 0  ),
    'Fear':     (128, 0,   128),
    'Happy':    (0,   255, 255),
    'Sad':      (255, 0,   0  ),
    'Surprise': (0,   165, 255),
    'Neutral':  (200, 200, 200),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_model.h5')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to static image. Omit for webcam mode.')
    parser.add_argument('--cascade', type=str,
                        default=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                        help='Haar Cascade XML path')
    return parser.parse_args()



# Face Detection (Viola-Jones / Haar Cascade)
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, gray



# Predict Emotion for a Single Face ROI
def predict_emotion(face_roi, model):
    processed = preprocess_image(face_roi, target_size=(48, 48))
    input_tensor = processed[np.newaxis, ..., np.newaxis]  # (1, 48, 48, 1)

    predictions = model.predict(input_tensor, verbose=0)[0]
    emotion_idx = np.argmax(predictions)
    confidence = predictions[emotion_idx]
    emotion = EMOTION_LABELS[emotion_idx]
    return emotion, confidence, predictions



# Draw Emotion Overlay on Frame
def draw_emotion(frame, x, y, w, h, emotion, confidence, predictions):
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Emotion label + confidence
    label = f"{emotion}: {confidence * 100:.1f}%"
    cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # Mini bar chart for all emotions (top-right overlay)
    bar_x = frame.shape[1] - 200
    bar_y_start = 10
    for i, (emo, prob) in enumerate(zip(EMOTION_LABELS, predictions)):
        bar_y = bar_y_start + i * 25
        bar_len = int(prob * 180)
        bar_color = EMOTION_COLORS.get(emo, (200, 200, 200))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 18),
                      bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 180, bar_y + 18),
                      (100, 100, 100), 1)
        cv2.putText(frame, f"{emo[:3]}: {prob * 100:.0f}%",
                    (bar_x - 60, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return frame



# Static Image Mode
def process_image(image_path, model, face_cascade):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    faces, gray = detect_faces(frame, face_cascade)

    if len(faces) == 0:
        print("[INFO] No faces detected.")
    else:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            emotion, confidence, predictions = predict_emotion(face_roi, model)
            print(f"[RESULT] Detected: {emotion} ({confidence * 100:.1f}%)")
            frame = draw_emotion(frame, x, y, w, h, emotion, confidence, predictions)

    cv2.imshow('Facial Emotion Recognition', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Webcam Real-Time Mode
def process_webcam(model, face_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[INFO] Real-time detection started. Press 'q' to quit, 's' to save frame.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces, gray = detect_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            emotion, confidence, predictions = predict_emotion(face_roi, model)
            frame = draw_emotion(frame, x, y, w, h, emotion, confidence, predictions)

        # Status bar
        status = f"Frame: {frame_count} | Faces: {len(faces)} | Press Q=Quit S=Save"
        cv2.putText(frame, status, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Facial Emotion Recognition — Real Time', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f'saved_frame_{frame_count}.jpg'
            cv2.imwrite(fname, frame)
            print(f"[INFO] Frame saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")



# Entry Point
def main():
    # Load model and cascade
    model_path = 'models/best_model.h5'
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    print(f"[INFO] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"[ERROR] Failed to load Haar Cascade from: {cascade_path}")
        return
    
    # Create GUI
    root = tk.Tk()
    root.title("Facial Emotion Recognition")
    root.geometry("800x500")
    root.resizable(False, False)

    # Create blurred blue-black gradient background image
    def create_gradient(width, height, start_color, end_color):
        img = Image.new('RGB', (width, height))
        for y in range(height):
            r = int(start_color[0] + (end_color[0] - start_color[0]) * (y / height))
            g = int(start_color[1] + (end_color[1] - start_color[1]) * (y / height))
            b = int(start_color[2] + (end_color[2] - start_color[2]) * (y / height))
            for x in range(width):
                img.putpixel((x, y), (r, g, b))
        return img

    img = create_gradient(800, 500, (10, 10, 40), (0, 0, 0))  # Dark blue/black gradient
    blurred = img.filter(ImageFilter.GaussianBlur(16))
    bg_image = ImageTk.PhotoImage(blurred)

    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Header
    title_label = tk.Label(root,
                           text="Emotion Recognition from facial expressions",
                           font=("Helvetica", 22, "bold"),
                           bg="#000000",
                           fg="#E4E9FF")
    title_label.place(relx=0.5, rely=0.18, anchor="center")

    subtitle_label = tk.Label(root,
                              text="Analyze facial expressions in real time or from images",
                              font=("Helvetica", 12),
                              bg="#000000",
                              fg="#C0C7FF")
    subtitle_label.place(relx=0.5, rely=0.27, anchor="center")

    # Buttons row
    button_frame = tk.Frame(root, bg="#000000")
    button_frame.place(relx=0.5, rely=0.55, anchor="center")

    def start_webcam():
        process_webcam(model, face_cascade)

    def browse_image():
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            process_image(file_path, model, face_cascade)

    button_style = {
        "width": 18,
        "height": 2,
        "bg": "#2F54EB",
        "fg": "white",
        "activebackground": "#1F3BB8",
        "activeforeground": "white",
        "bd": 0,
        "font": ("Helvetica", 11, "bold")
    }

    webcam_button = tk.Button(button_frame, text="Open Webcam", command=start_webcam, **button_style)
    webcam_button.pack(side="left", padx=12)

    browse_button = tk.Button(button_frame, text="Browse Image", command=browse_image, **button_style)
    browse_button.pack(side="left", padx=12)

    root.mainloop()


if __name__ == "__main__":
    main()

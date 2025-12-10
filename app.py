import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import tensorflow as tf
import os
import csv
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_DIR = r"C:\Users\USER\Desktop\DATASETS\XRAY"
MODEL_PATH = os.path.join(DATASET_DIR, "classifier_4class.h5")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
    raise e

CLASS_NAMES = ["COVID", "TB", "PNEUMONIA", "NORMAL"]
CLASS_COLORS = {
    "COVID": "#FF4C4C",       # Red
    "TB": "#FF9900",          # Orange
    "PNEUMONIA": "#FFD700",   # Yellow
    "NORMAL": "#4CAF50"       # Green
}
IMG_H, IMG_W = model.input_shape[1], model.input_shape[2]

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Cannot read image")
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img.astype(np.float32)/255.0
        img = np.expand_dims(img, 0)
        return img
    except Exception as e:
        messagebox.showerror("Image Error", f"Failed to load {image_path}\n{e}")
        return None

def predict_image(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return None, None
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx]*100

def overlay_prediction(image_path, label, confidence):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((300, 300))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{label} ({confidence:.2f}%)"
    draw.rectangle([0,0,img.width,25], fill=CLASS_COLORS.get(label, "black"))
    draw.text((5, 5), text, fill="white", font=font)
    return ImageTk.PhotoImage(img)

# -----------------------------
# GUI FUNCTIONS
# -----------------------------
def upload_predict_single():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    label, conf = predict_image(file_path)
    if label is None:
        return
    img_tk = overlay_prediction(file_path, label, conf)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    result_label.config(text=f"Prediction: {label}\nConfidence: {conf:.2f}%",
                        fg=CLASS_COLORS.get(label, "black"))

def batch_predict():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_paths:
        return
    results = []
    progress['value'] = 0
    progress['maximum'] = len(file_paths)
    listbox.delete(0, tk.END)
    save_folder = os.path.join(DATASET_DIR, "predictions")
    os.makedirs(save_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(save_folder, f"xray_predictions_{timestamp}.csv")
    for i, file_path in enumerate(file_paths, 1):
        label, conf = predict_image(file_path)
        if label is None:
            continue
        results.append([os.path.basename(file_path), label, f"{conf:.2f}%"])
        listbox.insert(tk.END, f"{os.path.basename(file_path)} → {label} ({conf:.2f}%)")
        listbox.itemconfig(i-1, {'fg': CLASS_COLORS.get(label, "black")})
        progress['value'] = i
        root.update_idletasks()
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Prediction", "Confidence"])
        writer.writerows(results)
    messagebox.showinfo("Batch Prediction", f"Predictions completed!\nSaved to: {csv_file}")

# -----------------------------
# BUILD GUI
# -----------------------------
root = tk.Tk()
root.title("🩺 AI Chest X-ray Diagnosis")
root.geometry("700x700")
root.resizable(False, False)
root.configure(bg="#f5f5f5")

# -----------------------------
# STYLE
# -----------------------------
style = ttk.Style()
# Blue text buttons
style.configure("BlueText.TButton",
                font=("Arial", 12),
                padding=6,
                foreground="#2196F3")  # Blue text
style.map("BlueText.TButton",
          foreground=[('active', '#1976D2')])  # Darker blue on hover
style.configure("TLabel", font=("Arial", 12), background="#f5f5f5")
style.configure("TProgressbar", thickness=20)

# Header
header_frame = tk.Frame(root, bg="#1976D2")
header_frame.pack(fill="x")
header = tk.Label(header_frame, text="🩺 AI Chest X-ray Diagnosis",
                  font=("Arial", 20, "bold"), bg="#1976D2", fg="white")
header.pack(pady=10)
subheader = tk.Label(root, text="Automatic X-ray analysis with deep learning",
                     font=("Arial", 12), fg="gray", bg="#f5f5f5")
subheader.pack(pady=5)

# Buttons with blue text
upload_btn = ttk.Button(root, text="Upload & Predict Single X-ray",
                        command=upload_predict_single, style="BlueText.TButton")
upload_btn.pack(pady=10)
batch_btn = ttk.Button(root, text="Batch Predict X-rays",
                       command=batch_predict, style="BlueText.TButton")
batch_btn.pack(pady=5)

# Image display
image_label = tk.Label(root, bg="#e0e0e0", width=300, height=300)
image_label.pack(pady=10)

# Prediction display
result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f5f5f5")
result_label.pack(pady=10)

# Listbox for batch predictions
listbox = tk.Listbox(root, width=90, height=10, bg="white", font=("Arial", 11))
listbox.pack(pady=10)

# Progress bar
progress = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
progress.pack(pady=10)

root.mainloop()
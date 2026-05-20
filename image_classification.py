import os
import cv2
import numpy as np
from data_collection import train_generator
from tensorflow.keras.models import load_model
from tkinter import filedialog

model = load_model('animal_classification_model.h5')

folder_path = filedialog.askdirectory(title="Select Folder with Images")

for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)
    frame = cv2.imread(img_path)

    resized_frame = cv2.resize(frame, (224, 224)) / 255.0
    prediction = model.predict(np.expand_dims(resized_frame, axis=0))

    animal_type = list(train_generator.class_indices.keys())[np.argmax(prediction)]
    print(f"Predicted animal: {animal_type}")

    save_path = f'./dataset/animals/{animal_type}/'
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{animal_type}_{len(os.listdir(save_path)) + 1}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Image saved to: {filename}")


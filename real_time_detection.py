import os
import cv2
from data_collection import train_generator
from tkinter import filedialog
from ultralytics import YOLO


model_yolo = YOLO('yolov8n.pt')

folder_path = filedialog.askdirectory(title="Select Folder with Images")

for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)
    frame = cv2.imread(img_path)

    results = model_yolo(frame)

    for result in results:
        annotated_frame = result.plot()

        cv2.imshow('Animal Detection', annotated_frame)

        animal_type = input("Enter detected animal: ").strip().lower()

        if animal_type in train_generator.class_indices:
            save_path = f'K:/Internship/tiger_detection/dataset/animals/{animal_type}/'
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, f"{animal_type}_{len(os.listdir(save_path)) + 1}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Image saved to: {filename}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

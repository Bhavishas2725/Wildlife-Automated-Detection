import os
from data_collection import train_generator, preprocess_image
import time
import pandas as pd


kdd_data = []
for animal, index in train_generator.class_indices.items():
    animal_folder = fr"C:\Users\bhavi\Documents\INTERN\animal_detection\dataset\animals/{animal}"
    for img in os.listdir(animal_folder):
        img_path = os.path.join(animal_folder, img)

        if os.path.isfile(img_path):
            processed_images = preprocess_image(img_path)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            kdd_data.append([img_path, animal, current_time] + processed_images[1:])

kdd_file = 'kdd_dataset.csv'
if os.path.exists(kdd_file):
    existing_kdd_df = pd.read_csv(kdd_file)
    kdd_df = pd.DataFrame(kdd_data, columns=['original_image', 'label', 'datetime', 'gray_image', 'blurred_image', 'edge_image', 'area', 'width', 'height'])
    kdd_df = pd.concat([existing_kdd_df, kdd_df], ignore_index=True)
else:
    kdd_df = pd.DataFrame(kdd_data, columns=['original_image', 'label', 'datetime', 'gray_image', 'blurred_image', 'edge_image', 'area', 'width', 'height'])

kdd_df.to_csv(kdd_file, index=False)
print("KDD dataset updated and saved as 'kdd_dataset.csv'")

from tensorflow.keras.models import load_model
from data_collection import train_generator
import os, cv2, numpy as np

model = load_model('animal_classification_model.h5')

def test_dataset(test_folder):
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        frame = cv2.imread(img_path)

        resized_frame = cv2.resize(frame, (224, 224)) / 255.0
        prediction = model.predict(np.expand_dims(resized_frame, axis=0))

        animal_type = list(train_generator.class_indices.keys())[np.argmax(prediction)]
        print(f"Predicted animal: {animal_type}")
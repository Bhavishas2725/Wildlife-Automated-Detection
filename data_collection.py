import os
import cv2
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return [image_path, None, None, None, 0, 0, 0]

    from PIL import Image
    with Image.open(image_path) as img:
        if img.mode in ('P', 'LA') or (img.mode == 'RGBA' and img.info.get('transparency')):
            print(f"Converting transparent image to RGBA: {image_path}")
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')

        image = np.array(img)

    if image is None:
        print(f"Failed to load image: {image_path}")
        return [image_path, None, None, None, 0, 0, 0]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    area = np.count_nonzero(edges)
    height, width = image.shape[:2]

    processed_folder = image_path.replace(r"C:\Users\bhavi\Documents\INTERN\animal_detection\dataset\animals", r"C:\Users\bhavi\Documents\INTERN\animal_detection\processed_dataset")
    os.makedirs(os.path.dirname(processed_folder), exist_ok=True)

    cv2.imwrite(processed_folder.replace(".jpg", "_gray.jpg"), gray)
    cv2.imwrite(processed_folder.replace(".jpg", "_blurred.jpg"), blurred)
    cv2.imwrite(processed_folder.replace(".jpg", "_edges.jpg"), edges)

    return [image_path, processed_folder.replace(".jpg", "_gray.jpg"), processed_folder.replace(".jpg", "_blurred.jpg"), processed_folder.replace(".jpg", "_edges.jpg"), area, width, height]

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    r"C:\Users\bhavi\Documents\INTERN\animal_detection\dataset\animals",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    r"C:\Users\bhavi\Documents\INTERN\animal_detection\dataset\animals",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Export for reuse in other files
__all__ = ['preprocess_image', 'train_generator', 'validation_generator']


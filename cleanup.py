import os
from PIL import Image

dataset_path = r"C:\Users\bhavi\Documents\INTERN\animal_detection\dataset\animals"

total = 0
removed = 0
converted = 0

for animal in os.listdir(dataset_path):
    animal_folder = os.path.join(dataset_path, animal)
    if not os.path.isdir(animal_folder):
        continue
    print(f"Cleaning {animal}/ folder...")
    for img_name in os.listdir(animal_folder):
        img_path = os.path.join(animal_folder, img_name)
        total += 1
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if not img_name.lower().endswith('.jpg'):
                    new_path = os.path.splitext(img_path)[0] + '.jpg'
                    img.save(new_path, 'JPEG', quality=90)
                    os.remove(img_path)
                    converted += 1
                else:
                    img.save(img_path, 'JPEG', quality=90)
        except Exception as e:
            print(f"Removing corrupt file: {img_name}")
            os.remove(img_path)
            removed += 1

print(f"Done! Total: {total} | Converted: {converted} | Removed: {removed}")
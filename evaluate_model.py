from data_collection import validation_generator
from tensorflow.keras.models import load_model
model = load_model('animal_classification_model.h5')

loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

model.summary()



import tensorflow as tf
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('flag_recognition_model.h5')

# Define a function to preprocess the new image data
def preprocess_image(image_path):
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Example usage: Make predictions on new images
image_paths = ['23.jpg', '49.jpg']

for image_path in image_paths:
    # Preprocess the image
    image = preprocess_image(image_path)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(image)
    class_index = np.argmax(predictions)
    classes = ['argentinean', 'uruguayan']  # Add the class names
    predicted_class = classes[class_index]

    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print()
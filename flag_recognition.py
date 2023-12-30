import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

# Apply data augmentation to increase dataset diversity
train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Change the output size to 2 for two classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 8

train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for multi-class classification
    classes=['argentinean', 'uruguayan']  # Add the class names
)

validation_generator = validation_data_gen.flow_from_directory(
    validation_data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for multi-class classification
    classes=['argentinean', 'uruguayan']  # Add the class names
)

epochs = 60

steps_per_epoch = math.ceil(train_generator.samples / batch_size)
validation_steps = math.ceil(validation_generator.samples / batch_size)

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

model.save('flag_recognition_model_update2.h5')
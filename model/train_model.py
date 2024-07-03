import tensorflow as tf
from tensorflow.keras import layers, models, optimizers 
from tensorflow.keras.callbacks import TensorBoard 
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from data.preprocess import get_preprocessed_data
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



def get_preprocessed_data():
# Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Image normalization: Convert pixel values ​​from 0-255 to 0-1
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert tags to one-hot encoding format
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = get_preprocessed_data()

# Definition of CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Model training
model.fit(train_images, train_labels, epochs=5, batch_size=64, 
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback])

# Model evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy: {test_acc:.2f}, Loss: {test_loss:.2f}')


# Increased data
datagen = ImageDataGenerator(
    rotation_range=10,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    shear_range=0.1,  
    zoom_range=0.1,  
    fill_mode='nearest'  
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Educational pipeline
train_generator = datagen.flow(train_images, train_labels, batch_size=32)
model.fit(train_generator, epochs=10, validation_data=(test_images, test_labels))

# Save the model in a suitable format for TensorFlow Serving
model.save('D:\MNIST_CNN\model\CNN_model.h5')

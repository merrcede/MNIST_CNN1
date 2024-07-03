import tensorflow as tf
from tensorflow.keras import layers, models, optimizers 
from tensorflow.keras.callbacks import TensorBoard 
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# تعریف مدل CNN
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

# کامپایل مدل
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# تنظیم TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# آموزش مدل
model.fit(train_images, train_labels, epochs=5, batch_size=64, 
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback])

# ارزیابی مدل
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy: {test_acc:.2f}, Loss: {test_loss:.2f}')


# افزایش داده
datagen = ImageDataGenerator(
    rotation_range=10,  # چرخش تصویر به اندازه 10 درجه
    width_shift_range=0.1,  # انتقال تصویر به اندازه 10% عرض تصویر
    height_shift_range=0.1,  # انتقال تصویر به اندازه 10% ارتفاع تصویر
    shear_range=0.1,  # برش تصویر
    zoom_range=0.1,  # بزرگنمایی تصویر
    fill_mode='nearest'  # پر کردن نقاط جدید با نزدیکترین رنگ
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# خط لوله آموزشی
train_generator = datagen.flow(train_images, train_labels, batch_size=32)
model.fit(train_generator, epochs=10, validation_data=(test_images, test_labels))

# ذخیره مدل در فرمت مناسب برای TensorFlow Serving
model.save('/path/to/mnist_model/1/', save_format='tf')

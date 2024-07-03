from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model(r'D:\MNIST_CNN\model\CNN_model.h5')

def log_prediction(image_data, prediction):
    with open('predictions.csv', 'a') as file:
        file.write(f"{image_data.tolist()},{prediction}\n")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    image = Image.open(file.stream).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array_reshaped  = image_array.reshape(1, 28, 28, 1) / 255.0

    prediction = model.predict(image_array_reshaped)
    predicted_class = np.argmax(prediction, axis=1)

# Log prediction along with the input image data
    log_prediction(image_array_reshaped, predicted_class[0])

    return jsonify({'prediction': int(predicted_class[0])})

if __name__ == '__main__':
    app.run(debug=True, port=5001)



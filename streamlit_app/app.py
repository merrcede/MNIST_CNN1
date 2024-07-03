# import streamlit as st
# import requests
# from PIL import Image
# import io

# st.title('MNIST Digit Recognition')
# uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('L')
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     # Prepare the image for sending to the Flask API
#     buf = io.BytesIO()
#     image.save(buf, format='PNG')
#     buf.seek(0)

#     url = 'http://localhost:5001/predict'
#     files = {'image': buf}
#     response = requests.post(url, files=files)
#     if response.status_code == 200:
#         prediction = response.json()['prediction']
#         st.write(f'Predicted digit: {prediction}')
#     else:
#         st.write('Failed to get prediction')


# import streamlit as st
# import requests
# from PIL import Image
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Load model
# model = load_model('path_to_your_model/mnist_model.h5')

# # Side bar for navigation
# option = st.sidebar.selectbox('Select an option', ('Prediction', 'Model Monitoring'))

# # Prediction page
# if option == 'Prediction':
#     st.header('MNIST Digit Prediction')
#     uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert('L')
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#         image = image.resize((28, 28))
#         image_array = np.array(image).reshape((1, 28, 28, 1)) / 255.0

#         # Make prediction
#         prediction = model.predict(image_array)
#         predicted_class = np.argmax(prediction, axis=1)
#         st.write(f'Predicted digit: {predicted_class[0]}')

# # Monitoring page
# elif option == 'Model Monitoring':
#     st.header('Model Monitoring Dashboard')
#     # Assuming 'data' is a pandas DataFrame containing historical performance metrics
#     # Load your data
#     # data = pd.read_csv('path_to_your_data.csv')

#     # Display key performance metrics
#     # st.write(data.describe())

#     # Example threshold check (you can elaborate based on your actual metrics)
#     accuracy = st.slider("Choose accuracy threshold", 0.0, 1.0, 0.95)
#     if data['accuracy'].iloc[-1] < accuracy:
#         st.error('Model accuracy is below the threshold!')

#     st.line_chart(data['accuracy'])


import streamlit as st
import requests
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model('D:\MNIST_CNN\model\CNN_model.h5')

# Function to load HTML
def load_html(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    else:
        return "فایل گزارش پیدا نشد."

# Side bar for navigation
option = st.sidebar.selectbox('Select an option', ('Prediction', 'Model Monitoring', 'Evidently Dashboard'))

# Prediction page
if option == 'Prediction':
    st.header('MNIST Digit Prediction')
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = image.resize((28, 28))
        image_array = np.array(image).reshape((1, 28, 28, 1)) / 255.0

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)
        st.write(f'Predicted digit: {predicted_class[0]}')

# Monitoring page
elif option == 'Model Monitoring':
    st.header('Model Monitoring Dashboard')
    # Load your data
    data = pd.read_csv('path_to_your_data.csv')  # Assuming you have this data
    # Display key performance metrics
    st.write(data.describe())
    # Example threshold check
    accuracy = st.slider("Choose accuracy threshold", 0.0, 1.0, 0.95)
    if data['accuracy'].iloc[-1] < accuracy:
        st.error('Model accuracy is below the threshold!')
    st.line_chart(data['accuracy'])

# Evidently Dashboard
elif option == 'Evidently Dashboard':
    st.header('Evidently Dashboard')
    # Load the HTML report generated by Evidently
    report_path = 'model_monitoring_report.html'
    html_content = load_html(report_path)
    st.markdown(html_content, unsafe_allow_html=True)

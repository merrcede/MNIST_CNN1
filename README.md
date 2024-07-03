# MNIST_CNN

# MNIST Handwritten Digit Recognition Project

# Project Description
The MNIST Handwritten Digit Recognition project is a complete system designed to identify digits 0 through 9 from handwritten images using TensorFlow, Flask, and Streamlit. This project includes training a Convolutional Neural Network (CNN) model, deploying this model in a Flask API, and developing a user interface with Streamlit to display and evaluate the model's results. Additionally, we use Evidently to monitor model's performance and data drift analysis, allowing us to create analytical dashboards for more precise monitoring.

# Prerequisites
To run this project, you need to install the following:
- Python 3.8+
- TensorFlow 2.x
- Flask
- Streamlit
- Evidently
- Pandas
- Numpy
- Pillow
- pylint
- pre-commit

You can install these dependencies by running the following command in your Python virtual environment:

Install all required dependencies:
`make instal`

# Project Structure
The project is organized into several main components within the following directories:

/my_mnist_project/
│
├── api/                    # Flask API directory
│   ├── api.py              # Flask application
│
├── model/                  # TensorFlow model directory
│   ├── train_model.py      # Script for training the model
│   ├── CNN_model.h5        # Saved model file
│
├── streamlit_app/          # Streamlit application directory
│   ├── app.py              # Streamlit script
│
├── monitoring/             # Model monitoring tools with Evidently
│   ├── monitor.py          # Script for generating Evidently dashboards
│
├── Makefile                # Makefile for command automation
│
└── README.md               # README file for project documentation

# Usage
Installing Make:
make is a useful tool for managing and automating commands defined in the Makefile. To install make on your system, follow these instructions:

# For Windows:
`choco install make`

# For macOS:
`brew install make`

# For Linux:
`sudo apt-get install build-essential`

# Using the Makefile
After installing make, you can run the commands defined in the Makefile from the root directory of the project. Here are some of the commands you can use:

Install all required dependencies:
`make instal`

Train the model:
`make train`

Run the Flask API:
`make serve`

Run the Streamlit dashboard:
`make dashboard`

Monitor the model with Evidently:
`make monitor`

Run Linter to check the quality of the code:
`make lint`

Run data preprocessing tests:
`make test_preprocess`

Run model-related tests:
`make test_model`

Clean temporary files:
`make clean`



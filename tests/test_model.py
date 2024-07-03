import unittest
import numpy as np
from tensorflow.keras.models import load_model

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loading the model for testing
        cls.model = load_model('D:\MNIST_CNN\model\CNN_model.h5')

    def test_model_prediction_shape(self):
        # Create a random data as input
        input_data = np.random.random((1, 28, 28, 1))
        result = self.model.predict(input_data)
        # Test that the output of the model has a suitable shape
        self.assertEqual(result.shape, (1, 10))

    def test_model_prediction_sum(self):
        # Ensure that the sum of the probabilities for a prediction is equal to 1
        input_data = np.random.random((1, 28, 28, 1))
        result = self.model.predict(input_data)
        self.assertAlmostEqual(np.sum(result), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()

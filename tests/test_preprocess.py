import unittest
import numpy as np
from model.train_model import get_preprocessed_data

class TestPreprocessing(unittest.TestCase):
    def test_data_shape(self):
        train_images, train_labels = get_preprocessed_data()
        # Data dimension test
        self.assertEqual(train_images.shape, (60000, 28, 28, 1))
        self.assertEqual(train_labels.shape, (60000, 10))

    def test_pixel_range(self):
        train_images, _ = get_preprocessed_data()
        # Test that all pixel values ​​are between 0 and 1
        self.assertTrue(np.all(train_images >= 0) and np.all(train_images <= 1))

    def test_one_hot_encoding(self):
        _, train_labels = get_preprocessed_data()
        # Test that tags are correctly in one-hot encoding format
        self.assertEqual(np.sum(train_labels[0]), 1)
        self.assertTrue(np.all((train_labels >= 0) & (train_labels <= 1)))

    def test_dtype(self):
        train_images, _ = get_preprocessed_data()
        # Data type test
        self.assertEqual(train_images.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()

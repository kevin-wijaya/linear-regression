from linear_regression import OLS
from .utils import generator_datasets
import unittest, numpy as np

class TestOLS(unittest.TestCase):
    def setUp(self):
        X, Y, self.actual_beta = generator_datasets()
        
        split = int(len(Y)*0.80)
        self.X_train, self.Y_train = X[0:split], Y[0:split]
        self.X_test, self.Y_test = X[split:-1], Y[split:-1]
        
        self.model = OLS()
        self.model.fit(self.X_train, self.Y_train)
        
    def test_fit(self):
        self.assertIsNotNone(self.model.beta_)
        
        np.testing.assert_array_almost_equal(self.model.beta_[1:], self.actual_beta, decimal=2)
        
    def test_predict(self):
        Y_pred = self.model.predict(self.X_test)
        
        self.assertEqual(Y_pred.shape, self.Y_test.shape)
        
        np.testing.assert_array_almost_equal(Y_pred, self.Y_test, decimal=0)

if __name__ == '__main__':
    unittest.main()
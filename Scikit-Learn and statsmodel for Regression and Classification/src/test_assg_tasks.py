import numpy as np
import pandas as pd
import sklearn
import statsmodels
#import unittest
from twisted.trial import unittest
from assg_tasks import task1_sklearn_linear_regression, task1_statsmodel_linear_regression
from assg_tasks import task2_label_encoding, task2_impute_missing_data
from assg_tasks import task2_sklearn_logistic_regression, task2_statsmodel_logistic_regression


class test_task1_sklearn_linear_regression(unittest.TestCase):
    
    def setUp(self):
        self.X = np.load('../data/regression_features.npy')
        self.y = np.load('../data/regression_labels.npy')
        self.model, self.intercept, self.slope, self.mse, self.rmse, self.rsquared = \
            task1_sklearn_linear_regression(self.X, self.y)

    def test_model(self):
        self.assertIsInstance(self.model, sklearn.linear_model._base.LinearRegression)
        self.assertTrue(self.model.get_params()['fit_intercept'])

    def test_intercept(self):
        self.assertAlmostEqual(self.intercept, 0.37578175021210747)
    
    def test_slope(self):
        self.assertAlmostEqual(self.slope, 0.3354845860060065)
    
    def test_mse(self):
        self.assertAlmostEqual(self.mse, 3.5473465427798607)
    
    def test_rmse(self):
        self.assertAlmostEqual(self.rmse, 1.8834400820784984)
    
    def test_rsquared(self):
        self.assertAlmostEqual(self.rsquared, 0.5008050204985712)


class test_task1_statsmodel_linear_regression(unittest.TestCase):
    
    def setUp(self):
        self.X = np.load('../data/regression_features.npy')
        self.y = np.load('../data/regression_labels.npy')
        self.model, self.intercept, self.slope, self.mse, self.rmse, self.rsquared = \
            task1_statsmodel_linear_regression(self.y, self.X)

    def test_model(self):
        self.assertIsInstance(self.model, statsmodels.regression.linear_model.RegressionResultsWrapper)
        self.assertEqual(self.model.k_constant, 1)
        self.assertEqual(self.model.nobs, 366)
        self.assertEqual(self.model.df_resid, 364)

    def test_intercept(self):
        self.assertAlmostEqual(self.intercept, 0.37578175021210747)
    
    def test_slope(self):
        self.assertAlmostEqual(self.slope, 0.3354845860060065)
    
    def test_mse(self):
        self.assertAlmostEqual(self.mse, 3.5473465427798607)

    def test_rmse(self):
        self.assertAlmostEqual(self.rmse, 1.8834400820784984)

    def test_rsquared(self):
        self.assertAlmostEqual(self.rsquared, 0.5008050204985712)


class test_task2_label_encoding(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.read_csv('../data/assg-02-weather.csv')
        self.y, self.ndim, self.shape, self.num_no, self.num_yes = \
            task2_label_encoding(self.df.RainTomorrow)

    def test_y(self):
        self.assertIsInstance(self.y, np.ndarray)
        #self.assertIsInstance(self.y.dtype, np.dtypes.Float64DType)

    def test_ndim(self):
        self.assertEqual(self.ndim, 1)

    def test_shape(self):
        self.assertEqual(self.shape, (366,))

    def test_num_no(self):
        self.assertEqual(self.num_no, 300)

    def test_num_yes(self):
        self.assertEqual(self.num_yes, 66)


class test_task2_impute_missing_data(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.read_csv('../data/assg-02-weather.csv')
        self.X, self.ndim, self.shape, self.columns, self.na_sum = \
            task2_impute_missing_data(self.df)

    def test_y(self):
        self.assertIsInstance(self.X, pd.DataFrame)

    def test_ndim(self):
        self.assertEqual(self.ndim, 2)

    def test_shape(self):
        self.assertEqual(self.shape, (366, 2))

    def test_columns(self):
        self.assertListEqual(list(self.columns), ['Sunshine', 'Pressure3pm'])
        pass

    def test_na_sum(self):
        series_dict = {'Sunshine': 0, 'Pressure3pm': 0}
        expected_series = pd.Series(data=series_dict, index=['Sunshine', 'Pressure3pm'])
        self.assertIsInstance(self.na_sum, pd.core.series.Series)
        self.assertTrue(self.na_sum.equals(expected_series))


class test_task2_sklearn_logistic_regression(unittest.TestCase):
    
    def setUp(self):
        df = pd.read_csv('../data/assg-02-weather.csv')
        X, _, _, _, _ = task2_impute_missing_data(df)
        y, _, _, _, _ = task2_label_encoding(df.RainTomorrow)
        self.model, self.intercept, self.slopes, self.accuracy = \
            task2_sklearn_logistic_regression(X, y)

    def test_model(self):
        self.assertIsInstance(self.model, sklearn.linear_model._logistic.LogisticRegression)
        # test that parameters are set as asked for for the logistic regression
        self.assertEqual(self.model.get_params()['C'], 500.0)
        self.assertEqual(self.model.get_params()['solver'], 'lbfgs')
        self.assertTrue(self.model.get_params()['fit_intercept'])

    def test_intercept(self):
        self.assertAlmostEqual(self.intercept, 186.590648, places=4)

    def test_slopes(self):
        self.assertIsInstance(self.slopes, np.ndarray)
        self.assertEqual(self.slopes.shape, (2,))
        self.assertAlmostEqual(self.slopes[0], -0.320885, places=4)
        self.assertAlmostEqual(self.slopes[1], -0.183120, places=4)

    def test_accuracy(self):
        self.assertAlmostEqual(self.accuracy, 0.863388, places=4)


class test_task2_statsmodel_logistic_regression(unittest.TestCase):
    
    def setUp(self):
        df = pd.read_csv('../data/assg-02-weather.csv')
        X, _, _, _, _ = task2_impute_missing_data(df)
        y, _, _, _, _ = task2_label_encoding(df.RainTomorrow)
        self.model, self.intercept, self.slopes, self.accuracy = \
            task2_statsmodel_logistic_regression(y, X)

    def test_model(self):
        self.assertIsInstance(self.model, statsmodels.discrete.discrete_model.BinaryResultsWrapper)
        self.assertEqual(self.model.k_constant, 1)
        self.assertEqual(self.model.nobs, 366)
        self.assertEqual(self.model.df_resid, 363)

    def test_intercept(self):
        self.assertAlmostEqual(self.intercept, 186.590648, places=2)

    def test_slopes(self):
        self.assertIsInstance(self.slopes, pd.core.series.Series)
        self.assertEqual(self.slopes.shape, (2,))
        self.assertAlmostEqual(self.slopes['Sunshine'], -0.320885, places=2)
        self.assertAlmostEqual(self.slopes['Pressure3pm'], -0.183120, places=2)

    def test_accuracy(self):
        self.assertAlmostEqual(self.accuracy, 0.863388, places=2)
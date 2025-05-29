import numpy as np
import pandas as pd
import sklearn
#import unittest
from twisted.trial import unittest
from assg_tasks import fibonacci_inefficient, fibonacci_efficient
from assg_tasks import task_2_numpy_operations
from assg_tasks import iterate_julia_set
from assg_tasks import task_4_dataframe_information, task_4_dataframe_mutate


class test_fibonacci_inefficient(unittest.TestCase):
    
    def setUp(self):
        pass
        
    def test_base_case_1(self):
        self.assertEqual(fibonacci_inefficient(1), 1)
        
    def test_base_case_2(self):
        self.assertEqual(fibonacci_inefficient(2), 2)

    def test_recursive_case_3(self):
        self.assertEqual(fibonacci_inefficient(3), 3)

    def test_recursive_case_4(self):
        self.assertEqual(fibonacci_inefficient(4), 5)

    def test_recursive_case_10(self):
        self.assertEqual(fibonacci_inefficient(10), 89)

    def test_recursive_case_37(self):
        self.assertEqual(fibonacci_inefficient(37), 39088169)


class test_fibonacci_efficient(unittest.TestCase):
    
    def setUp(self):
        pass
        
    def test_base_case_1(self):
        self.assertEqual(fibonacci_efficient(1), 1)
        
    def test_base_case_2(self):
        self.assertEqual(fibonacci_efficient(2), 2)

    def test_recursive_case_3(self):
        self.assertEqual(fibonacci_efficient(3), 3)

    def test_recursive_case_4(self):
        self.assertEqual(fibonacci_efficient(4), 5)

    def test_recursive_case_10(self):
        self.assertEqual(fibonacci_efficient(10), 89)

    def test_recursive_case_37(self):
        self.assertEqual(fibonacci_efficient(37), 39088169)


class test_task_2_numpy_operations(unittest.TestCase):

    def setUp(self):
        self.M, self.T, self.Z = task_2_numpy_operations()

    def test_M_results(self):
        self.assertEqual(self.M.shape, (4, 5))
        self.assertIsInstance(self.M.dtype, np.dtypes.BoolDType)
        expected_M = np.array(
            [[False,  True,  True,  True, False],
             [False,  True,  True,  True, False],
             [False,  True,  True,  True, False],
             [False,  True,  True,  True, False]])
        self.assertTrue(np.array_equal(self.M, expected_M))

    def test_T_results(self):
        self.assertEqual(self.T.shape, (4, 5))
        self.assertIsInstance(self.T.dtype, np.dtypes.Float64DType)
        expected_T = np.array(
            [[0., 1., 1., 1., 0.],
             [0., 1., 1., 1., 0.],
             [0., 1., 1., 1., 0.],
             [0., 1., 1., 1., 0.]])
        self.assertTrue(np.array_equal(self.T, expected_T))

    def test_Z_results(self):
        self.assertEqual(self.Z.shape, (4, 5))
        self.assertIsInstance(self.Z.dtype, np.dtypes.Complex128DType)
        expected_Z = np.array(
            [[-2.        -1.j        , -0.4       +2.6j       , -1.4       +0.6j       , -0.4       -1.4j       , 2.        -1.j        ],
             [-2.        -0.33333333j,  0.48888889+1.26666667j, -0.51111111+0.6j       ,  0.48888889-0.06666667j, 2.        -0.33333333j],
             [-2.        +0.33333333j,  0.48888889-0.06666667j, -0.51111111+0.6j       ,  0.48888889+1.26666667j, 2.        +0.33333333j],
             [-2.        +1.j        , -0.4       -1.4j       , -1.4       +0.6j       , -0.4       +2.6j       , 2.        +1.j        ]])
        self.assertTrue(np.allclose(self.Z, expected_Z))


class test_iterate_julia_set(unittest.TestCase):

    def setUp(self):
        cols = 480
        rows = 320
        scale = 300
        x = np.linspace(-cols / scale, cols / scale, num=cols).reshape(1, cols)
        y = np.linspace(-rows / scale, rows / scale, num=rows).reshape(rows, 1)
        self.Z = np.tile(x, (rows, 1)) + 1j * np.tile(y, (1, cols))

    def test_julia_set(self):
        # make a copy of Z, it should not change if first step is done correctly
        Z_copy = self.Z.copy()

        # get results from function
        T = iterate_julia_set(self.Z)

        self.assertTrue(np.array_equal(self.Z, Z_copy))
        self.assertFalse(self.Z is Z_copy)
        self.assertEqual(T.shape, self.Z.shape)
        self.assertIsInstance(T.dtype, np.dtypes.Float64DType)

        # Z is a bit big to test full array, so spot check
        self.assertEqual(T.min(), 0.0)
        self.assertEqual(T.max(), 255.0)
        self.assertEqual(sum(sum(T == 1.0)), 18948)
        self.assertEqual(sum(sum(T == 100.0)), 146)
        self.assertEqual(sum(sum(T == 200.0)), 24)
        self.assertEqual(sum(sum(T == 255.0)), 434)


class test_task_4_dataframe_information(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv('../data/assg-01-data.csv')
        self.jan_sales_sum, self.feb_sales_min, self.mar_sales_avg = task_4_dataframe_information(self.df)
        
    def test_jan_sales_sum(self):
        self.assertEqual(self.jan_sales_sum, 1462000)

    def test_feb_sales_min(self):
        self.assertEqual(self.feb_sales_min, 10000)

    def test_mar_sales_avg(self):
        self.assertEqual(self.mar_sales_avg, 47800.0)


class test_task_4_dataframe_mutate(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv('../data/assg-01-data.csv')
        self.new_df, self.num_missing_states, self.num_missing_zipcodes = task_4_dataframe_mutate(self.df)
        
    def test_num_missing_states(self):
        self.assertEqual(self.num_missing_states, 2)

    def test_num_missing_zipcodes(self):
        self.assertEqual(self.num_missing_zipcodes, 1)

    def test_new_df(self):
        columns = self.new_df.columns.to_list()
        expected_columns = ['account', 'name', 'street', 'city', 'state', 'zipcode', 'Jan', 'Feb', 'Mar', 'total']
        self.assertEqual(columns, expected_columns)

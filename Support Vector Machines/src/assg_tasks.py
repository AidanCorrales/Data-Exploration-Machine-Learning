import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def task_1_1_load_data():
    """Task 1 loads data from the `../data/assg-04-data1.csv` file.  This
    function should load the two features into a Pandas dataframe, and the
    y labels should be returned as a separate NumPy array.  Make sure that
    you correctly label the feature columns 'x_1' and 'x_2' respectively.
    This data file does not have a special row that specifies the feature /
    target labels, you need to correctly specify these when or after loading
    the data.

    Params
    ------

    Returns
    -------
    X - Returns the 2 features as a pandas dataframe, labeled 'x_1' and
        'x_2' respectively.  The shape should be (51,2) for this dataframe. 
    y - Returns the binary labels as a regular (51,) shaped Numpy 1-D vector.
    """
    # need to actually load the correct data file and return the training features
    # as X, a pandas data frame, and y, a numpy array of the binary labels
    #d = {'col1': [0, 1, 2, 3], 'col2': pd.Series([2, 3], index=[2, 3])}
    #X = pd.DataFrame(data=d, index=[0, 1, 2, 3])
    #y = np.array([0])

    df = pd.read_csv('../data/assg-04-data1.csv', header=None)

    X = df.iloc[:, :2]
    X.columns = ['x_1', 'x_2']

    y = df.iloc[:, 2].values

    return X, y


def task_1_2_linear_svm_classifier(X, y, C=1.0):
    """Task 1 part 2, create a linear classifier.  The features are expected
    to be passed in as a pandas dataframe `X`, and the binary labels in a numpy
    array `y`.  The `C` parameter controls the amount of regularization for
    a support vector classifier SVC as we discussed in class.  This parameter
    should be passed into and used when creating the SVC instance.

    You are expected to create and return a fitted pipeline from this function
    on the given data using the given C parameter.  The pipeline should first
    use a standard scalar to scale the input feature data.  Then a SVC
    classifier should be created, using the C parameter and a `linear` kernel.
    The resulting fitted pipeline / model should be returned from this function.

    Params
    ------
    X - A dataframe loaded with the task 1 data containing two features x_1 and
        x_2
    y - The binary classification labels for task 1 as a numpy array of 0 / 1
        values
    C - The regularization parameter to use for the SVC pipeline that will be
        created, defaults to C=1.0

    Returns
    -------
    pipeline - Returns a sklearn pipeline that contains a standard scalar that
      feeds into a SVC classifier using a linear kernel and the indicated C
      regularization parameter.
    """
    # create a pipeline to scale the data and fit a SVC classifier using a 
    # linear kernel.  Make sure you use the passed in C parameter when
    # creating your model

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=C))
    ])

    model.fit(X, y)

    return model

def gaussian_kernel(xi, xj, sigma):
    """ Define gaussian kernel function.  Given two separate points xi and xj, calculate
    the gaussian kernel similarity.  The sigma parameter controls the width of the gaussian kernel
    similarity.
    
    Paramters
    ---------
    xi, xj - Numpy vectors of of 2 n-dimensional points.  Both vectors must be (n,) shaped (of same size
      and shape).
    sigma - meta parameter to control the width of the gaussian kernel (e.g. the standard deviation of the
      gaussian distribution being used.
      
    Returns
    -------
    K_gaussian - returns the gaussian kernel similarity measure of the distance between the 2 points.
    """
    # implement the described gaussian kernel / similarity function here
    distance_sq = np.sum((xi - xj) ** 2)

    similarity = np.exp(-distance_sq / (2 * sigma ** 2))

    return similarity


def task_3_1_load_data():
    """Task 1 loads data from the `../data/assg-04-data2.csv` file.  This
    function should load the two features into a Pandas dataframe, and the
    y labels should be returned as a separate NumPy array.  Make sure that
    you correctly label the feature columns 'x_1' and 'x_2' respectively.
    This data file does not have a special row that specifies the feature /
    target labels, you need to correctly specify these when or after loading
    the data.

    Params
    ------

    Returns
    -------
    X - Returns the 2 features as a pandas dataframe, labeled 'x_1' and
        'x_2' respectively.  The shape should be (51,2) for this dataframe. 
    y - Returns the binary labels as a regular (51,) shaped Numpy 1-D vector.
    """
    # need to actually load the correct data file and return the training features
    # as X, a pandas data frame, and y, a numpy array of the binary labels
    df = pd.read_csv('../data/assg-04-data2.csv', header=None)

    X = df.iloc[:, :-1]
    X.columns = ['x_1', 'x_2']

    y = df.iloc[:, -1]
    y = y.to_numpy()

    return X, y


def task_3_2_rbf_svm_classifier(X, y, kernel='rbf', C=1.0, gamma=8.0):
    """Task 3 part 2, create a SVC classifier using a nonlinear `rbf` kernel.
    The features are expected to be passed in as a pandas dataframe `X`,
    and the binary labels in a numpy array `y`.  The `C` parameter controls
    the amount of regularization for a support vector classifier SVC as
    we discussed in class.  Likewise the `gamma` parameter controls the
    shape of the `rbf` kernel used in this function.  These parameters
    should be passed into and used when creating the SVC instance.

    You are expected to create and return a fitted pipeline from this function
    on the given data using the given C parameter.  The pipeline should first
    use a standard scalar to scale the input feature data.  Then a SVC
    classifier should be created, using the C parameter and a `linear` kernel.
    The resulting fitted pipeline / model should be returned from this function.

    Params
    ------
    X - A dataframe loaded with the task 1 data containing two features x_1 and
        x_2
    y - The binary classification labels for task 1 as a numpy array of 0 / 1
        values
    C - The regularization parameter to use for the SVC pipeline that will be
        created, defaults to C=1.0
    gamma - The "spread" of the radial basis kernel function to use.

    Returns
    -------
    pipeline - Returns a sklearn pipeline that contains a standard scalar that
      feeds into a SVC classifier using a nonlinear rbf kernel and the indicated C
      and gamma regularization parameters.
    """
    # create a pipeline to scale the given data and then fit a SVC support
    # vector machine classifier to the data.  You need to use the
    # specified kernel for your SVC classifier, as well as the given
    # C and gamma parameters.

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=C, gamma=gamma))
    ])

    pipeline.fit(X, y)

    return pipeline

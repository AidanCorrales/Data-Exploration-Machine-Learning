import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fibonacci_inefficient(n):
    """Calculate the nth Fibonacci number of the Fibonacci sequence using
    recursion.  The base cases of the sequence are defined as:
    
    fib(1) = 1
    fib(2) = 2
    
    And the recursive case is that
    
    fib(n) = fib(n - 1) + fib(n - 2)
    
    Paramters
    ---------
    n - We are to calculate the nth number of the Fibonacci sequence and return it.
    
    Returns
    -------
    fib(n) - Returns the calculated nth Fibonacci number of the Fibonacci sequence.

    Tests
    -----
    # base cases
    >>> fibonacci_inefficient(1)
    1
    >>> fibonacci_inefficient(2)
    2

    # 3 and 4 require recursive cases
    >>> fibonacci_inefficient(3)
    3
    >>> fibonacci_inefficient(4)
    5

    # some more complex, and inefficient cases
    >>> fibonacci_inefficient(10)
    89
    >>> fibonacci_inefficient(37)
    39088169

    """
    if n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        return fibonacci_inefficient(n - 1) + fibonacci_inefficient(n - 2)


# You need to create a dictionary here as a global and initialize the base
# cases for the 1st and 2nd fibonacci numbers

fib_dict = {1: 1, 2: 2}

def fibonacci_efficient(n):
    """Calculate the nth Fibonacci number of the Fibonacci sequence using
    recursion and memoization.  The base cases of the sequence are
    defined as:

    fib(1) = 1
    fib(2) = 2

    And the recursive case is that

    fib(n) = fib(n - 1) + fib(n - 2)

    But this version should use memoization instead of the inefficient double
    recursion, so it should be much faster than the given inefficient version.

    Paramters
    ---------
    n - We are to calculate the nth number of the Fibonacci sequence and
        return it.
    fib_dict - An external dictionary of cached previously computed results.  By default, the
        base cases of fib(1) = 1 and fib(2) = 2 are initialized for the
        dictionary, so that the first call to this function can be simply
        done as:
             fibonacci_efficient(10)

    Returns
    -------
    fib(n) - Returns the calculated nth Fibonacci number of the 
        Fibonacci sequence.

    Tests
    -----
    # base cases
    >>> fibonacci_efficient(1)
    1
    >>> fibonacci_efficient(2)
    2

    # 3 and 4 require recursive cases
    >>> fibonacci_efficient(3)
    3
    >>> fibonacci_efficient(4)
    5

    # some more complex, and inefficient cases
    >>> fibonacci_efficient(10)
    89
    >>> fibonacci_efficient(37)
    39088169

    """
    
    # your task 1 implementaiton of memoized fibonacci goes here
    if n in fib_dict:
        return fib_dict[n]
    
    fib_dict[n] = fibonacci_efficient(n - 1) + fibonacci_efficient(n - 2)
    
    return fib_dict[n]


def task_2_numpy_operations():
    """This function tests your work for task 2.  Copy the code (no print statements
    or other output) you did and put it in this function.  Return the M, T and Z
    final matrices that you created as the result from this function

    Parameters
    ----------
    
    Returns
    -------
    M - The boolean Mask. A (4,5) shaped numpy array of boolean values
    T - The Time stamp index. A (4, 5) shaped numpy array of floats with
        1.0 values where the mask M was True.
    Z - The julia set complex values.  Result of vectorized calculation Z = Z^2 + C
        only updated where the mask M is True.
    """
    # your task 2 implementation goes here, remove the stub code and correctly
    # create M, T and Z
    x = np.linspace(-2.0, 2.0, 5)
    x = x.reshape(1, 5)
    print(x)

    y = np.linspace(-1, 1, 4)
    y = y.reshape(4, 1)
    print(y)

    rows = 4
    X = np.tile(x, (rows,1))
    print(X)
    print(X.shape)

    column = 5
    Y = np.tile(y, (1,column))
    print(Y)
    print(Y.shape)

    Y = Y * (0 + 1j)
    print(Y)
    print(Y.shape)

    Z = X + Y
    print(Z)
    print(Z.shape)

    T = np.zeros((4,5))
    print(T)
    print(T.shape)

    M = np.ones((4, 5), dtype=bool)
    print(M)
    print(M.shape)
    print(M.dtype)

    C = np.full((4,5),-0.4 + 0.6j)
    print(C)
    print(C.shape)
    print(C.dtype)

    np.abs(Z)

    M = np.abs(Z) <= 2
    print(M)

    T[M] = 1
    print(T)

    Z[M] = Z[M]**2 + C[M]
    print(Z)

    #M = np.zeros((2,2))
    #T = np.zeros((2,2))
    #Z = np.zeros((2,2))
    return M, T, Z


def iterate_julia_set(Z, c=-0.4+0.6j, num_iters=256):
    """Iterate the array of complex numbers Z a number of times, updating them using the 
    quadratic polynomial to calculate the Julia set and Julia fractals.
    
    Parameters
    ----------
    Z - A 2d NumPy array of complex numbers.  Should be a tiled grid of real+complex parts 
        linearly spaces over some area we want to calculate the julia set for.
    c - A complex number, the constant to be added to each value on each iteration for numbers
        still in the julia set.  Defaults to c=-0.4+0.6j
    num_iters - Number of iterations/updates to perform of the Z, M and T matrices.  Defaults to
        performing 256 iterations
        
    Returns
    -------
    T - Returns a NumPy array of the same shape as the input Z. T contains the time step/stamp
        of when each point in Z fell out of the julia set during iterations.
    """
    # task 3 implementation goes here.  Remove the stub code and implement the described function
    #T = np.zeros((2,2))
    Zcopy = np.copy(Z)

    C = np.full(Zcopy.shape, c, dtype=complex)
    M = np.abs(Zcopy) <= 2
    T = np.zeros(Zcopy.shape, dtype=np.float64)

    for time in range(1, num_iters + 1):
        Zcopy[M] = Zcopy[M]**2 + C[M]
        M = np.abs(Zcopy) <= 2
        T[M] = time - 1
    
    return T


def task_4_dataframe_information(df):
    """Given the dataframe you should have read in for Task 4, find and
    return the following information:
        - The total sum of sales in the month of Jan
        - The minimum sales amount in the month of Feb
        - The average (mean) sales for the month of Mar
    You are required to use Pandas dataframe operations to find and return this
    information.

    Parameters
    ----------
    df - A Pandas DataFrame with data read from the assignment 01 csv file.

    Returns
    -------
    jan_sales_sum - The sum of the sales for the month of Jan
    feb_sales_min - The minimum sales amount in Feb
    mar_sales_avg - The average of the sales for Mar
    """
    # task 4 implementation of getting dataframe information goes here
    df = pd.read_csv('../data/assg-01-data.csv')

    janTotal = df['Jan'].sum()
    febMin = df['Feb'].min()
    marAve = df['Mar'].mean()

    return janTotal, febMin, marAve


def task_4_dataframe_mutate(df):
    """Given the dataframe you should have read in for Task 4, change the name
    of the column holding the zip code information for the sales data.  Then
    determine the number of missing values in the states and zip codes and return them.
    The function should create a new dataframe with the renamed zip code column,
    and return all of the required information

    Parameters
    ----------
    df - A Pandas DataFrame with data read from the assignment 01 csv file.

    Returns
    -------
    new_df - The mutated dataframe with the newly named zip code column
    num_missing_states - The number of missing items in the states feature
    num_missing_zipcodes - The number of missing items in the newly named zipcode feature
    """
    # task 4 implementaiton of mutating dataframe and searching for missing values goes here
    df = pd.read_csv('../data/assg-01-data.csv')
    new_df = pd.DataFrame()
    new_df = df

    new_df = new_df.rename(columns={'postal-code':'zipcode'})

    new_df['total'] = new_df[['Jan', 'Feb', 'Mar']].sum(axis=1)

    num_missing_states = new_df['state'].isna().sum()
    num_missing_zipcodes = new_df['zipcode'].isna().sum()

    return new_df, num_missing_states, num_missing_zipcodes

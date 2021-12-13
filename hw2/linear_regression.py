import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def linear_regression(x=None, y=None, data_set=None):
    # Reading the data using CSV file path and pandas
    data = pd.read_csv(data_set, ',')
    print(data.head())

    # Drop the NaN values from the Data
    A = data
    A.dropna()

    matrix = np.array(A.values, 'float')

    # Assign input and target variable
    X = matrix[:, 0]
    y = matrix[:, 1]

    # calculate 95% credible interval
    credible_intervals = st.t.interval(alpha=0.95, df=len(X) - 1, loc=np.mean(X), scale=st.sem(X))
    # print(credible_intervals)

    # feature normalization
    # input variable divided by maximum value among input values in X
    X = X / (np.max(X))

    # Plot the data
    plt.plot(X, y, 'bo')
    plt.ylabel('sqft living')
    plt.xlabel('Price')
    plt.title('Price Vs. sqft living')
    plt.grid()
    plt.show()

    def computecost(x, y, theta):
        a = 1 / (2 * m)
        b = np.sum(((x @ theta) - y) ** 2)
        j = a * b
        return j

    # initialising parameter
    m = np.size(y)
    X = X.reshape([len(X), 1])
    x = np.hstack([np.ones_like(X), X])
    y_test = y.reshape([len(X), 1])

    theta = np.zeros([2, 1])

    # print(theta, '\n', m)

    # print(computecost(x, y, theta))

    def gradient(x, y, theta):
        alpha = 0.00001
        iteration = 20000
        # gradient descend algorithm
        J_history = np.zeros([iteration, 1])
        for iter in range(0, 2000):
            error = (x @ theta) - y
            temp0 = theta[0] - ((alpha / m) * np.sum(error * x[:, 0]))
            temp1 = theta[1] - ((alpha / m) * np.sum(error * x[:, 1]))
            theta = np.array([temp0, temp1]).reshape(2, 1)
            J_history[iter] = (1 / (2 * m)) * (np.sum(((x @ theta) - y) ** 2))  # compute J value for each iteration
        return theta, J_history

    theta, J = gradient(x, y, theta)
    # print(theta)

    theta, J = gradient(x, y, theta)
    # print(J)

    regression_estimates = x @ theta
    # error = [1, X] @ theta
    # print(error - y)
    # print(regression_estimates.T)
    standard_errors = regression_estimates - y_test
    print(standard_errors)
    # print(standard_errors)
    # standard_errors = standard_errors[:, 0]
    # print(x @ theta)

    # plot linear fit for our theta
    plt.plot(X, y, 'bo')
    plt.plot(X, x @ theta, 'r-')
    plt.ylabel('sqft living')
    plt.xlabel('Price')
    plt.title('Price Vs. sqft living')
    plt.legend(['Data Points', 'LinearFit'])
    plt.grid()
    plt.show()

    return regression_estimates, standard_errors, credible_intervals

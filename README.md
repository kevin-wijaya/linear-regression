# Linear Regression From Scratch with Numpy

## Table of Contents
+ [Overview](#overview)
+ [Dependencies](#dependencies)
+ [Installation](#installation)
+ [Examples](#examples)
+ [Comparison](#comparison)

## Overview <a name = "overview"></a>

A module for solving linear regression problems. This module includes two methods for linear regression: Ordinary Least Squares (OLS) and Gradient Descent with Batch Optimization.

## Dependencies <a name = "dependencies"></a>
This project requires Python ^3.10

- Numpy

## Installation

Install from GitHub
``` sh
pip install git+https://github.com/kevin-wijaya/linear-regression.git
```

Install from Poetry
``` sh
poe add git+https://github.com/kevin-wijaya/linear-regression.git
```

After the installation, you can import the included models:
``` python
from linear_regression import OLS, GD
```

## Examples

Generate and split synthetic dataset
``` python
# import library
from sklearn.datasets import make_regression

# create random dataset 
X, Y, true_coefficients = make_regression(n_samples=1000000, n_features=10, noise=0.1, coef=True, random_state=42)

# split train and test (80%)
split_idx = int(len(Y)*0.80)
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_test, Y_test = X[split_idx:], Y[split_idx:]
```

Modelling linear regression with Ordinary Least Squares (OLS)
``` python
# import library
from linear_regression import OLS
from sklearn.metrics import r2_score

# train and evaluate
regressor = OLS()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
print(r2_score(Y_pred, Y_test)) # output: 0.9999994994440174
```

Modelling linear regression with Gradient Descent
``` python
# import library
from linear_regression import GD
from sklearn.metrics import r2_score

# train and evaluate
regressor = GD()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
print(r2_score(Y_pred, Y_test)) # output: 0.9987232082831907
```

## Comparison with `sklearn.linear_model.linearRegression` <a name="comparison"></a>

Helper Functions
```python
# helper function to measure execution time
def time_exec(func):
    def wrapper(*args, **kwargs):
        start = time()  
        result = func(*args, **kwargs)  
        duration = time() - start
        print(f"\ntime exec: '{func.__name__}' took {duration:.4f} seconds\n")
        return duration
    return wrapper

# helper function to print model results
def _print(coefficient, intercept, predicted, actual):
    print(f'Coefficient: {coefficient}')
    print(f'Intercept: {intercept}')
    print(f'Predicted(2): {predicted}')
    print(f'Actual(2): {actual}')

@time_exec
def own_model(regressor, X_train, Y_train, X_test, Y_test):
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)
    _print(regressor.beta_[1:], regressor.beta_[0], y_pred[:2], Y_test[:2])

@time_exec  
def sklearn_model(regressor, X_train, Y_train, X_test, Y_test):
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)
    _print(regressor.coef_, regressor.intercept_, y_pred[:2], Y_test[:2])
```

Execution and Timing Collection
``` python
hist_own = []
hist_sklearn = []

for i in range(10, 120, 10):
    print(i, 100000*i)
    X, Y = make_regression(n_samples=100000*i, n_features=10, noise=0.1, random_state=42)
    hist_own.append(own_model(OLS(), X_train, Y_train, X_test, Y_test))
    hist_sklearn.append(sklearn_model(LinearRegression(), X_train, Y_train, X_test, Y_test))
```

Plot Execution Time Comparison
``` python
import matplotlib.pyplot as plt

size_data = [100000*i for i in range(10, 120, 10)]

plt.plot(size_data, hist_own, color='red', marker='o', linestyle='-')
plt.plot(size_data, hist_sklearn, color='blue', marker='o', linestyle='-')

plt.xticks(ticks=size_data, labels=[f'{size:,}' for size in size_data], rotation=-45)
plt.legend(['Linear Regression (from scratch)', 'Linear Regression (with sklearn)'], loc='upper right') 
plt.ylabel('Seconds')
plt.xlabel('Number of Samples')
plt.title('Execution Time Comparison: From Scratch vs. Scikit-Learn')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()
```

Cell Output
```
10 1000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1602 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.5375 seconds

20 2000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1500 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.5645 seconds

30 3000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.3026 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.4915 seconds

40 4000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1450 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 1.1230 seconds

50 5000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1650 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.5353 seconds

60 6000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1980 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.4870 seconds

70 7000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1680 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.5130 seconds

80 8000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1430 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.4470 seconds

90 9000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1740 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.5080 seconds

100 10000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.2370 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.7560 seconds

110 11000000
Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.0001798655741094904
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'own_model' took 0.1507 seconds

Coefficient: [ 9.40331553 35.67578749 56.91149548 21.18130738 21.69260065 48.84513374
 49.31999333 46.47708275 68.17339005 53.89662291]
Intercept: -0.00017986557410718668
Predicted(2): [-143.45936234 -189.45884628]
Actual(2): [-143.56219175 -189.30183472]

time exec: 'sklearn_model' took 0.5890 seconds
```

Plot Result
![plot](https://github.com/kevin-wijaya/resources/raw/main/images/linear-regression/output.png)
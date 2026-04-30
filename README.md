# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Initialize Parameters
Set weights w = [0, 0, ..., 0] (size = number of features)
Set bias b = 0
Repeat for each epoch (E times):

For each training example (xi, yi):

a. Prediction

y_pred = w · xi + b

b. Compute Error

error = y_pred - yi

c. Update Weights

w = w - α * error * xi

d. Update Bias

b = b - α * error
End Loop
Return Final Parameters
Return updated w and b

## Program:
```
# Code cell
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pandas as pd

data = pd.read_csv("california_housing_train.csv")

X = data.iloc[:, :3]
Y = data.iloc[:, [8, 6]]  # adjust columns as needed

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X.iloc[0])
print("Example Y (first row):", Y.iloc[0])

# Code cell
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes: ", X_test.shape, Y_test.shape)

# Code cell
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit on training data and transform both train and test
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))

# Code cell
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # you can also set alpha, eta0, penalty etc.
multi_output_sgd = MultiOutputRegressor(sgd)

# Fit on scaled training data
multi_output_sgd.fit(X_train_scaled, Y_train_scaled)


Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)   
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)          
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)     

print("First 5 predictions (original units):")
print(Y_pred[:5])

# Code cell
mse = mean_squared_error(Y_test_orig, Y_pred)
print("Mean Squared Error (multi-output):", mse)

# Per-output MSE (optional, helpful for debugging)
mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)


for i in range(5):
    print(f"Example {i+1}")
    print("Inputs (raw):", X_test.iloc[i])     
    print("True outputs:", Y_test_orig[i])     
    print("Predicted:", Y_pred[i])

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = fetch_california_housing()
X, y = data.data[:, :3], data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# SGD Regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

# Compare
print("LinearRegression MSE:", mean_squared_error(y_test, lr_pred))
print("SGDRegressor MSE:", mean_squared_error(y_test, sgd_pred))



```

## Output:

<img width="665" height="256" alt="image" src="https://github.com/user-attachments/assets/cdb8e1b3-b28a-48cb-b9c0-daedfcbc4266" />

<img width="421" height="86" alt="image" src="https://github.com/user-attachments/assets/2d20345c-bf74-4308-a1e5-ce19e728ecc6" />

<img width="796" height="88" alt="image" src="https://github.com/user-attachments/assets/6f3cc5f6-f929-421d-a41e-4480cc8adf99" />

<img width="525" height="177" alt="image" src="https://github.com/user-attachments/assets/b8941902-9ef4-4d5a-8f73-171ea75b0bcc" />

<img width="621" height="84" alt="image" src="https://github.com/user-attachments/assets/3e6988b2-28ab-4238-82d1-891f4036bf53" />

<img width="766" height="723" alt="image" src="https://github.com/user-attachments/assets/2eaa9c2d-7b43-4a64-9d86-4eb0d970f3a3" />

<img width="487" height="84" alt="image" src="https://github.com/user-attachments/assets/8637d028-7e53-4474-89d4-f2a43e0f3cb7" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

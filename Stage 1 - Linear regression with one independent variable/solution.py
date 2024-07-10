import os
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# Checking if the ../Data directory exists
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Downloading data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# Reading the data
data = pd.read_csv('../Data/data.csv')

# Prepare predictor and target variables
X = data[['rating']]
y = data['salary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the salary on test data
y_pred = model.predict(X_test)

# Calculate the MAPE
mape_value = mean_absolute_percentage_error(y_test, y_pred)

# Get the model parameters
intercept = model.intercept_
slope = model.coef_[0]

# Print the results
print(f"{intercept:.5f} {slope:.5f} {mape_value:.5f}")

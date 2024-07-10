import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    # Check if the../Data directory exists
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Downloading data if it is unavailable
    if 'data.csv' not in os.listdir('../Data'):
        url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/data.csv', 'wb').write(r.content)

    # 1. Read the data
    data = pd.read_csv('../Data/data.csv')

    # 2. Load the data
    # (Already done by reading the CSV)

    # 3. Make x and y
    X = data.drop('salary', axis=1)
    y = data['salary']

    # 4. Split the predictors and target into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # 5. Fit the model predicting salary based on all other variables
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Print the model coefficients separated by a comma
    print(', '.join(str(c) for c in model.coef_))


if __name__ == "__main__":
    main()

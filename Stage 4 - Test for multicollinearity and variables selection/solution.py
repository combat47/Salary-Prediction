import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


def main():
    # Check if the '../Data' directory exists
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Downloading data if it is unavailable
    if 'data.csv' not in os.listdir('../Data'):
        url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/data.csv', 'wb').write(r.content)

    # 1. Read the data
    data = pd.read_csv('../Data/data.csv')

    # 2. Calculate correlation matrix
    corr_matrix = data.corr()

    # 3. Find variables with correlation coefficient > 0.2
    correlated_vars = corr_matrix[corr_matrix.abs() > 0.2].stack().dropna().index.tolist()

    # Extract variable names with correlation > 0.2
    high_corr_vars = set()
    for (var1, var2) in correlated_vars:
        if var1 != var2 and var1 not in high_corr_vars and var2 not in high_corr_vars:
            high_corr_vars.add(var1)
            high_corr_vars.add(var2)

    high_corr_vars = list(high_corr_vars)

    # 4. Make X and y
    X = data.drop('salary', axis=1)
    y = data['salary']

    # 5. Split the predictors and target into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # 6. Fit models removing highly correlated variables
    best_mape = float('inf')

    # Fit models removing each variable
    for var in high_corr_vars:
        if var in X_train.columns:
            X_train_reduced = X_train.drop(var, axis=1)
            X_test_reduced = X_test.drop(var, axis=1)

            model = LinearRegression()
            model.fit(X_train_reduced, y_train)

            y_pred = model.predict(X_test_reduced)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            if mape < best_mape:
                best_mape = mape

    # Fit models removing each pair of variables
    for i in range(len(high_corr_vars)):
        for j in range(i + 1, len(high_corr_vars)):
            var1 = high_corr_vars[i]
            var2 = high_corr_vars[j]

            if var1 in X_train.columns and var2 in X_train.columns:
                X_train_reduced = X_train.drop([var1, var2], axis=1)
                X_test_reduced = X_test.drop([var1, var2], axis=1)

                model = LinearRegression()
                model.fit(X_train_reduced, y_train)

                y_pred = model.predict(X_test_reduced)
                mape = mean_absolute_percentage_error(y_test, y_pred)

                if mape < best_mape:
                    best_mape = mape

    # 7. Print the lowest MAPE rounded to five decimal places
    print(f"{best_mape:.5f}")


if __name__ == "__main__":
    main()

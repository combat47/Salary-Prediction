import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


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
    X = data[["rating"]]
    y = data["salary"]

    # Store the best MAPE
    best_mape = float('inf')
    best_power = 0

    for power in range(2, 5):
        # 4. Raise predictor to the power
        X_powered = X.copy()
        X_powered["rating"] = X_powered["rating"] ** power

        # 5. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_powered, y, test_size=0.3, random_state=100
        )

        # 6. Fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 7. Make predictions and calculate MAPE
        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Update best MAPE
        if mape < best_mape:
            best_mape = mape
            best_power = power

    # Print best MAPE
    print(f"{best_mape:.5f}")


if __name__ == "__main__":
    main()

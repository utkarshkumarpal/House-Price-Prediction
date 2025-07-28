# 📘 Improved House Price Prediction Model (with Higher R²)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv("Bengaluru_House_Data.csv")

# Drop unused columns
data.drop(columns=['area_type', 'availability', 'society'], inplace=True)
data.dropna(inplace=True)

# Clean location
data['location'] = data['location'].apply(lambda x: x.strip())
location_stats = data['location'].value_counts()
data['location'] = data['location'].apply(lambda x: 'Other' if location_stats[x] <= 10 else x)

# Extract bedrooms from size
data['bedrooms'] = data['size'].apply(lambda x: int(x.split(' ')[0]))

# Clean total_sqft (handle ranges)
def clean_sqft(x):
    try:
        if '-' in x:
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(clean_sqft)
data.dropna(inplace=True)

# Filter outliers
data['sqft_per_bed'] = data['total_sqft'] / data['bedrooms']
data = data[data['sqft_per_bed'] >= 300]
data['price_per_sqft'] = data['price'] * 100000 / data['total_sqft']
data = data[data['price_per_sqft'] >= 2000]
data.drop(columns=['size', 'sqft_per_bed', 'price_per_sqft'], inplace=True)

# Features and target
X = data.drop('price', axis=1)
y = data['price']

# Column transformer for encoding
col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['location']),
    remainder='passthrough'
)

# Build pipeline
model = make_pipeline(col_trans, RandomForestRegressor(n_estimators=100, random_state=42))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n\U0001f3af Improved R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

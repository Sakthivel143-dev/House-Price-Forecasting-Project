# House-Price-Forecasting-Project

house_price_forecast/
├── app.py
├── templates/
│   └── index.html

training code of this project
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# Training data: [size (sqft), bedrooms, age (years)]
X = np.array([
    [800, 1, 5],
    [1000, 2, 10],
    [1200, 2, 15],
    [1500, 3, 5],
    [1800, 3, 15],
    [2000, 3, 8],
    [2200, 4, 12],
    [2400, 4, 5],
    [2600, 4, 18],
    [2800, 5, 8],
    [3000, 4, 10],
    [3200, 5, 15],
    [3500, 5, 20],
    [4000, 6, 5],
    [4200, 6, 2],
    [4500, 7, 10]
])

# Prices in Indian Rupees
y = np.array([
    3500000, 5000000, 4800000, 7500000, 7200000,
    8000000, 9500000, 12000000, 11000000, 13000000,
    15000000, 15500000, 14000000, 18000000,
    20000000, 22000000
])

# Create pipeline: Standardize inputs then apply regression
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X, y)

# Evaluate performance
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print("✅ Model trained successfully!")
print(f"📊 Training RMSE: ₹{rmse:,.0f}")
print(f"📈 R² Score: {r2:.4f} (1.0 means perfect fit)")

# Save model for later use in Flask app
joblib.dump(model, 'model.joblib')



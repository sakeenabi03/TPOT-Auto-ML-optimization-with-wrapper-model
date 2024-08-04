import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tpot import TPOTRegressor

# Fetch dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Introduce some missing values for demonstration
X[::10, 1] = np.nan  # Let's say the 1st feature has missing values

# Separate data with missing values and complete data
X_complete = X[~np.isnan(X[:, 1])]
y_complete = y[~np.isnan(X[:, 1])]
X_missing = X[np.isnan(X[:, 1])]

# Train smaller model to predict missing feature
X_small = X_complete[:, [0, 2, 3, 4, 5, 6, 7]]  # Use all but the 1st feature
y_small = X_complete[:, 1]  # Predict the 1st feature

scaler = StandardScaler()
X_small_scaled = scaler.fit_transform(X_small)
smaller_model = LinearRegression().fit(X_small_scaled, y_small)

# Train TPOT AutoML model
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_complete, y_complete)

# Print the pipeline chosen by TPOT
print("Best pipeline chosen by TPOT:")
print(tpot.fitted_pipeline_)

# Save the models and scaler
with open('tpot_model.pkl', 'wb') as f:
    pickle.dump(tpot.fitted_pipeline_, f)

with open('smaller_model.pkl', 'wb') as f:
    pickle.dump(smaller_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

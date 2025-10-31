import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import tensorflow as tf

# ========= Helper metric ==========
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# ========= Load data ==========
print("Loading data from Data/ ...")
data_dir = "Data"
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print("Found CSV files:", files)

dfs = []
for f in files:
    path = os.path.join(data_dir, f)
    df = pd.read_csv(path)
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
print("Data shape:", data.shape)

# ========= Select target column ==========
target_col = "arrivalDelay"
if target_col not in data.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

# Convert target to numeric and drop NaNs
data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
data = data.dropna(subset=[target_col])

# ========= Select numeric features ==========
X_numeric = data.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")

# ========= Handle missing values ==========
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_numeric)

# ====== Create DataFrame with dynamic column names ======
X = pd.DataFrame(X_imputed, columns=[f"num_{i}" for i in range(X_imputed.shape[1])])

y = data[target_col]

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# ========= Split & scale ==========
if len(X) < 10:
    raise ValueError("Dataset too small after cleaning — check input CSVs!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = pd.to_numeric(y_train, errors='coerce').values
y_test = pd.to_numeric(y_test, errors='coerce').values

# Remove NaNs in target
mask_train = ~np.isnan(y_train)
mask_test = ~np.isnan(y_test)

X_train = X_train[mask_train]
y_train = y_train[mask_train]
X_test = X_test[mask_test]
y_test = y_test[mask_test]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

# ========= Model evaluation helper ==========
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))  # squared=False helyett
        sm = smape(y_test, preds)
        results.append([name, mae, rmse, sm])
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={sm:.3f}")
    except Exception as e:
        print(f"{name} failed: {e}")

# ========= Train models ==========
evaluate_model("Linear Regression", LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test)
evaluate_model("Gradient Boosting", GradientBoostingRegressor(random_state=42), X_train, X_test, y_train, y_test)
evaluate_model("SVR", SVR(kernel='rbf', C=10, gamma=0.1), X_train_scaled, X_test_scaled, y_train, y_test)

# ========= Deep Neural Network ==========
print("Training simple DNN...")
dnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
dnn.compile(optimizer='adam', loss='mae')
dnn.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=0)
preds_dnn = dnn.predict(X_test_scaled).flatten()
mae_dnn = mean_absolute_error(y_test, preds_dnn)
rmse_dnn = np.sqrt(mean_squared_error(y_test, preds_dnn))
sm_dnn = smape(y_test, preds_dnn)
results.append(["DNN", mae_dnn, rmse_dnn, sm_dnn])
print(f"DNN: MAE={mae_dnn:.3f}, RMSE={rmse_dnn:.3f}, sMAPE={sm_dnn:.3f}")

# ========= Save results ==========
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "sMAPE"])
output_path = "results.csv"
results_df.to_csv(output_path, index=False)
print("\nSaved results to", output_path)
print(results_df)

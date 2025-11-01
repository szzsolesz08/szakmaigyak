import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Adatok beolvasása ---
print("Loading data from Data/ ...")
data_dir = "Data"
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print(f"Found CSV files: {files}")

dfs = []
for file in files:
    path = os.path.join(data_dir, file)
    df = pd.read_csv(path, low_memory=False)
    dfs.append(df)

data = pd.concat(dfs, axis=0, ignore_index=True)
print(f"Data shape: {data.shape}")

# --- Célváltozó és featurek ---
target_col = "arrivalDelay"
if target_col not in data.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

X = data.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
y = data[target_col].astype(float)

# --- hiányzó értékek kezelése ---
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# --- csak véletlen mintavétel a nagy fájlokhoz ---
if len(X) > 100000:
    sample_size = 100000
    X = X.sample(sample_size, random_state=42)
    y = y.loc[X.index]
    print(f"Using random sample of {sample_size} rows")

# --- train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Features shape: {X_train.shape}, Target shape: {y_train.shape}")

# --- normalizálás ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    smape = 100 * np.mean(
        np.abs((y_pred - y_test) / ((np.abs(y_pred) + np.abs(y_test)) / 2))
    )
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "sMAPE": smape})
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape:.3f}")

# --- Klasszikus modellek ---
evaluate_model("Linear Regression", LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model("Random Forest", RandomForestRegressor(n_estimators=50, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model("Gradient Boosting", GradientBoostingRegressor(random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model("SVR", SVR(), X_train_scaled, X_test_scaled, y_train, y_test)

# --- DNN modell ---
print("Training simple DNN...")
dnn = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

dnn.compile(optimizer="adam", loss="mae")
dnn.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=0)

y_pred_dnn = dnn.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_pred_dnn)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_dnn))  # <-- fix: nincs squared paraméter
smape = 100 * np.mean(np.abs((y_pred_dnn - y_test) / ((np.abs(y_pred_dnn) + np.abs(y_test)) / 2)))

results.append({"Model": "DNN", "MAE": mae, "RMSE": rmse, "sMAPE": smape})
print(f"DNN: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape:.3f}")

# --- eredmények mentése ---
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print("\nSaved results to results.csv")
print(results_df)

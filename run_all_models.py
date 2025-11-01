import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
import warnings

warnings.filterwarnings("ignore")

print("Loading data from Data/ ...")

# --- Adatok betöltése ---
data_dir = "Data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print(f"Found CSV files: {csv_files}")

dfs = []
for file in csv_files:
    path = os.path.join(data_dir, file)
    df = pd.read_csv(path, low_memory=False)
    dfs.append(df)

df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"Data shape: {df.shape}")

# --- Ellenőrzés, hogy megvan-e a céloszlop ---
target_col = "arrivalDelay"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

# --- Konverzió percekre ---
df[target_col] = pd.to_numeric(df[target_col], errors="coerce") / 60.0

# --- Timestamp átalakítása órává ---
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour

# --- Aggregálás stopID + hour szinten ---
df_grouped = (
    df.groupby(["stopID", "hour"], as_index=False)
    .agg({"arrivalDelay": "mean"})
    .dropna()
)

# --- Featurek és target ---
X = df_grouped[["stopID", "hour"]]
y = df_grouped["arrivalDelay"]

# --- stopID kategória kódolása ---
X["stopID"] = X["stopID"].astype("category").cat.codes

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Eredmények tárolása ---
results = []

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    smape = 100 * np.mean(
        np.abs((y_pred - y_test) / ((np.abs(y_pred) + np.abs(y_test)) / 2))
    )
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "sMAPE": smape})
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape:.3f}")


# --- Modellek ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
}

for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        evaluate_model(name, model, X_test_scaled, y_test)
    except Exception as e:
        print(f"{name} failed: {e}")

# --- DNN modell ---
print("Training simple DNN...")
dnn = keras.Sequential(
    [
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ]
)

dnn.compile(optimizer="adam", loss="mae")
dnn.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=0)

y_pred_dnn = dnn.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_pred_dnn)
rmse = mean_squared_error(y_test, y_pred_dnn, squared=False)
smape = 100 * np.mean(
    np.abs((y_pred_dnn - y_test) / ((np.abs(y_pred_dnn) + np.abs(y_test)) / 2))
)
results.append({"Model": "DNN", "MAE": mae, "RMSE": rmse, "sMAPE": smape})
print(f"DNN: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape:.3f}")

# --- Eredmények mentése ---
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print("\nSaved results to results.csv")
print(results_df)

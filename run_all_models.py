import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow import keras

# --- 1. Adatok betöltése ---
print("Loading data from Data/ ...")

data_dir = "Data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print("Found CSV files:", csv_files)

dfs = []
for f in csv_files:
    path = os.path.join(data_dir, f)
    df = pd.read_csv(path, low_memory=False)
    dfs.append(df)

data = pd.concat(dfs, axis=0, ignore_index=True)
print("Data shape:", data.shape)

# --- 2. Céloszlop és featurek kiválasztása ---
target_col = "arrivalDelay"
if target_col not in data.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

possible_features = [
    "stopSequence",
    "depatureDelay",
    "timestamp",
    "arrivalTime",
    "depatureTime",
    "tripID",
    "stopID",
]
existing_features = [f for f in possible_features if f in data.columns]
X = data[existing_features].copy()
y = data[target_col].copy()

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# --- 3. Numerikus konverzió ---
# Minden oszlopot numerikus típusra próbálunk konvertálni, a nem konvertálható értékek NaN lesznek
for col in X.columns:
    X[col] = (
        X[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .replace("nan", np.nan)
    )
    X[col] = pd.to_numeric(X[col], errors="coerce")

# --- 4. Hiányzó értékek kezelése ---
imputer = SimpleImputer(strategy="median")

# Csak a nem üres oszlopok maradnak
valid_columns = X.columns[X.notna().any()].tolist()
X = X[valid_columns]

X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=valid_columns)

# Célváltozó tisztítása
y = pd.to_numeric(y, errors="coerce")
mask = ~y.isna()
X = X.loc[mask]
y = y.loc[mask]

# --- 5. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 6. Standardizálás ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 7. Modellek betanítása és kiértékelése ---
results = []

def evaluate_model(name, model):
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        smape = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test) + 1e-8)) * 100
        results.append((name, mae, rmse, smape))
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape:.3f}")
    except Exception as e:
        print(f"{name} failed: {e}")

evaluate_model("Linear Regression", LinearRegression())
evaluate_model("Random Forest", RandomForestRegressor(n_estimators=50, random_state=42))
evaluate_model("Gradient Boosting", GradientBoostingRegressor(random_state=42))
evaluate_model("SVR", SVR())

# --- 8. Egyszerű DNN ---
print("Training simple DNN...")

dnn = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

dnn.compile(optimizer="adam", loss="mae")
dnn.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=0)

y_pred_dnn = dnn.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_pred_dnn)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_dnn))
smape = np.mean(2 * np.abs(y_pred_dnn - y_test) / (np.abs(y_pred_dnn) + np.abs(y_test) + 1e-8)) * 100

results.append(("DNN", mae, rmse, smape))
print(f"DNN: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape:.3f}")

# --- 9. Eredmények mentése ---
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "sMAPE"])
results_df.to_csv("results.csv", index=False)
print("\nSaved results to results.csv")
print(results_df)

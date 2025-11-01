import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- SMAPE függvény ---
def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

print("Loading data from Data/ ...")
data_dir = "Data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print("Found CSV files:", csv_files)

dfs = []
for file in csv_files:
    path = os.path.join(data_dir, file)
    df = pd.read_csv(path, low_memory=False)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Data shape: {data.shape}")

# --- Célváltozó ---
target_col = "arrivalDelay"
if target_col not in data.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

# Csak numerikus feature-ök
X = data.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
y = data[target_col]

# --- NaN-ok kezelése ---
X = X.dropna(axis=1, how="all")  # teljesen üres oszlopok eldobása
y = y.replace([np.inf, -np.inf], np.nan).dropna()

# Szinkronizálás, hogy X és y ugyanannyi sort tartalmazzon
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]

# --- Imputálás ---
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

# --- Modellek futtatása ---
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    smape_val = smape(y_test, y_pred)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape_val:.3f}")
    results.append([name, mae, rmse, smape_val])


# 1. Linear Regression
evaluate_model("Linear Regression", LinearRegression(), X_train, X_test, y_train, y_test)

# 2. Random Forest
evaluate_model("Random Forest", RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
               X_train, X_test, y_train, y_test)

# 3. Gradient Boosting
evaluate_model("Gradient Boosting", GradientBoostingRegressor(random_state=42),
               X_train, X_test, y_train, y_test)

# 4. SVR
evaluate_model("SVR", SVR(kernel='rbf'), X_train, X_test, y_train, y_test)

# --- 5. Deep Neural Network ---
print("Training simple DNN...")

dnn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
dnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
dnn.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1, callbacks=[early_stop])

y_pred_dnn = dnn.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred_dnn)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_dnn))
smape_val = smape(y_test, y_pred_dnn)
print(f"DNN: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape_val:.3f}")
results.append(["DNN", mae, rmse, smape_val])

# --- Mentés ---
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "sMAPE"])
results_df.to_csv("results.csv", index=False)
print("\nSaved results to results.csv")
print(results_df)

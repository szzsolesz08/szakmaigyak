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

# --- Funkciók ---
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)


# --- Adatbetöltés ---
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

# --- Célváltozó és featurek ---
target_col = "arrivalDelay"
if target_col not in data.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

X = data.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
y = data[target_col].astype(float)

# Teljesen üres oszlopok eldobása
X = X.dropna(axis=1, how="all")

# --- Hiányzó értékek kezelése ---
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# --- 1. Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
smape_val = smape(y_test, y_pred_lr)
print(f"Linear Regression: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape_val:.3f}")
results.append(["Linear Regression", mae, rmse, smape_val])

# --- 2. Random Forest ---
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
smape_val = smape(y_test, y_pred_rf)
print(f"Random Forest: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape_val:.3f}")
results.append(["Random Forest", mae, rmse, smape_val])

# --- 3. Gradient Boosting ---
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_gb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
smape_val = smape(y_test, y_pred_gb)
print(f"Gradient Boosting: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape_val:.3f}")
results.append(["Gradient Boosting", mae, rmse, smape_val])

# --- 4. Support Vector Regressor ---
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_svr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr))
smape_val = smape(y_test, y_pred_svr)
print(f"SVR: MAE={mae:.3f}, RMSE={rmse:.3f}, sMAPE={smape_val:.3f}")
results.append(["SVR", mae, rmse, smape_val])

# --- 5. Egyszerű DNN ---
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

# --- Eredmények mentése ---
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "sMAPE"])
results_df.to_csv("results.csv", index=False)
print("\nSaved results to results.csv")
print(results_df)

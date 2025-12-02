import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import tensorflow as tf

# ========= Helper metric ==========
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# ========= Load data ==========
data_dir = "Data"
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in files]
data = pd.concat(dfs, ignore_index=True)

target_col = "arrivalDelay"

# ========= Select numeric features ==========
X_numeric = data.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
y = data[target_col]

# ========= Imputer ==========
imputer_X = SimpleImputer(strategy='median')
X_numeric_imputed = imputer_X.fit_transform(X_numeric)
X_numeric = pd.DataFrame(X_numeric_imputed, columns=[f"num_{i}" for i in range(X_numeric_imputed.shape[1])])

imputer_y = SimpleImputer(strategy='median')
y = pd.Series(imputer_y.fit_transform(y.values.reshape(-1,1)).flatten())

# ========= Train-test split ==========
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, random_state=42
)

# ========= Scaling ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# ========= Model evaluation helper ==========
results = []
def evaluate_model(name, y_test, preds):
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    sm = smape(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append([name, mae, rmse, sm, r2])
    print(f"{name}: R2={r2:.5f}, MAE={mae:.5f}, RMSE={rmse:.5f}, sMAPE={sm:.5f}")

# ========= Linear Regression ==========
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
evaluate_model("Linear Regression", y_test, y_pred)

# ========= Random Forest ==========
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_full, y_train)
y_pred = rf.predict(X_test_full)
evaluate_model("Random Forest", y_test, y_pred)

# ========= Gradient Boosting ==========
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train_full, y_train)
y_pred = gb.predict(X_test_full)
evaluate_model("Gradient Boosting", y_test, y_pred)

# ========= SVM-FS (feature selection + SVR) ==========
selector = SelectKBest(f_regression, k=15)
X_train_fs = selector.fit_transform(X_train_scaled, y_train)
X_test_fs = selector.transform(X_test_scaled)

svr_fs = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
svr_fs.fit(X_train_fs, y_train)
y_pred = svr_fs.predict(X_test_fs)
evaluate_model("SVM-FS", y_test, y_pred)

# ========= Deep Neural Network ==========
dnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
dnn.compile(optimizer='adam', loss='mae')
dnn.fit(X_train_scaled, y_train, epochs=10, batch_size=64, verbose=0)
y_pred = dnn.predict(X_test_scaled).flatten()
evaluate_model("DNN", y_test, y_pred)

# ========= Save results ==========
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "sMAPE", "R2"])
results_df.to_csv("results.csv", index=False)
print("\nSaved results to results.csv")
print(results_df)

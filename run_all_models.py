import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# SMAPE function
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

print("Loading data from Data/ ...")

# Collect all CSV files from Data/
data_path = "Data"
csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
print("Found CSV files:", csv_files)

# Load CSVs
dfs = [pd.read_csv(os.path.join(data_path, f), low_memory=False) for f in csv_files]

# Handle duplicate column names and concatenate
dfs_fixed = []
existing_cols = set()
for i, df_temp in enumerate(dfs):
    new_cols = []
    for col in df_temp.columns:
        if col in existing_cols:
            new_cols.append(f"{col}_{i}")
        else:
            new_cols.append(col)
        existing_cols.add(new_cols[-1])
    df_temp.columns = new_cols
    dfs_fixed.append(df_temp)

df = pd.concat(dfs_fixed, axis=1, ignore_index=False)
print("Data shape:", df.shape)

# Replace commas with dots (for numeric conversion)
df = df.replace(",", ".", regex=True)

# Encode non-numeric columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Drop constant or all-NaN columns
X = X.loc[:, X.nunique() > 1]

# Convert all columns to numeric safely
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Remove NaNs from target
mask = ~np.isnan(y)
X = X.loc[mask]
y = y.loc[mask]

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results.append(["Linear Regression",
                mean_absolute_error(y_test, y_pred_lr),
                np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                smape(y_test, y_pred_lr)])

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append(["Random Forest",
                mean_absolute_error(y_test, y_pred_rf),
                np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                smape(y_test, y_pred_rf)])

# 3. Gradient Boosting
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
results.append(["Gradient Boosting",
                mean_absolute_error(y_test, y_pred_gb),
                np.sqrt(mean_squared_error(y_test, y_pred_gb)),
                smape(y_test, y_pred_gb)])

# 4. SVM-FS (replaces SVR)
print("Running SVM-FS model...")
selector = SelectKBest(f_regression, k=min(15, X_train.shape[1]))
X_train_fs = selector.fit_transform(X_train, y_train)
X_test_fs = selector.transform(X_test)

svm_fs = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
svm_fs.fit(X_train_fs, y_train)
y_pred_svmfs = svm_fs.predict(X_test_fs)

results.append(["SVM-FS",
                mean_absolute_error(y_test, y_pred_svmfs),
                np.sqrt(mean_squared_error(y_test, y_pred_svmfs)),
                smape(y_test, y_pred_svmfs)])

# 5. Simple DNN
print("Training simple DNN...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mae')
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1, callbacks=[early_stop])

y_pred_dnn = model.predict(X_test).flatten()
results.append(["DNN",
                mean_absolute_error(y_test, y_pred_dnn),
                np.sqrt(mean_squared_error(y_test, y_pred_dnn)),
                smape(y_test, y_pred_dnn)])

# Save results
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "sMAPE"])
print("Saved results to results.csv")
print(results_df)
results_df.to_csv("results.csv", index=False)

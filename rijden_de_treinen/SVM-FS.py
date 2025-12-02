# svr_disruptions_pipeline.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===== CONFIG =====
DATA_PATH = '../Data/disruptions-2024.csv'   # állítsd a fájlod elérési útjára
OUTPUT_XLSX = 'Predicted_values_SVR.xlsx'
TEST_FRACTION = 0.2
RANDOM_SEED = 1
# ==================

np.random.seed(RANDOM_SEED)

def smape(A, F):
    A = np.array(A, dtype=float)
    F = np.array(F, dtype=float)
    denom = (np.abs(A) + np.abs(F))
    denom[denom == 0] = 1.0
    return 100.0 / len(A) * np.sum(2 * np.abs(F - A) / denom)

# ----- Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Nem található a fájl: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)

# ----- időbélyeg
time_col = None
for c in ['start_time','timestamp','end_time']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')
        time_col = c
        break
if time_col is None:
    raise RuntimeError("Nem található időbélyeg oszlop (start_time/timestamp/end_time)")

# ----- Feature engineering
df['hour'] = df[time_col].dt.hour.fillna(0).astype(int)
df['minute'] = df[time_col].dt.minute.fillna(0).astype(int)
df['weekday'] = df[time_col].dt.dayofweek.fillna(0).astype(int)
df['hr_compact'] = (df['weekday'].astype(str) +
                     df['hour'].astype(str).str.zfill(2) +
                     df['minute'].astype(str).str.zfill(2)).astype(int)

# hány állomás érintett
if 'rdt_station_names' in df.columns:
    df['n_stations'] = df['rdt_station_names'].fillna('').apply(
        lambda s: 0 if str(s).strip() == '' else len([x for x in str(s).split(',') if x.strip()!=''])
    ).astype(int)
else:
    df['n_stations'] = 0

# kategóriák enkódolása
def label_encode_column(df, colname, newname):
    if colname in df.columns:
        le = preprocessing.LabelEncoder()
        df[newname] = le.fit_transform(df[colname].astype(str)).astype(float)
    else:
        df[newname] = 0.0

label_encode_column(df, 'cause_group', 'cause_group_enc')
label_encode_column(df, 'cause_en', 'cause_en_enc')
label_encode_column(df, 'rdt_lines', 'rdt_lines_enc')
label_encode_column(df, 'ns_lines', 'ns_lines_enc')

# ----- Target variable
if 'duration_minutes' not in df.columns:
    raise RuntimeError("Nincs 'duration_minutes' oszlop a target-hez")
df = df.dropna(subset=['duration_minutes']).copy()
df['duration_minutes'] = df['duration_minutes'].astype(float)

# ----- Features
features = [
    'hour', 'minute', 'weekday', 'hr_compact', 'n_stations',
    'cause_group_enc', 'cause_en_enc', 'rdt_lines_enc', 'ns_lines_enc'
]
features = [f for f in features if f in df.columns]

X = df[features].fillna(0).astype(float)
y = pd.to_numeric(df['duration_minutes'], errors='coerce').fillna(0).astype(float)

# ----- Feature selection
X_selected = SelectKBest(f_regression, k=min(15, X.shape[1])).fit_transform(X, y)

# ----- Train-test split időrendi
df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
split_idx = int((1.0 - TEST_FRACTION) * len(df_sorted))
X_train = X_selected[:split_idx]
X_test = X_selected[split_idx:]
y_train = y[:split_idx].values
y_test = y[split_idx:].values

print(f"Összes minta: {len(df_sorted)}, Train: {len(X_train)}, Test: {len(X_test)}")

# ----- Support Vector Regression
svr_model = SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1)
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)

# ----- Metrikák
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
sm = smape(y_test, y_pred)

print("\n--- Értékelés (valós skála) ---")
print(f"R2: {r2:.6f}")
print(f"MAE: {mae:.4f} perc")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f} perc")
print(f"SMAPE: {sm:.4f} %")

# ----- Vizualizáció (teszt első 300 minta)
N = min(300, len(y_test))
plt.figure(figsize=(14,6))
plt.plot(range(N), y_pred[:N], 'g-', label='Predicted')
plt.plot(range(N), y_test[:N], 'b-', label='Actual')
plt.xlabel('Test sample index (first N)')
plt.ylabel('Duration (minutes)')
plt.title('SVR Prediction (first N samples)')
plt.legend()
plt.grid(True)
plt.show()

# ----- Mentés
y_pred_df = pd.DataFrame(y_pred, columns=['Predicted'])
y_pred_df.to_excel(OUTPUT_XLSX, index=False)
print(f"Predikciók elmentve: {OUTPUT_XLSX}")

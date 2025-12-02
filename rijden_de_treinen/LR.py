# lr_disruptions_pipeline.py
# Linear Regression pipeline a 'disruptions-2024.csv' fájlra.
# Használat:
#  - python lr_disruptions_pipeline.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error

# ====== CONFIG ======
DATA_PATH = '../Data/disruptions-2024.csv'   # állítsd a saját fájlod elérési útjára
OUTPUT_XLSX = 'Predicted_values_LR.xlsx'
TEST_FRACTION = 0.2
RANDOM_SEED = 1
# ====================

np.random.seed(RANDOM_SEED)

def smape(A, F):
    A = np.array(A, dtype=float)
    F = np.array(F, dtype=float)
    denom = (np.abs(A) + np.abs(F))
    denom[denom == 0] = 1.0
    return 100.0 / len(A) * np.sum(2 * np.abs(F - A) / denom)

# ----- load
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Nem található a fájl: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

# ----- biztos datetime parse
for c in ['start_time', 'end_time', 'timestamp']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# preferált időoszlop: start_time > timestamp > end_time
time_col = None
for candidate in ['start_time','timestamp','end_time']:
    if candidate in df.columns:
        time_col = candidate
        break
if time_col is None:
    raise RuntimeError("Nem található időbélyeg oszlop (start_time/timestamp/end_time) a fájlban.")

# ----- időből jellemzők
df['date'] = df[time_col].dt.date
df['hour'] = df[time_col].dt.hour.fillna(0).astype(int)
df['minute'] = df[time_col].dt.minute.fillna(0).astype(int)
df['weekday'] = df[time_col].dt.dayofweek.fillna(0).astype(int)
df['hr_compact'] = (df['weekday'].astype(str) +
                    df['hour'].astype(str).str.zfill(2) +
                    df['minute'].astype(str).str.zfill(2)).astype(int)

# ----- egyszerű extra jellemzők
# hány állomás érintett?
if 'rdt_station_names' in df.columns:
    df['n_stations'] = df['rdt_station_names'].fillna('').apply(
        lambda s: 0 if str(s).strip()=='' else len([x for x in str(s).split(',') if x.strip()!=''])
    ).astype(int)
else:
    df['n_stations'] = 0

# rdt_lines_id numeric vagy label-encode
if 'rdt_lines_id' in df.columns:
    df['rdt_lines_id_num'] = pd.to_numeric(df['rdt_lines_id'], errors='coerce')
    if df['rdt_lines_id_num'].isnull().all():
        le_tmp = preprocessing.LabelEncoder()
        df['rdt_lines_id_num'] = le_tmp.fit_transform(df['rdt_lines_id'].astype(str)).astype(float)
    else:
        df['rdt_lines_id_num'] = df['rdt_lines_id_num'].fillna(-1).astype(float)
else:
    df['rdt_lines_id_num'] = 0.0

# kategória enkódolások, csak ha léteznek
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

# ----- célváltozó: duration_minutes
if 'duration_minutes' not in df.columns:
    raise RuntimeError("Nincs 'duration_minutes' oszlop. Ez szükséges a target-hez.")

df = df.dropna(subset=['duration_minutes']).copy()
df['duration_minutes'] = df['duration_minutes'].astype(float)

# ----- feature lista (egyszerű, adaptálható)
features = [
    'rdt_lines_id_num', 'rdt_lines_enc', 'ns_lines_enc', 'cause_group_enc', 'cause_en_enc',
    'hour', 'minute', 'weekday', 'hr_compact', 'n_stations'
]
# csak a meglévő feature-öket válasszuk ki
features = [f for f in features if f in df.columns]

if len(features) == 0:
    raise RuntimeError("Nincsenek elérhető feature-ök a kiválasztott listából.")

X = df[features].copy()
y = df['duration_minutes'].copy()

# ----- hiányzó érték kezelés
X = X.fillna(-1)

# ----- egyszerű scaling opcionálisan (lineáris regressziónál általában oké)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- időrendi train/test split
df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
X_sorted = pd.DataFrame(scaler.transform(df_sorted[features].fillna(-1)), columns=features)
y_sorted = df_sorted['duration_minutes'].reset_index(drop=True)

split_idx = int((1.0 - TEST_FRACTION) * len(df_sorted))
X_train = X_sorted.iloc[:split_idx].values
X_test = X_sorted.iloc[split_idx:].values
y_train = y_sorted.iloc[:split_idx].values
y_test = y_sorted.iloc[split_idx:].values

print(f"Összes minta: {len(df_sorted)}, Train: {len(X_train)}, Test: {len(X_test)}")
print("Használt feature-ök:", features)

# ----- Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# ----- metrikák
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
median_abs = median_absolute_error(y_test, y_pred)
sm = smape(y_test, y_pred)

print("\n--- Értékelés (valós skála) ---")
print(f"R2: {r2:.6f}")
print(f"MAE: {mae:.4f} perc")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f} perc")
print(f"Median AE: {median_abs:.4f} perc")
print(f"SMAPE: {sm:.4f} %")

# ----- vizualizáció (teszt első N)
N = min(300, len(y_test))
plt.figure(figsize=(14,6))
plt.plot(range(N), y_pred[:N], label='Predicted', linestyle='-')
plt.plot(range(N), y_test[:N], label='Actual', linestyle='-')
plt.xlabel('Test sample index (first N)')
plt.ylabel('Duration (minutes)')
plt.title('Linear Regression Prediction (first N samples)')
plt.legend()
plt.grid(True)
plt.show()

# ----- mentés (teszt rész + predikciók)
out_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)
out_df['predicted_minutes'] = y_pred
out_df.to_excel(OUTPUT_XLSX, index=False)
print(f"Előrejelzések elmentve: {OUTPUT_XLSX}")

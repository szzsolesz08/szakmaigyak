# adapt_disruptions_pipeline_log.py
# Log-transzformált regressziós pipeline a disruptions-2024.csv fájlra.
# Használat:
#  - python adapt_disruptions_pipeline_log.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
import os

def smape(A, F):
    A = np.array(A, dtype=float)
    F = np.array(F, dtype=float)
    denom = (np.abs(A) + np.abs(F))
    denom[denom == 0] = 1.0
    return 100.0 / len(A) * np.sum(2 * np.abs(F - A) / denom)

# ======== CONFIG ========
# Állítsd a fájl elérési útját:
path = '../Data/disruptions-2024.csv'   # Windows például: r'C:\Users\...\disruptions-2024.csv'
output_excel = 'Predicted_values_from_disruptions_log.xlsx'
# Ha szeretnél, engedélyezhetsz clippinget (pl. 2000 perc fölötti értékeket levágni):
CLIP_OUTLIERS = False
CLIP_MAX = 2000.0
# =======================

if not os.path.exists(path):
    raise FileNotFoundError(f"Nem található a fájl: {path}")

# Betöltés
data = pd.read_csv(path, low_memory=False)

# Datetime parse (start_time preferált, különben end_time)
for col in ['start_time', 'end_time']:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

time_col = 'start_time' if 'start_time' in data.columns else ('end_time' if 'end_time' in data.columns else None)
if time_col is None:
    raise RuntimeError("Nincs start_time vagy end_time oszlop az adatokban. Szükséges időbélyeg hiányzik.")

# Idő alapú jellemzők
data['date'] = data[time_col].dt.date
data['hour'] = data[time_col].dt.hour.fillna(0).astype(int)
data['minute'] = data[time_col].dt.minute.fillna(0).astype(int)
data['weekday'] = data[time_col].dt.dayofweek.fillna(0).astype(int)
data['hr_compact'] = (data['weekday'].astype(str) +
                      data['hour'].astype(str).str.zfill(2) +
                      data['minute'].astype(str).str.zfill(2)).astype(int)

# Station count feature (rdt_station_names)
data['rdt_station_names'] = data.get('rdt_station_names', '')
data['n_stations'] = data['rdt_station_names'].fillna('').apply(
    lambda s: 0 if str(s).strip() == '' else len([x for x in str(s).split(',') if x.strip() != ''])
)

# rdt_lines_id numeric vagy label-encode
if 'rdt_lines_id' in data.columns:
    data['rdt_lines_id_num'] = pd.to_numeric(data['rdt_lines_id'], errors='coerce')
    if data['rdt_lines_id_num'].isnull().all():
        le_tmp = preprocessing.LabelEncoder()
        data['rdt_lines_id_num'] = le_tmp.fit_transform(data['rdt_lines_id'].astype(str)).astype(float)
    else:
        data['rdt_lines_id_num'] = data['rdt_lines_id_num'].fillna(-1).astype(float)
else:
    data['rdt_lines_id_num'] = 0.0

# Categorical encodings (cause_group, cause_en, rdt_lines)
def safe_label_encode(df, col_name, new_name):
    if col_name in df.columns:
        le = preprocessing.LabelEncoder()
        df[new_name] = le.fit_transform(df[col_name].astype(str)).astype(float)
    else:
        df[new_name] = 0.0

safe_label_encode(data, 'cause_group', 'cause_group_enc')
safe_label_encode(data, 'cause_en', 'cause_en_enc')
safe_label_encode(data, 'rdt_lines', 'rdt_lines_enc')

# Target: duration_minutes
if 'duration_minutes' not in data.columns:
    raise RuntimeError("Nincs 'duration_minutes' oszlop — ez a script ezt használja célváltozónak.")
data = data.dropna(subset=['duration_minutes'])
data['duration_minutes'] = data['duration_minutes'].astype(float)

# (Opcionális) outlier clipping: ha engedélyezed, vágjuk le a nagyon nagy értékeket.
if CLIP_OUTLIERS:
    data['duration_minutes'] = data['duration_minutes'].clip(upper=CLIP_MAX)

# Log-transzformáció a célon (stabilizálja a heavy-tail eloszlást)
# y_train_log = log1p(duration_minutes)
data['duration_log'] = np.log1p(data['duration_minutes'])

# Feature lista (igény szerint bővíthető)
feature_cols = [
    'rdt_lines_id_num', 'rdt_lines_enc', 'cause_group_enc', 'cause_en_enc',
    'hour', 'minute', 'weekday', 'hr_compact', 'n_stations'
]

X = data[feature_cols].copy()
y_log = data['duration_log'].copy()
y_raw = data['duration_minutes'].copy()

# Train/test split időrendben (80/20)
train_pct_index = int(0.8 * len(y_log))
X_train, X_test = X.iloc[:train_pct_index], X.iloc[train_pct_index:]
y_train_log, y_test_log = y_log.iloc[:train_pct_index], y_log.iloc[train_pct_index:]
y_train_raw, y_test_raw = y_raw.iloc[:train_pct_index], y_raw.iloc[train_pct_index:]

print(f"Teljes sorok: {len(data)} | Train: {len(X_train)} | Test: {len(X_test)}")
print("Használt feature-ök:", feature_cols)
print("Outlier clipping engedélyezve:", CLIP_OUTLIERS, " CLIP_MAX:", CLIP_MAX)

# Modell tanítása a log-transzformált célon
reg = GradientBoostingRegressor(random_state=1)
reg.fit(X_train, y_train_log)
y_pred_log = reg.predict(X_test)

# Visszatranszformálás a valós skálára
y_pred_raw = np.expm1(y_pred_log)   # expm1(inverse of log1p)

# Kiértékelés a valós skálán (érthető metrikák)
r2_orig = r2_score(y_test_raw, y_pred_raw)
mae_orig = mean_absolute_error(y_test_raw, y_pred_raw)
mse_orig = mean_squared_error(y_test_raw, y_pred_raw)
rmse_orig = np.sqrt(mse_orig)
sm_orig = smape(y_test_raw.values, y_pred_raw)
median_abs_orig = median_absolute_error(y_test_raw, y_pred_raw)

# Kiértékelés a log-skálán (opcionális, mutatja, hogyan teljesít a modell a transzformált célon)
r2_log = r2_score(y_test_log, y_pred_log)
mae_log = mean_absolute_error(y_test_log, y_pred_log)
mse_log = mean_squared_error(y_test_log, y_pred_log)
rmse_log = np.sqrt(mse_log)

print("\n== Kiértékelés (valós skála, visszatranszformálva) ==")
print(f"R2 Score (orig): {r2_orig:.6f}")
print(f"MAE (orig): {mae_orig:.4f} (perc)")
print(f"MSE (orig): {mse_orig:.4f}")
print(f"RMSE (orig): {rmse_orig:.4f} (perc)")
print(f"Median Absolute Error (orig): {median_abs_orig:.4f} (perc)")
print(f"SMAPE (orig): {sm_orig:.4f} %")

print("\n== Kiértékelés (log-skála) ==")
print(f"R2 Score (log): {r2_log:.6f}")
print(f"MAE (log): {mae_log:.6f}")
print(f"MSE (log): {mse_log:.6f}")
print(f"RMSE (log): {rmse_log:.6f}")

# Plot (teszt minták egy része) - valós skála összehasonlítás
nplot = min(300, len(y_test_raw))
plt.figure(figsize=(14,6))
plt.plot(range(nplot), y_pred_raw[:nplot], label='Predicted (raw scale)', linestyle='-')
plt.plot(range(nplot), y_test_raw.values[:nplot], label='Actual (raw scale)', linestyle='-')
plt.xlabel('Test sample index (first n)')
plt.ylabel('Duration (minutes)')
plt.title('Prediction of Disruption Duration (minutes) - (log-trained -> raw predictions)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mentés: a teszt rész halmaz eredményei + predikciók
out_df = data.iloc[train_pct_index:].copy().reset_index(drop=True)
out_df['predicted_duration_minutes'] = y_pred_raw
out_df['predicted_duration_log'] = y_pred_log
out_df.to_excel(output_excel, index=False, sheet_name='Predictions')
print(f"\nElőrejelzések elmentve: {output_excel}")

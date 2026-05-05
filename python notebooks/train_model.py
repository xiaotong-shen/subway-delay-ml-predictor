"""
Production training script — Random Forest classifier + regressor.
Replaces the neural network as the serving model after benchmarking showed
RF beats NN on accuracy (+4.7pp), weighted F1 (+0.04), and macro F1 (+0.04)
with similar MAE, and trains in seconds instead of minutes.
Run from the 'python notebooks/' directory.
"""

import numpy as np
import pandas as pd
import glob
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, f1_score,
    mean_absolute_error, accuracy_score,
)

# ── 1. Load & preprocess ──────────────────────────────────────────────────────

parts = [pd.read_csv(f) for f in sorted(glob.glob("subway-data-2021-*.csv"))]
df = pd.concat(
    parts + [pd.read_csv(f) for f in ["subway-data-2022.csv", "subway-data-2023.csv", "subway-data-2024.csv"]]
)

remove_lines = ["SRT", "109 RANEE", "TRACK LEVEL ACTIVITY", "YU/BD/SHP"]
df = df[~df["Line"].isin(remove_lines)]
df = df.drop(columns=["Code", "Min Gap", "Bound", "Line", "Vehicle"])
df = df[(df["Min Delay"] > 0) & (df["Min Delay"] <= 60)]

remove_stations = [
    "111 SPADINA ROAD", "1900 YONGE MCBRIEN BLD", "1900 YONGE ST- MCBRIEN",
    "2233 SHEPPARD WEST", "ALL STATIONS", "BLOOR DANFORTH LINE",
    "YONGE UNIVERSITY LINE", "YONGE UNIVERSITY SUBWA", "EGLINTON STATION (MIGR",
    "TORONTO TRANSIT COMMIS",
]
df = df[~df["Station"].isin(remove_stations)]
df = df[~df["Station"].str.contains("app|yard| to |towards", case=False)]

df["Station"] = df["Station"].str.replace(" BD STATION", " STATION", regex=False)
df["Station"] = df["Station"].str.replace(" YUS STATION", " STATION", regex=False)
df["Station"] = df["Station"].str.replace("ST. ", "ST ", regex=False)
df.loc[df["Station"] == "PIONEER VILLAGE STATIO", "Station"] = "PIONEER VILLAGE STATION"
df.loc[df["Station"] == "YORK UNIVERSITY STATIO", "Station"] = "YORK UNIVERSITY STATION"
df.loc[df["Station"] == "DAVISVILLE BUILD UP", "Station"] = "DAVISVILLE STATION"
df.loc[df["Station"] == "SHEPPARD STATION", "Station"] = "SHEPPARD-YONGE STATION"
df.loc[df["Station"] == "BLOOR STATION", "Station"] = "BLOOR-YONGE STATION"
df.loc[df["Station"] == "YONGE STATION", "Station"] = "BLOOR-YONGE STATION"
df.loc[df["Station"] == "YONGE-UNIVERSITY AND B", "Station"] = "BLOOR-YONGE STATION"
df.loc[df["Station"] == "YONGE/UNIVERSITY AND B", "Station"] = "BLOOR-YONGE STATION"

station_counts = df["Station"].value_counts()
df = df[~df["Station"].isin(station_counts[station_counts < 50].index)]

df["Date"] = df["Date"].str.replace("/", "-", regex=False)
df["Date_dt"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
df = df.dropna(subset=["Date_dt"])

df["station_encoded"], station_mapping = pd.factorize(df["Station"])

start_min, end_min = 6 * 60, 25 * 60 + 30
mins = df["Time_dt"].dt.hour * 60 + df["Time_dt"].dt.minute
df["minutes_since_6am"] = mins.copy()
df.loc[df["Time_dt"].dt.hour < 6, "minutes_since_6am"] += 24 * 60
df["time_norm"] = (df["minutes_since_6am"] - start_min) / (end_min - start_min)
df["time_norm"] = df["time_norm"].clip(0, 1)

df["month"]             = df["Date_dt"].dt.month
df["day_of_week"]       = df["Date_dt"].dt.dayofweek
df["is_weekend"]        = df["Date_dt"].dt.dayofweek.isin([5, 6]).astype(int)
df["is_morning_rush"]   = ((df["Time_dt"].dt.hour >= 7) & (df["Time_dt"].dt.hour <= 9)).astype(int)
df["is_evening_rush"]   = ((df["Time_dt"].dt.hour >= 16) & (df["Time_dt"].dt.hour <= 18)).astype(int)
df["is_holiday_season"] = df["Date_dt"].dt.month.isin([12, 1]).astype(int)
df["is_back_to_school"] = df["Date_dt"].dt.month.isin([9]).astype(int)

# ── 2. Aggregate by operational context ───────────────────────────────────────

group_keys = [
    "station_encoded",
    "is_morning_rush", "is_evening_rush",
    "is_weekend",
    "is_holiday_season", "is_back_to_school",
]
agg = df.groupby(group_keys).agg(
    y_length    = ("Min Delay", "mean"),
    time_norm   = ("time_norm", "mean"),
    month       = ("month", "median"),
    day_of_week = ("day_of_week", "median"),
    n_events    = ("Min Delay", "count"),
).reset_index()
agg["month"]       = agg["month"].round().astype(int)
agg["day_of_week"] = agg["day_of_week"].round().astype(int)

# Explicit station-level mean — top-2 feature in importance ranking
station_means = df.groupby("station_encoded")["Min Delay"].mean().rename("station_mean_delay")
agg = agg.merge(station_means, on="station_encoded", how="left")

agg["delay_severity"] = pd.cut(
    agg["y_length"],
    bins=[-0.1, 5.0, 9.0, float("inf")],
    labels=["Minor", "Moderate", "Severe"],
)
agg = agg.dropna(subset=["delay_severity"])
agg["delay_severity"] = agg["delay_severity"].cat.remove_unused_categories()

print(f"Training on {len(agg):,} aggregated groups")
print(f"Class split:\n{agg['delay_severity'].value_counts()}\n")

# ── 3. Features & split ───────────────────────────────────────────────────────

FEATURES = [
    "station_encoded",
    "is_morning_rush", "is_evening_rush", "is_weekend",
    "is_holiday_season", "is_back_to_school",
    "time_norm", "month", "day_of_week",
    "n_events",
    "station_mean_delay",
]

le = LabelEncoder()
y_cls = le.fit_transform(agg["delay_severity"])
y_reg = agg["y_length"].values
X = agg[FEATURES].values

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

# ── 4. Train ──────────────────────────────────────────────────────────────────

print("Training Random Forest classifier...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_cls_train)

print("Training Random Forest regressor...")
reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)
reg.fit(X_train, y_reg_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────

y_cls_pred = clf.predict(X_test)
y_reg_pred = reg.predict(X_test)

majority_acc = accuracy_score(y_cls_test, np.full_like(y_cls_test, np.bincount(y_cls_train).argmax()))

acc  = accuracy_score(y_cls_test, y_cls_pred)
wf1  = f1_score(y_cls_test, y_cls_pred, average="weighted", zero_division=0)
mf1  = f1_score(y_cls_test, y_cls_pred, average="macro",    zero_division=0)
mae  = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(np.mean((y_reg_test - y_reg_pred) ** 2))

print("\n" + "=" * 55)
print("BENCHMARK RESULTS — Random Forest")
print("=" * 55)
print(f"  Majority-class baseline accuracy : {majority_acc:.1%}")
print(f"  Model accuracy                   : {acc:.1%}")
print(f"  Weighted F1                      : {wf1:.3f}")
print(f"  Macro F1                         : {mf1:.3f}")
print(f"\nPer-Class Report:")
print(classification_report(y_cls_test, y_cls_pred,
                            target_names=le.classes_, zero_division=0))
print(f"  Delay Length MAE  : {mae:.2f} min")
print(f"  Delay Length RMSE : {rmse:.2f} min")
print("=" * 55)

# ── 6. Pre-compute predictions for all station × time combinations ────────────
# Build the full grid as a DataFrame first, then call predict once on the whole
# matrix — avoids 294K individual predict() calls which are extremely slow.

print("\nPre-computing predictions for all stations × time combinations...")

total_op_min = (25 * 60 + 30) - (6 * 60)
time_grid    = np.linspace(0, 1, 50)
n_ev_median  = int(agg["n_events"].median())

# Build a lookup from (station, morning, evening, weekend, holiday, school) → n_events
grp_index = agg.set_index(group_keys)["n_events"].to_dict()

rows = []
meta = []
for station_id in range(len(station_mapping)):
    s_mean = float(station_means.get(station_id, station_means.mean()))
    for month in range(1, 13):
        is_holiday = int(month in [12, 1])
        is_school  = int(month == 9)
        for dow in range(7):
            is_weekend = int(dow in [5, 6])
            for t in time_grid:
                minute_of_day = t * total_op_min + 6 * 60
                hour = int(minute_of_day // 60) % 24
                is_morning = int(7 <= hour <= 9)
                is_evening = int(16 <= hour <= 18)
                key = (station_id, is_morning, is_evening, is_weekend, is_holiday, is_school)
                n_ev = grp_index.get(key, n_ev_median)

                rows.append([station_id, is_morning, is_evening, is_weekend,
                              is_holiday, is_school, t, month, dow, n_ev, s_mean])
                abs_min = 6 * 60 + t * total_op_min
                h = int(abs_min // 60) % 24
                m_m = int(abs_min % 60)
                meta.append({
                    "station":      station_id,
                    "station_name": station_mapping[station_id],
                    "month":        month,
                    "day_of_week":  dow,
                    "is_weekend":   is_weekend,
                    "time_norm":    t,
                    "time_hhmm":    f"{h:02d}:{m_m:02d}",
                })

X_pred = np.array(rows)
sev_idx = clf.predict(X_pred)
delays  = reg.predict(X_pred)

pred_df = pd.DataFrame(meta)
pred_df["delay_severity"] = le.classes_[sev_idx]
pred_df["delay_length"]   = delays

pred_df.to_csv("subway_delay_predictions.csv", index=False)
print(f"Saved {len(pred_df):,} predictions to subway_delay_predictions.csv")

# ── 7. Save model artifacts ───────────────────────────────────────────────────

artifacts = {
    "classifier":      clf,
    "regressor":       reg,
    "label_encoder":   le,
    "station_mapping": station_mapping,
    "station_means":   station_means,
    "features":        FEATURES,
}
with open("rf_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)
print("Saved model artifacts to rf_model.pkl")

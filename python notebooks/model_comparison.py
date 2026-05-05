"""
Model comparison: Neural Network vs XGBoost vs LightGBM vs Random Forest
Same aggregated data and train/test split for all models.
Outputs a side-by-side benchmark table.
"""

import numpy as np
import pandas as pd
import glob
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report, f1_score, mean_absolute_error,
    accuracy_score
)
import xgboost as xgb
import lightgbm as lgb

# ── 1. Load & preprocess (identical to neuralnet.py) ──────────────────────────

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

# ── 2. Aggregate by operational context (same as neuralnet.py) ────────────────

group_keys = [
    "station_encoded",
    "is_morning_rush", "is_evening_rush",
    "is_weekend",
    "is_holiday_season", "is_back_to_school",
]
agg = df.groupby(group_keys).agg(
    y_length   = ("Min Delay", "mean"),
    time_norm  = ("time_norm", "mean"),
    month      = ("month", "median"),
    day_of_week= ("day_of_week", "median"),
    n_events   = ("Min Delay", "count"),
    station_avg= ("Min Delay", "mean"),  # same as y_length here but explicit
).reset_index()
agg["month"]       = agg["month"].round().astype(int)
agg["day_of_week"] = agg["day_of_week"].round().astype(int)

# Add station-level mean delay as an explicit feature (gives tree models a strong prior)
station_means = df.groupby("station_encoded")["Min Delay"].mean().rename("station_mean_delay")
agg = agg.merge(station_means, on="station_encoded", how="left")

# Severity labels (25th/75th percentile bins of group averages)
agg["delay_severity"] = pd.cut(
    agg["y_length"],
    bins=[-0.1, 5.0, 9.0, float("inf")],
    labels=["Minor", "Moderate", "Severe"],
)
agg = agg.dropna(subset=["delay_severity"])
agg["delay_severity"] = agg["delay_severity"].cat.remove_unused_categories()

print(f"Dataset: {len(agg):,} groups")
print(f"Class distribution:\n{agg['delay_severity'].value_counts()}\n")

# ── 3. Features & split ───────────────────────────────────────────────────────

FEATURES = [
    "station_encoded",
    "is_morning_rush", "is_evening_rush", "is_weekend",
    "is_holiday_season", "is_back_to_school",
    "time_norm", "month", "day_of_week",
    "n_events",          # group size — proxy for how reliable the avg is
    "station_mean_delay", # station-level prior
]

le = LabelEncoder()
y_cls = le.fit_transform(agg["delay_severity"])
y_reg = agg["y_length"].values

X = agg[FEATURES].values

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

# ── 4. Run all models ─────────────────────────────────────────────────────────

results = {}

def evaluate(name, y_true_cls, y_pred_cls, y_true_reg, y_pred_reg):
    acc  = accuracy_score(y_true_cls, y_pred_cls)
    wf1  = f1_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
    mf1  = f1_score(y_true_cls, y_pred_cls, average="macro",    zero_division=0)
    mae  = mean_absolute_error(y_true_reg, y_pred_reg)
    rmse = np.sqrt(np.mean((y_true_reg - y_pred_reg) ** 2))
    results[name] = dict(accuracy=acc, weighted_f1=wf1, macro_f1=mf1, mae=mae, rmse=rmse)
    print(f"\n── {name} ──")
    print(f"  Accuracy:    {acc:.3f}")
    print(f"  Weighted F1: {wf1:.3f}")
    print(f"  Macro F1:    {mf1:.3f}")
    print(f"  MAE:         {mae:.2f} min")
    print(f"  RMSE:        {rmse:.2f} min")
    print(classification_report(y_true_cls, y_pred_cls,
                                target_names=le.classes_, zero_division=0))

baseline_pred = np.full_like(y_cls_test, np.bincount(y_cls_train).argmax())
baseline_reg  = np.full(len(y_reg_test), y_reg_train.mean())
evaluate("Majority-class baseline", y_cls_test, baseline_pred, y_reg_test, baseline_reg)

# Random Forest
print("\nTraining Random Forest...")
rf_cls = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
rf_cls.fit(X_train, y_cls_train)
rf_reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_reg_train)
evaluate("Random Forest", y_cls_test, rf_cls.predict(X_test),
         y_reg_test, rf_reg.predict(X_test))

# XGBoost
print("\nTraining XGBoost...")
xgb_cls = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, verbosity=0,
)
xgb_cls.fit(X_train, y_cls_train)
xgb_reg = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)
xgb_reg.fit(X_train, y_reg_train)
evaluate("XGBoost", y_cls_test, xgb_cls.predict(X_test),
         y_reg_test, xgb_reg.predict(X_test))

# LightGBM
print("\nTraining LightGBM...")
lgb_cls = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=42, verbosity=-1,
)
lgb_cls.fit(X_train, y_cls_train)
lgb_reg = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=-1,
)
lgb_reg.fit(X_train, y_reg_train)
evaluate("LightGBM", y_cls_test, lgb_cls.predict(X_test),
         y_reg_test, lgb_reg.predict(X_test))

# ── 5. Summary table ──────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
header = f"{'Model':<28} {'Acc':>6} {'W-F1':>6} {'M-F1':>6} {'MAE':>7} {'RMSE':>7}"
print(header)
print("-" * 65)
for name, r in results.items():
    print(f"{name:<28} {r['accuracy']:>6.3f} {r['weighted_f1']:>6.3f} "
          f"{r['macro_f1']:>6.3f} {r['mae']:>6.2f}m {r['rmse']:>6.2f}m")
print("=" * 65)

# Feature importance from best tree model (XGBoost)
print("\nXGBoost feature importances (classification):")
fi = pd.Series(xgb_cls.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in fi.items():
    print(f"  {feat:<25} {imp:.4f}")

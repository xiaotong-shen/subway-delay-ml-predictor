# TTC Subway Delay Prediction Neural Network - Implementation Notes

## Overview
This document outlines the key design decisions, implementation choices, and learning outcomes from developing the TTC subway delay prediction neural network system.

Note: Most of the changes made to the neuralnet.py file is in consideration that the prediction data is going to be used a reference for user queries.

---

## 0. Post-Datathon Improvements & Bug Fixes

After the datathon, I added formal benchmarking and found several issues that needed fixing before any metrics were meaningful.

### Bugs Fixed

**Bug 1 — Date features extracted from wrong column.** `month`, `day_of_week`, `is_weekend`, `is_holiday_season`, and `is_back_to_school` were all pulling from `Time_dt` (parsed from `HH:MM`) instead of `Date_dt`. Parsing `"13:45"` with `%H:%M` defaults to `1900-01-01`, making those five features constant across all rows. Fixed to use `Date_dt`.

**Bug 2 — Softmax + CrossEntropyLoss double application.** The severity head had `Softmax` as its final activation, but `CrossEntropyLoss` applies `log_softmax` internally. Stacking them pushes values into `log(~0)` = `-inf`, producing `NaN` loss silently from epoch 1. The model appeared to train (no error thrown) but never updated weights. Fixed by removing `Softmax` from the model head.

**Bug 3 — 2024 date format normalization.** The 2024 CSV uses `/` separators (`2024/01/01`) while 2021–2023 use `-`. After `pd.concat`, pandas infers format from the first rows and silently coerces the 2024 dates to `NaT`, dropping 24K records from training. Fixed by normalizing separators before parsing.

### Key Accuracy Improvement — Aggregate Before Training

Training on individual delay records is the wrong target for this use case. A single train's delay severity is largely determined by the specific incident (signal failure, passenger assistance, etc.), not by the temporal and station features the model has access to. That's fundamentally unpredictable.

What *is* learnable is the **average pattern**: "Bloor-Yonge during morning rush on weekdays in September averages 7.5 minutes of delay." The pre-computed lookup the frontend uses is exactly this kind of average — so training data should match.

Fix: aggregate by operational context flags `(station, is_morning_rush, is_evening_rush, is_weekend, is_holiday_season, is_back_to_school)` → 1,171 groups averaging 26 events each. Train on group averages. This halved the MAE and improved all classification metrics.

**Why not aggregate by (station, hour, day_of_week, month)?** That's 70 × 24 × 7 × 12 = 141K possible slots against 31K records — average coverage under 1 event per slot. Almost no smoothing happens. The coarser flag-based grouping gives 1,171 groups with real density.

### Severity Bin Rebalancing

Original bins `(-0.1, 1, 5, 15, ∞)` were designed for raw individual delay events. After aggregation, group averages compress toward the center (averages of 26 events don't produce 1-minute or 30-minute averages). The original bins put 95%+ of groups into Minor.

New bins `(-0.1, 5.0, 9.0, ∞)` are set at the 25th and 75th percentiles of group average delays, giving roughly a 25/60/15 class split — all classes are represented and learnable.

### Final Benchmark Results

| Metric | Buggy datathon model | Fixed + aggregated |
|---|---|---|
| Model accuracy | ~0% (NaN loss, no learning) | **60.4%** |
| Weighted F1 | — | **0.588** |
| Macro F1 | — | **0.458** |
| Delay length MAE | — | **2.09 min** |
| RMSE | — | **3.61 min** |
| Majority-class baseline | 60.7% | 59.1% |



---

## 1. Categorical vs Continuous Temporal Features

### **Key Decision: Categorical Approach**
**Question**: How do categorical patterns (weekends, rush hours, seasonal changes) impact TTC delays?

**Implementation Choice**: Categorical features instead of continuous datetime

### **Why Categorical Approach:**
- **Avoids artificial relationships**: Continuous datetime creates false "closeness" between December 31st and January 1st
- **Fundamental categorical patterns**: TTC delays follow distinct categorical patterns (Monday rush ≠ Friday rush)
- **Better neural network learning**: Categorical features are easier for neural networks to learn and interpret
- **Actionable insights**: "Weekends have 30% fewer delays" is more actionable than complex continuous relationships
- **Domain-specific patterns**: Captures real TTC operational realities

### **Features Implemented:**
```python
# Temporal Features (10 total)
- time_norm: Normalized time within TTC operating hours (0.0-1.0)
- month: Month of year (1-12) for seasonal patterns
- day_of_week: Day of week (0=Monday, 6=Sunday) for weekly patterns  
- is_weekend: Binary flag for weekend service patterns
- is_morning_rush: Binary flag for 7-9 AM rush hour
- is_evening_rush: Binary flag for 4-6 PM rush hour
- is_holiday_season: Binary flag for December/January patterns
- is_back_to_school: Binary flag for September patterns
```

---

## 2. Comprehensive Prediction Dataset

### **Full Coverage Implementation:**
- **All 12 months**: January through December
- **All 7 days**: Monday through Sunday
- **All 50 time points**: 6:00 AM to 1:30 AM (TTC operating hours)
- **All stations**: Every TTC subway station

### **Why Full Coverage:**
- **User expectations**: Users want predictions for any date/time they choose
- **No missing data**: Frontend can always find a prediction for user queries
- **Complete pattern analysis**: Captures all seasonal and weekly variations
- **Production ready**: Handles real-world user scenarios

### **Prediction Structure:**
```python
{
    'station': int,              # Station ID
    'station_name': str,         # "Bloor-Yonge Station"
    'month': int,                # 1-12
    'day_of_week': int,          # 0-6 (Mon-Sun)
    'is_weekend': int,           # 0 or 1
    'time_norm': float,          # 0.0-1.0
    'time_hhmm': str,            # "06:00" to "01:30"
    'delay_severity': str,       # "Minimal", "Minor", "Moderate", "Severe"
    'delay_length': float        # Minutes
}
```

---

## 3. Frontend Integration Design

### **Prediction Dataset as Reference:**
The prediction dataset serves as the **lookup table** for the frontend:

```
User Input: "Bloor-Yonge Station, February 15th, 8:30 AM"
    ↓
Frontend Query: month=2, day_of_week=2, time_norm=0.13
    ↓
Dataset Lookup: Find matching prediction
    ↓
Response: "Expected 8.5 minute moderate delay"
```

### **Key Benefits:**
- **Fast response**: No real-time model inference needed
- **Consistent results**: Same query always returns same prediction
- **Scalable**: Can handle thousands of concurrent users
- **Offline capable**: Predictions work without model server

---

## 4. Neural Network Architecture Changes

### **Model Enhancements:**
- **Input dimensions**: 3 → 10 temporal features
- **Variable naming**: `time_*` → `temporal_*` for clarity
- **Feature engineering**: Rich temporal context for better predictions

### **Architecture Details:**
```python
class MultiOutputModel(nn.Module):
    def __init__(self, temporal_features=10):  # Was 3
        # Input: 10 temporal + 16 station embedding = 26 features
        input_dim = temporal_features + embedding_dim
```
---

## 6. Learning Outcomes

### **Technical Insights:**
- **Feature engineering matters**: Categorical features often outperform continuous for domain-specific problems
- **Data type consistency**: PyTorch is sensitive to dtype mismatches
- **Memory vs functionality**: Full coverage datasets are worth the memory cost for production systems

### **Domain Insights:**
- **TTC patterns**: Weekdays vs weekends, rush hours, seasonal effects are crucial
- **User needs**: Complete coverage is essential for real-world applications
- **Operational reality**: Transit systems follow categorical, not continuous, patterns

### **Product Insights:**
- **User expectations**: Users want predictions for any date/time, not just samples
- **Actionable data**: Categorical features provide clearer insights for operators
- **Scalability**: Pre-computed predictions enable fast, scalable frontend responses

---

## 7. File Format Changes

### **From Jupyter Notebook to Python Script:**
- **Better version control**: Easier to track changes in .py files
- **Production deployment**: Python scripts are easier to deploy
- **Code organization**: Cleaner structure without cell outputs
- **Reproducibility**: More reliable execution across environments

---

## 9. Performance Metrics

### **Model Performance:**
- **Multi-task learning**: Simultaneous classification (severity) and regression (length)
- **Balanced loss**: 60% severity classification, 40% delay length regression
- **Early stopping**: Prevents overfitting with patience-based stopping
- **Learning rate scheduling**: Adaptive learning rate for better convergence

### **System Performance:**
- **Prediction coverage**: 100% of possible user queries
- **Response time**: Instant lookup from pre-computed dataset
- **Scalability**: Handles all TTC stations and temporal combinations
- **Reliability**: Robust error handling and data validation 
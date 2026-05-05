# TTC Subway Delay Predictor

**[Live Demo](#)** · [GitHub](https://github.com/xiaotong-shen/subway-delay-ml-predictor) · Python · scikit-learn · Streamlit · Plotly · Pandas

---

## Origin: SDSS Datathon 2025

This project started as a 24-hour group submission for the **2025 SDSS (Students in Data Science and Statistics) Datathon**. The prompt was a dataset of TTC subway delay records — figure out what you can do with it.

Our team built a proof-of-concept: a basic neural network that could predict whether a delay was likely at a given station and time. It worked, but it was rough — a Jupyter notebook model, weak UI, no way for a non-technical user to actually use it.

The core idea was good. So I kept going.

---

## What I Built After

After the datathon, I rebuilt this project independently from scratch — new model, new frontend, new data pipeline. The datathon gave me the idea and the initial architecture; everything in this repo is my own extension of it.

### The Model

The datathon model used a single normalized time float as input. That felt wrong — a continuous timestamp treats 11:55 PM and 12:05 AM as close to each other, and treats a Tuesday rush hour the same as a Saturday afternoon. Transit delays don't work like that.

So I redesigned the features around **operational context flags**:

```
is_morning_rush   → 7–9 AM
is_evening_rush   → 4–6 PM
is_weekend        → Saturday / Sunday
is_holiday_season → December / January
is_back_to_school → September
+ station identity, time_norm, month, day_of_week
+ station_mean_delay (explicit historical prior per station)
```

Training data: **~31,000 delay records** spanning 2021–2024, capped at 60-minute delays to exclude rare extreme incidents that would skew group averages. Delays were then **aggregated by operational context** — 1,171 station-context groups averaging 26 events each — before training. The model predicts group averages rather than individual events.

### Model Selection

I benchmarked four approaches on the same aggregated dataset with an 80/20 train-test split:

| Model | Accuracy | Weighted F1 | Macro F1 | MAE |
|---|---|---|---|---|
| Majority-class baseline | 59.1% | 0.440 | 0.248 | 2.15 min |
| Neural Network (PyTorch) | 60.4% | 0.588 | 0.458 | 2.09 min |
| XGBoost | 62.6% | 0.605 | 0.491 | 2.17 min |
| LightGBM | 58.7% | 0.587 | 0.485 | 2.20 min |
| **Random Forest** | **65.1%** | **0.629** | **0.502** | **2.07 min** |

Random Forest won across every classification metric and matched the NN on regression MAE. Tree-based models tend to outperform neural networks on small tabular datasets — this confirmed that. The production model is Random Forest.

**Why aggregation was the key insight:** training on individual delay records is the wrong target. A single train's delay severity depends on the specific incident (signal failure, door issue, passenger assistance), not the time or station. What *is* learnable is the average pattern: "Bloor-Yonge during morning rush on weekdays averages 7 minutes." Aggregating to 1,171 groups halved the regression MAE and lifted all classification metrics. The severity bins were set at the 25th/75th percentile of group averages (~5 and ~9 min) to give a roughly balanced three-class split.

**Feature importances (XGBoost):** `n_events` (group data density) and `station_mean_delay` (explicit station prior) ranked first and second — confirming that adding the historical station average as an explicit feature was the right call rather than relying solely on a learned embedding.

**Three bugs fixed in the datathon code before any measurement was meaningful:** date features (`month`, `day_of_week`, `is_weekend`) were extracted from the time column instead of the date column, making them constant; `Softmax` before `CrossEntropyLoss` produced silent `NaN` loss every epoch; the 2024 CSV used `/` date separators causing pandas to silently drop 24K records.

### The Frontend Decision: Pre-Compute Everything

The biggest architectural decision was how the frontend talks to the model. The obvious path is real-time inference — user picks a station and time, the app runs the model, returns a result. Simple, but it introduces latency and a dependency on a model server.

I went the other direction: **pre-compute predictions for every possible user query** and store them as a lookup table.

Every combination of station × month × day-of-week × hour gets a row in `enriched_predictions_full.csv`. At runtime, the app just filters a DataFrame — no inference, no model loaded in memory. The result is instant responses, no cold-start problem, and a frontend that could theoretically run completely offline.

The tradeoff is a larger static file and predictions that can't adapt to real-time conditions. For a historical pattern tool, that's a fine tradeoff.

### The Interface

I built the frontend in **Streamlit** with custom CSS for a dark-mode aesthetic. The main view is a **Plotly scatter map** of Toronto where every TTC station is a dot — color (red → green) encodes delay likelihood, and dot size amplifies the difference so it's actually readable at a glance. Both channels encoding the same variable is intentional: color alone is hard to compare across a busy map.

Controls live in the sidebar: hour slider, month selector, day-of-week selector, station picker. Changing any of them instantly re-filters the pre-computed dataset and re-renders the map. On the right panel, a selected station shows its current prediction (likelihood, severity, expected delay in minutes) alongside a **daily timeline chart** — so you can see not just "is Tuesday at 8 AM bad?" but the full shape of the day for that stop.

![Main Interface](images/main-interface.png)
![Interactive Map](images/interactive-map.png)
![User Controls](images/user-controls.png)
![Risk Assessment](images/risk-assessment.png)

---

## Stack

| Layer | Tech |
|---|---|
| Model | scikit-learn Random Forest (classifier + regressor) |
| Data | Pandas, ~31K records aggregated to 1,171 groups (2021–2024) |
| Benchmarking | NN, XGBoost, LightGBM, RF compared on same split |
| Frontend | Streamlit, Plotly, custom CSS |
| Serving | Pre-computed CSV lookup, no runtime inference |

---

## Run It Locally

```bash
git clone https://github.com/xiaotong-shen/subway-delay-ml-predictor
cd subway-delay-ml-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain: run `python train_model.py` from the `python notebooks/` directory (Random Forest, ~10s). The original neural network is in `neuralnet.py` for reference. Training data CSVs are not included in the repo — download from [open.toronto.ca](https://open.toronto.ca/dataset/ttc-subway-delay-data/).

---

## What I'd Do Next

- **Delay cause classification** — the raw data has delay code columns (mechanical, signal, passenger assistance) that I dropped. Training a separate classifier on those codes and surfacing it in the UI would make the severity predictions much more interpretable.
- **SMOTE or oversampling for the Severe class** — Severe delays (F1: 0.19) are the hardest to predict and the most useful to catch. Synthetic oversampling during training could improve recall without sacrificing precision on the other classes.
- **Live TTC feed layer** — the TTC open data API publishes real-time delay events. Layering that on top of the historical predictions would let the map show "predicted risk" vs "happening right now."
- **Station ranking panel** — the analysis is already built and commented out in `app.py`; just needs wiring up to the UI.

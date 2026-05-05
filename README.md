# TTC Subway Delay Predictor

**[Live Demo](#)** · [GitHub](https://github.com/xiaotong-shen/subway-delay-ml-predictor) · Python · PyTorch · Streamlit · Plotly · Pandas

---

## Origin: SDSS Datathon 2025

This project started as a 24-hour group submission for the **2025 SDSS (Students in Data Science and Statistics) Datathon**. The prompt was a dataset of TTC subway delay records — figure out what you can do with it.

Our team built a proof-of-concept: a basic neural network that could predict whether a delay was likely at a given station and time. It worked, but it was rough — a Jupyter notebook model, weak UI, no way for a non-technical user to actually use it.

The core idea was good. So I kept going.

---

## What I Built After

After the datathon, I rebuilt this project independently from scratch — new model, new frontend, new data pipeline. The datathon gave me the idea and the initial architecture; everything in this repo is my own extension of it.

### The Model

The datathon model used a single normalized time float as input. That felt wrong to me — a continuous timestamp treats 11:55 PM and 12:05 AM as close to each other, and treats a Tuesday rush hour the same as a Saturday afternoon. Transit delays don't work like that.

So I redesigned the feature set around **categorical temporal patterns**:

```
time_norm        → Normalized position within TTC operating hours (6AM–1:30AM)
month            → Seasonal effects (back-to-school in September, holiday chaos in December)
day_of_week      → Weekly rhythm (Monday ≠ Friday ≠ Sunday)
is_weekend       → Binary service pattern flag
is_morning_rush  → 7–9 AM flag
is_evening_rush  → 4–6 PM flag
is_holiday_season → December/January flag
is_back_to_school → September flag
```

Combined with a **16-dimensional station embedding**, the model takes 26 inputs total — up from 3 in the original. The architecture is a multi-output network with two heads: one for **delay severity classification** (Minimal / Minor / Moderate / Severe) and one for **delay length regression** (minutes). Loss is weighted 60/40 in favour of the classification head. I added early stopping and a learning rate scheduler to keep it from overfitting.

Training data: **277,200 records** spanning 2021–2024 across all active TTC subway lines. I excluded the SRT (discontinued) and a handful of sparse/unclassifiable line codes that would have added noise without meaningful signal.

### Model Performance

The model is evaluated against a majority-class baseline — the accuracy you'd get by always predicting the most common severity category ("Moderate").

| Metric | Value |
|---|---|
| Majority-class baseline accuracy | 59.1% |
| **Model accuracy** | **60.4%** ✓ beats baseline |
| **Weighted F1** | **0.588** |
| **Macro F1** | **0.458** |
| Random 3-class baseline (F1) | 0.333 |
| **Delay length MAE** | **2.09 min** |
| Delay length RMSE | 3.61 min |

The model beats the majority baseline on both accuracy and F1. Macro F1 (0.458 vs 0.333 random) is the honest number — it weights all three classes equally and doesn't flatter the majority class. The regression head is the most reliable output: a MAE of 2.09 minutes on average delay prediction is practically useful for a transit tool.

**What changed from the datathon model to get here:**

*Three bugs fixed first.* The datathon code extracted `month`, `day_of_week`, and `is_weekend` from the time column (`HH:MM`) instead of the date column — parsing `"13:45"` with `%H:%M` defaults to `1900-01-01`, making five features constant noise. The severity head had `Softmax` before `CrossEntropyLoss`, which applies its own log-softmax — the double application pushed values into `log(~0)` = `-inf`, producing `NaN` loss silently for every epoch. The 2024 CSV uses `/` date separators while 2021–2023 use `-`; pandas infers format from the first rows and silently dropped 24K records.

*Then the key insight for accuracy:* training on individual delay records is the wrong target. A single train's delay severity is mostly determined by the specific incident, not by the time or station — that's fundamentally unpredictable from the features available. What *is* learnable is the average pattern: "Bloor-Yonge during morning rush on weekdays averages 7 minutes." So I aggregated training data by operational context `(station × is_morning_rush × is_evening_rush × is_weekend × seasonal flags)` — 1,171 groups averaging 26 events each — and trained the model to predict group averages. This halved the MAE and improved all classification metrics. The severity bins were also rebalanced to the 25th/75th percentile split of group averages (~5 and ~9 min), giving a roughly balanced class distribution rather than 60% minority.

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
| Model | PyTorch (multi-output NN, station embeddings) |
| Data | Pandas, scikit-learn, 277K records (2021–2024) |
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

To retrain the model, run `python notebooks/neuralnet.py` from the `python notebooks/` directory. Training data CSVs are included.

---

## What I'd Do Next

- Pull live TTC delay feeds (the open data API exists) to add a "right now" layer on top of the historical predictions
- Expand the model to predict delay *cause* categories (mechanical, signal, passenger assistance) — the delay code column in the raw data has this but I left it out of v1
- Surface the station ranking analysis I have commented out in `app.py` — it's built, just not exposed yet

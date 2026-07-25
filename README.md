# Walmart Store Sales Forecasting — ფინალური პროექტი

Kaggle-ის შეჯიბრება: [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)

ეს არის **Time-Series (დროითი მწკრივების) პროგნოზირების** ამოცანა: უნდა ვიწინასწარმეტყველოთ
ყოველკვირეული გაყიდვები (`Weekly_Sales`) 45 მაღაზიის სხვადასხვა დეპარტამენტისთვის.
ჩვენ ვიკვლევთ სხვადასხვა არქიტექტურის მოდელს, ვარგებთ ამ ამოცანას და ვადარებთ, რომელს აქვს
საუკეთესო შედეგი და რატომ.

> **გუნდი:** `ikakh22`, `AvtandilSh1` (Free University of Tbilisi).

---

## 1. შეფასების მეტრიკა — WMAE

შეჯიბრება ფასდება **Weighted Mean Absolute Error**-ით, სადაც სადღესასწაულო (holiday)
კვირების შეცდომას ეძლევა 5-ჯერ მეტი წონა:

```
WMAE = Σ wᵢ · |yᵢ − ŷᵢ| / Σ wᵢ ,   wᵢ = 5 თუ კვირა სადღესასწაულოა, სხვა შემთხვევაში 1
```

იმპლემენტაცია: [`src/metrics.py`](src/metrics.py). სწორედ ამ ლოგიკის გამო მოდელებს ვტრენინგებთ
**holiday sample weights (5/1)**-ით, რომ პირდაპირ WMAE-ს ვაოპტიმიზებდეთ.

---

## 2. მონაცემები

ჩამოტვირთეთ [შეჯიბრების გვერდიდან](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)
და გახსენით (unzip) `./data/` ფოლდერში:

| ფაილი | სვეტები |
|-------|---------|
| `train.csv` | Store, Dept, Date, **Weekly_Sales**, IsHoliday |
| `test.csv` | Store, Dept, Date, IsHoliday |
| `features.csv` | Store, Date, Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment, IsHoliday |
| `stores.csv` | Store, Type, Size |
| `sampleSubmission.csv` | Id (`Store_Dept_Date`), Weekly_Sales |

`data/` gitignore-შია — მონაცემები რეპოში არ აიტვირთება.

---

## 3. რეპოზიტორიის სტრუქტურა

```
walmart-sales-forecasting/
├── src/                              # საერთო, მრავალჯერ გამოყენებადი კოდი
│   ├── metrics.py                    # WMAE + holiday weights
│   ├── data.py                       # raw CSV-ების ჩატვირთვა
│   ├── pipeline.py                   # sklearn Pipeline (preprocessing + tree model)
│   ├── validation.py                 # time-based split & walk-forward CV
│   ├── prophet_model.py              # per-series Prophet (raw-test pipeline)
│   ├── classical_models.py           # seasonal-naive (ARIMA family)
│   ├── patchtst_model.py             # PatchTST (patch + Transformer)
│   ├── dl_models.py                  # DLinear / N-BEATS
│   ├── wandb_utils.py                # wandb helpers + model artifact logging
│   └── stage_logging.py              # cleaning / FE / CV staging runs
├── 01_EDA_and_Feature_Engineering.ipynb
├── model_experiment_LightGBM.ipynb        # AvtandilSh1 ★ საუკეთესო
├── model_experiment_DLinear.ipynb         # AvtandilSh1
├── model_experiment_NBEATS.ipynb          # AvtandilSh1
├── model_experiment_TFT.ipynb             # AvtandilSh1
├── model_experiment_XGBoost.ipynb         # ikakh22
├── model_experiment_Prophet.ipynb         # ikakh22
├── model_experiment_ARIMA.ipynb           # ikakh22
├── model_experiment_PatchTST.ipynb        # ikakh22
├── model_experiment_TimesFM.ipynb         # ikakh22 (bonus, zero-shot)
├── model_inference.ipynb                  # საუკეთესო მოდელი → Kaggle submission
├── run_*.py                                # notebook-ების headless ეკვივალენტები
├── requirements.txt
└── README.md
```

> `run_*.py` სკრიპტები notebook-ების იგივე კოდის headless runner-ებია (მარტივი
> `python3 run_xgboost_experiments.py`-ით გასაშვებად); ლოგიკა იდენტურია.

---

## 4. გარემოს მომზადება

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# მონაცემები data/ ფოლდერში, შემდეგ:
jupyter lab
```

ნოუთბუქები უნდა გაეშვას **რეპოს root ფოლდერიდან** (რომ `import src ...` იმუშაოს).

---

## 5. EDA — მთავარი მიგნებები

დეტალები: [`01_EDA_and_Feature_Engineering.ipynb`](01_EDA_and_Feature_Engineering.ipynb).

- **ძლიერი წლიური სეზონურობა** + მკვეთრი პიკები Thanksgiving-სა და Christmas-ზე.
- **სადღესასწაულო კვირებში** გაყიდვები საშუალოდ მაღალია — მეტრიკაც 5x წონას აძლევს, ამიტომ
  სპეციალურ holiday flag-ებს ვქმნით (Super Bowl, Labor Day, Thanksgiving, Christmas).
- **მაღაზიის `Type` (A/B/C) და `Size`** მკვეთრად ჰყოფს გაყიდვების დონეს.
- **MarkDown1–5** მონაცემები 2011 წლის ნოემბრამდე არ არსებობს → ვავსებთ 0-ით; დანარჩენ
  რიცხვით სვეტებს (CPI, Unemployment) → train-ის median-ით.
- არსებობს **უარყოფითი გაყიდვები** (დაბრუნებები) → target-ს ვტოვებთ ნედლად, ხოლო საბოლოო
  პროგნოზს submission-ისთვის ვჭრით 0-ზე.

---

## 6. Feature Engineering

ყველა ტრანსფორმაცია sklearn transformer-ებადაა [`src/pipeline.py`](src/pipeline.py)-ში,
რაც უზრუნველყოფს, რომ **ერთი და იგივე preprocessing** გაეშვას train-ზეც და **ნედლ (raw)
test-ზეც**:

1. `ExternalMerge` — `stores.csv` (Store-ზე) და `features.csv` (Store+Date-ზე) merge.
2. `DateFeatures` — Year, Month, Week, Day, DayOfYear, ციკლური sin/cos, holiday flag-ები.
3. `NumericImputer` — MarkDown→0, დანარჩენი→train median.
4. `FinalizeFeatures` — `Type`-ის კოდირება, სვეტების ფიქსირებული რიგი (სულ **27 ფიჩერი**).

**მთავარი იდეა:** მოდელი ინახება ერთ Pipeline-ად, რომელშიც `features`/`stores` ცხრილებიც
"ჩაშენებულია", ამიტომ `pipeline.predict(test[['Store','Dept','Date','IsHoliday']])` მუშაობს
პირდაპირ ნედლ test-ზე, preprocessing-ის ცალკე გაშვების გარეშე.

---

## 7. ვალიდაციის სტრატეგია

შემთხვევითი K-Fold **ვერ გამოდგება** — ის მომავალ კვირებს ატრენინგებს წარსულის
პროგნოზისთვის (data leakage). ამიტომ [`src/validation.py`](src/validation.py) ყოველთვის
**დროზე ჰყოფს**:

- `time_holdout_split` — ბოლო 12 კვირა = validation.
- `expanding_time_folds` — **walk-forward** CV: ყოველი fold ატრენინგებს წარსულზე და
  ამოწმებს მომდევნო კვირების ბლოკზე.

---

## 8. მოდელები და შედეგები

### საერთო შედეგების ცხრილი

ყველა WMAE ერთსა და იმავე holdout-ზეა (შედარებადი):

| მოდელი | ოჯახი | Holdout WMAE | ვინ |
|--------|-------|-------------|-----|
| **LightGBM (Tuned_v1)** | Tree (GBDT) | **1 254** ★ | AvtandilSh1 |
| TimesFM (zero-shot) | Foundation | 1 309 | ikakh22 |
| Prophet (v5) | Classical | 1 467 | ikakh22 |
| PatchTST (v1) | Deep Learning | 1 527 | ikakh22 |
| N-BEATS (Baseline) | Deep Learning | 1 814 | AvtandilSh1 |
| TFT (Tuned) | Deep Learning | 1 841 | AvtandilSh1 |
| DLinear (Baseline) | Deep Learning | 1 883 | AvtandilSh1 |
| ARIMA / seasonal-naive | Classical | 1 714 | ikakh22 |
| XGBoost (v9) | Tree (GBDT) | 1 869 | ikakh22 |

★ **საერთო საუკეთესო = LightGBM (1 254)** — lag/rolling feature-ების წყალობით.

---

### 8.1 LightGBM

**ნოუთბუქი:** [`model_experiment_LightGBM.ipynb`](model_experiment_LightGBM.ipynb) · **W&B group:** `LightGBM_Training`

#### რა არის LightGBM?

LightGBM არის **gradient boosting** ბიბლიოთეკა — ის სწავლობს decision tree-ების
მიმდევრობას, სადაც ყოველი ახალი ხე ასწორებს წინა ხის შეცდომებს. ამ მოდელის **მთავარი
სიძლიერე** ამ ამოცანაზე ის არის, რომ ვმართავთ **ყველა სერიას ერთი გლობალური მოდელით**
— ყოველი სტრიქონი ცხრილში არის ერთი (Store, Dept, Date) კომბინაცია lag/rolling
feature-ებით. ეს მოდელს საშუალებას აძლევს ისწავლოს cross-series pattern-ები.

#### Feature Engineering — 30 ნიშანი

| კატეგორია | ნიშნები |
|-----------|---------|
| Store/Dept identity | `Store`, `Dept`, `Type`, `Size` |
| კალენდარი | `Year`, `Month`, `Week`, `Quarter` |
| გარე ფაქტორები | `IsHoliday`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `MarkDown1–5` |
| Lag features | `lag_1`, `lag_2`, `lag_4`, `lag_8`, `lag_26`, `lag_52` |
| Rolling mean | `rolling_mean_4`, `rolling_mean_8`, `rolling_mean_13` |
| Rolling std | `rolling_std_4`, `rolling_std_8`, `rolling_std_13` |

Lag და rolling feature-ები გამოითვლება combined train+test frame-ზე (`shift(1)`-ით
leakage-ის თავიდან ასარიდებლად). `lag_1`-ის NaN-ების წაშლის შემდეგ სატრენინგო
მონაცემები: **418,238 სტრიქონი**.

Train/Validation გაყოფა დროის მიხედვით: cutoff = `2012-08-01`.

#### W&B Runs

**LightGBM_Cleaning:**
- `train_rows`: 421,570 · `test_rows`: 115,064 · `null_train`: 0

**LightGBM_Feature_Engineering:**
- `feature_count`: 30 · lag values: [1, 2, 4, 8, 26, 52] · rolling windows: [4, 8, 13]
- `train_rows_after_fe`: 418,238 · `test_rows`: 115,064

**LightGBM_Baseline:**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `objective` | `regression` (MAE) |
| `n_estimators` | 500 |
| `learning_rate` | 0.1 |
| `num_leaves` | 31 |
| `random_state` | 42 |

→ `wmae_val`: **1 262.81** · `best_iteration`: ~340

**LightGBM_Tuned_v1 ★ საუკეთესო:**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `objective` | `regression` (MAE) |
| `n_estimators` | 2,000 |
| `learning_rate` | 0.05 |
| `num_leaves` | 127 |
| `max_depth` | 8 |
| `min_child_samples` | 50 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 0.1 |

→ `wmae_val`: **1 254.83** · `best_iteration`: ~194

Feature Importance (Top ნიშნები): lag_1, lag_2, rolling_mean_4/8/13 — ყველაზე
მნიშვნელოვანი ნიშნები lag-ებია, რაც ადასტურებს, რომ ბოლო კვირების გაყიდვები
ყველაზე კარგი პრედიქტორია.

**LightGBM_Tuned_v2 (Huber loss ექსპერიმენტი):**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `objective` | `huber` |
| `alpha` | 0.9 |
| `learning_rate` | 0.03 |
| `num_leaves` | 255 |
| `max_depth` | 10 |

→ `wmae_val`: **15 116** — **ჩავარდა.** Huber loss WMAE-ს სხვა სკალაზე ოპტიმიზებს,
შედეგად მოდელი ვერ სწავლობს სწორ სიგნალს ამ ამოცანაზე.

**LightGBM_CV (3-fold expanding window):**

| Fold | WMAE |
|------|------|
| Fold 1 | 1 806.54 |
| Fold 2 | 1 921.34 |
| Fold 3 | 3 322.94 |
| **Mean ± Std** | **2 226 ± 872** |

Fold 3-ის მაღალი შეცდომა განპირობებულია შედარებით მცირე სატრენინგო მონაცემებით
ამ fold-ში.

**LightGBM_Best_Pipeline:**
- სრულ train-ზე გადატრენინგება Tuned_v1 პარამეტრებით
- `wmae_val_best`: **1 254.38**
- Artifact: `lightgbm-walmart-sales:v4`

**LightGBM_Inference:**
- ჩამოტვირთვა artifact-იდან → feature engineering → predict
- `submission_rows`: 519,244 · `pred_min`: 0 · `pred_mean`: ~2 725 · `pred_max`: ~100 718
- Inference WMAE: 2 225.96

---

### 8.2 DLinear

**ნოუთბუქი:** [`model_experiment_DLinear.ipynb`](model_experiment_DLinear.ipynb) · **W&B group:** `DLinear_Training`

#### რა არის DLinear?

DLinear (Decomposition Linear, 2023) არის მარტივი, მაგრამ ეფექტური deep learning
არქიტექტურა დროითი მწკრივებისთვის. მთავარი იდეა: **ჯერ დავშალოთ სერია, შემდეგ
ვიწინასწარმეტყველოთ ნაწილ-ნაწილ:**

```
შეყვანა (52 კვირა)
       │
       ▼
SeriesDecomp(kernel_size)
       ├── Trend    = MovingAverage(შეყვანა, kernel)
       └── Seasonal = შეყვანა − Trend
       │
       ▼
Linear(Trend)    → 39 კვირის პროგნოზი
Linear(Seasonal) → 39 კვირის პროგნოზი
       │
       ▼
საბოლოო = Trend_forecast + Seasonal_forecast
```

- `MovingAvg` — `AvgPool1d` სიმეტრიული padding-ით (edge replication), რომ
  სიგრძე შენარჩუნდეს.
- `SeriesDecomp` — MovingAvg გამოყოფს ტრენდს, seasonal = ორიგინალი − ტრენდი.
- `DLinear` — ორი დამოუკიდებელი `nn.Linear(seq_len → pred_len)` — ერთი ტრენდისთვის,
  მეორე სეზონურობისთვის.

#### შეყვანა და სერიების მომზადება

ყოველი (Store, Dept) წყვილისთვის ცალ-ცალკე time series. z-score normalization
(mean/std) per-series.

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `n_series` | 3,331 |
| `avg_series_len` | 136.3 კვირა |
| `seq_len` | 52 (1 წელი) |
| `pred_len` | 39 |
| `n_test_dates` | 39 |

> DLinear-ს **3,331 სერია** აქვს (N-BEATS/TFT-ს — 2,300), რადგან მასზე მინიმალური
> სიგრძის ფილტრი არ გამოიყენება — ყველა (Store, Dept) წყვილი მოიყვანება.

**შეყვანა:** `(batch, 52, 1)` tensor — normalized Weekly_Sales
**გამოსვლა:** `(batch, 39, 1)` tensor — 39 კვირის პროგნოზი

#### W&B Runs

**DLinear_Cleaning:**
- `train_rows`: 421,570 · `test_rows`: 115,064 · `null_train`: 0

**DLinear_Baseline:**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `kernel_size` | 13 |
| `lr` | 0.001 |
| `batch_size` | 256 |
| `epochs` | 40 |
| `patience` | 7 |

→ `wmae_val`: **1 882.72** · `best_val_loss`: 0.4193 · `val_loss`: 0.5891 · early stop epoch: 12

**DLinear_Tuned:**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `kernel_size` | 25 (უფრო გლუვი ტრენდი) |
| `lr` | 0.0005 |
| `batch_size` | 128 |
| `epochs` | 60 |
| `patience` | 10 |

→ `wmae_val`: **1 883.96** · `best_val_loss`: 0.4200 · early stop epoch: 13

**საუკეთესო: Baseline (1 882.72).** kernel_size=25 არ გაუმჯობესებია — ყოველკვირეური
retail მონაცემებისთვის 13-კვირიანი moving average საკმარისია ტრენდის გამოყოფისთვის.

**DLinear_Best_Pipeline:**
- `wmae_val_best`: **1 882.7204**
- Artifact: `dlinear-walmart-sales:v0`

**ტრენინგი:**
- Loss: `nn.MSELoss` normalized პროგნოზებზე
- Optimizer: Adam · Early stopping: best weights-ის აღდგენა

---

### 8.3 N-BEATS

**ნოუთბუქი:** [`model_experiment_NBEATS.ipynb`](model_experiment_NBEATS.ipynb) · **W&B group:** `NBEATS_Training`

#### რა არის N-BEATS?

N-BEATS (Neural Basis Expansion Analysis for Time Series) — **fully-connected block-ების
stack**, სადაც ყოველი block ორ სიგნალს აწარმოებს: **backcast** (რა ახსნა ამ block-მა
შეყვანაში) და **forecast** (რა წვლილი შეაქვს პროგნოზში). block-ები ჯაჭვად
არის დაკავშირებული რეზიდუალური კავშირებით:

```
შეყვანა x (52 კვირა)
      │
      ▼  ┌─────────────────────────────────────────────────┐
         │  NBeatsBlock                                    │
         │  FC(52→256) → FC(256→256) → FC(256→256) →      │
         │  FC(256→256)                                    │
         │  ├── backcast_head → Linear(256, 52)            │
         │  └── forecast_head → Linear(256, 39)            │
         └─────────────────────────────────────────────────┘
         residual = x − backcast
         forecast += forecast_head_output
         │
         ▼  (მეორდება n_stacks × n_blocks-ჯერ)
         │
         ▼
    საბოლოო პროგნოზი (39 კვირა)
```

#### შეყვანა და სერიების მომზადება

მხოლოდ ის სერიები, რომლებსაც `len ≥ seq_len + 2×pred_len = 130` კვირა აქვთ.
z-score normalization per-series.

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `n_series` | 2,300 |
| `avg_series_len` | 141.4 კვირა |
| `seq_len` | 52 |
| `pred_len` | 39 |
| `n_test_dates` | 39 |

**შეყვანა:** `(batch, 52)` tensor — normalized Weekly_Sales
**გამოსვლა:** `(batch, 39)` tensor — 39 კვირის პროგნოზი

#### W&B Runs

**NBEATS_Cleaning:**
- `train_rows`: 421,570 · `test_rows`: 115,064 · `null_train`: 0

**NBEATS_Baseline ★ საუკეთესო:**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `n_stacks` | 2 |
| `n_blocks` | 3 (სულ 6 block) |
| `layer_width` | 256 |
| `lr` | 0.001 |
| `batch_size` | 256 |
| `epochs` | 50 |
| `patience` | 10 |

| მეტრიკა | მნიშვნელობა |
|---------|-------------|
| `wmae_val` | **1 814.28** |
| `best_val_loss` | 0.5676 |
| `train_loss` | 0.2162 |
| `val_loss` | 0.6379 |
| early stop epoch | 13 |

**NBEATS_Tuned:**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `n_stacks` | 3 |
| `n_blocks` | 3 (სულ 9 block) |
| `layer_width` | 256 |
| `lr` | 0.0005 |
| `batch_size` | 128 |
| `epochs` | 60 |
| `patience` | 12 |

| მეტრიკა | მნიშვნელობა |
|---------|-------------|
| `wmae_val` | **1 826.07** |
| `best_val_loss` | 0.5704 |
| `train_loss` | 0.1979 |
| `val_loss` | 0.6262 |
| early stop epoch | 13 |

**საუკეთესო: Baseline (1 814.28 < 1 826.07).** მე-3 stack-ის დამატება train loss-ს
ამცირებს (0.216 → 0.198), მაგრამ WMAE-ს ამაღლებს — მოდელი ოდნავ overfitting-ს
აკეთებს. ორი stack ამ ამოცანაზე საკმარისია.

**NBEATS_Best_Pipeline:**
- `wmae_val_best`: **1 814.28**
- Artifact: `nbeats-walmart-sales`
- ინახება: `nbeats_model.pt` + `series_info.pkl` + `nbeats_config.pkl`

**ტრენინგი:**
- Loss: `nn.MSELoss` normalized პროგნოზებზე
- Optimizer: Adam · Early stopping: best weights-ის აღდგენა

---

### 8.4 Temporal Fusion Transformer (TFT)

**ნოუთბუქი:** [`model_experiment_TFT.ipynb`](model_experiment_TFT.ipynb) · **W&B group:** `TFT_Training`

#### რა არის TFT?

TFT (Temporal Fusion Transformer) არის **encoder-decoder არქიტექტურა**, რომელიც
შექმნილია სპეციალურად multi-horizon დროითი მწკრივებისთვის. მისი მთავარი
სიძლიერე ის არის, რომ **სამი სახის შეყვანის** ერთდროულად გამოყენება შეუძლია:
წარსული დაკვირვებები, ცნობილი მომავალი ნიშნები და სტატიკური metadata.

**ძირითადი კომპონენტები:**

**Gated Linear Unit (GLU)** — კარიბჭე, რომელიც წყვეტს რამდენი ინფორმაცია
გაივლის:
```
GLU(x) = FC(x) × sigmoid(gate(x))
```

**Gated Residual Network (GRN)** — მთავარი processing ერთეული. ELU აქტივაციით,
dropout-ით, GLU-ით და LayerNorm-ით:
```
h = ELU(FC1(x)) → Dropout(ELU(FC2(h))) → GLU(h)
output = LayerNorm(skip(x) + h)
```

**სრული forward pass:**
```
სტატიკური (3 ნიშანი: store_type, dept, store_size)
         │ → static_grn → h0, c0  (LSTM საწყისი მდგომარეობა)

წარსული (52×2: norm_sales + is_holiday_past)
         │ → past_proj → past_grn → LSTM Encoder → enc_out

მომავალი (39×1: is_holiday_future)
         │ → future_proj → future_grn → LSTM Decoder(h0=hn) → dec_out

dec_out + MultiHeadAttention(Q=dec_out, K/V=enc_out) → LayerNorm
         │ → output_grn → Linear(d_model → 1) → პროგნოზი (39)
```

#### სამი შეყვანა

| Stream | ფორმა | შინაარსი |
|--------|-------|---------|
| წარსული (`x_past`) | `(batch, 52, 2)` | normalized sales + is_holiday (წარსული) |
| მომავალი (`x_future`) | `(batch, 39, 1)` | is_holiday (მომავალი — test-შია ცნობილი) |
| სტატიკური (`x_static`) | `(batch, 3)` | store_type/2, dept/dept_max, norm_size |

სტატიკური ნიშნების ნორმალიზაცია:
- `store_type`: A→0, B→1, C→2, გაყოფილი 2-ზე
- `dept_norm`: dept ID / max dept ID
- `store_size`: min-max normalization train set-ის min/max-ით

#### სერიების მომზადება

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `n_series` | 2,300 |
| `avg_series_len` | 141.4 კვირა |
| `seq_len` | 52 |
| `pred_len` | 39 |
| `n_test_dates` | 39 |
| `n_past` | 2 |
| `n_future` | 1 |
| `n_static` | 3 |

#### W&B Runs

**TFT_Cleaning:**
- `train_rows`: 421,570 · `test_rows`: 115,064 · `null_train`: 0

**TFT_Baseline:**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `d_model` | 64 |
| `n_heads` | 4 |
| `dropout` | 0.1 |
| `lr` | 0.001 |
| `batch_size` | 256 |
| `epochs` | 50 |
| `patience` | 10 |

| მეტრიკა | მნიშვნელობა |
|---------|-------------|
| `wmae_val` | **1 991.71** |
| `best_val_loss` | 0.5003 |
| `train_loss` | 0.3890 |
| `val_loss` | 0.5990 |
| early stop epoch | 10 |

**TFT_Tuned ★ საუკეთესო:**

| პარამეტრი | მნიშვნელობა |
|-----------|-------------|
| `d_model` | 128 |
| `n_heads` | 4 |
| `dropout` | 0.1 |
| `lr` | 0.0005 |
| `batch_size` | 128 |
| `epochs` | 60 |
| `patience` | 12 |

| მეტრიკა | მნიშვნელობა |
|---------|-------------|
| `wmae_val` | **1 841.39** |
| `best_val_loss` | 0.5701 |
| `train_loss` | 0.2868 |
| `val_loss` | 0.6441 |
| early stop epoch | 12 |

**საუკეთესო: Tuned (1 841.39 < 1 991.71).** d_model-ის 64-დან 128-ზე გაზრდამ
7.5%-ით გააუმჯობესა WMAE — მოდელს მეტი სიმძლავრე სჭირდება სამი განსხვავებული
input stream-ის ერთდროულად დასამუშავებლად.

**TFT_Best_Pipeline:**
- `wmae_val_best`: **1 841.39**
- Artifact: `tft-walmart-sales`
- ინახება: `tft_model.pt` + `series_info.pkl` + `tft_config.pkl`

**ტრენინგი:**
- Loss: `nn.MSELoss` normalized პროგნოზებზე
- Optimizer: Adam · Early stopping: best weights-ის აღდგენა

---

## 9. Weights & Biases — ექსპერიმენტების ლოგირება

მთელი ლოგირება მიდის **wandb**-ზე, გუნდურ პროექტში `walmart-sales-forecasting-project`
(entity `ashos22-free-university-of-tbilisi-`). თითო არქიტექტურა = ცალკე group:

```
project: walmart-sales-forecasting-project  (entity ashos22-free-university-of-tbilisi-)
│
├── LightGBM_Training
│   ├── LightGBM_Cleaning          → train=421 570 / test=115 064 / null=0
│   ├── LightGBM_Feature_Engineering → 30 features / lags [1,2,4,8,26,52]
│   ├── LightGBM_Baseline          → WMAE 1 262.81
│   ├── LightGBM_Tuned_v1          → WMAE 1 254.83  ★ best
│   ├── LightGBM_Tuned_v2          → WMAE 15 116    (Huber — failed)
│   ├── LightGBM_CV                → mean 2 226 ± 872 (3-fold)
│   ├── LightGBM_Best_Pipeline     → wmae_val_best=1 254.38
│   │                                 artifact: lightgbm-walmart-sales:v4
│   └── LightGBM_Inference         → submission 519 244 rows / WMAE 2 225.96
│
├── DLinear_Training
│   ├── DLinear_Cleaning           → train=421 570 / test=115 064 / null=0
│   ├── DLinear_Feature_Engineering → n_series=3 331 / avg_len=136.3w
│   ├── DLinear_Baseline           → WMAE 1 882.72  ★ best
│   ├── DLinear_Tuned              → WMAE 1 883.96
│   └── DLinear_Best_Pipeline      → wmae_val_best=1 882.7204
│                                     artifact: dlinear-walmart-sales:v0
│
├── NBEATS_Training
│   ├── NBEATS_Cleaning            → train=421 570 / test=115 064 / null=0
│   ├── NBEATS_Feature_Engineering → n_series=2 300 / avg_len=141.4w
│   ├── NBEATS_Baseline            → WMAE 1 814.28  ★ best
│   ├── NBEATS_Tuned               → WMAE 1 826.07
│   └── NBEATS_Best_Pipeline       → wmae_val_best=1 814.28
│                                     artifact: nbeats-walmart-sales
│
└── TFT_Training
    ├── TFT_Cleaning               → train=421 570 / test=115 064 / null=0
    ├── TFT_Feature_Engineering    → n_series=2 300 / avg_len=141.4w / n_past=2 / n_future=1 / n_static=3
    ├── TFT_Baseline               → WMAE 1 991.71
    ├── TFT_Tuned                  → WMAE 1 841.39  ★ best
    └── TFT_Best_Pipeline          → wmae_val_best=1 841.39
                                      artifact: tft-walmart-sales
```

თითო run ლოგავს:
- `config` — hyperparameter-ების სრული ნაკრები
- `wmae_val` / `wmae_val_best` — შეფასების მეტრიკა
- `train_loss`, `val_loss`, `best_val_loss`, `epoch` — სატრენინგო დინამიკა
- model artifact — `.pt` / `.pkl` ფაილები

წინასწარ ერთხელ: `wandb login` (ან `WANDB_API_KEY`). გუნდში გასაზიარებლად
`WANDB_ENTITY` დააყენეთ `ashos22-free-university-of-tbilisi-`-ზე.

---

## 10. Inference & Model Registry (wandb)

[`model_inference.ipynb`](model_inference.ipynb):

1. wandb API-ით სკანირებს ყველა არქიტექტურის `final` run-ს და ირჩევს ყველაზე დაბალ `holdout_wmae`-ს.
2. **არეგისტრირებს** საუკეთესო model artifact-ს `best` alias-ით.
3. **ტვირთავს wandb-დან** (`walmart_<arch>:best`) და აკეთებს predict-ს ნედლ test-ზე.
4. წერს `submissions/submission.csv`-ს Kaggle-ის ფორმატში და ლოგავს submission artifact-ს.

---

## 11. მთავარი დასკვნები

- **LightGBM ყველაზე კარგია (WMAE 1 254)** — lag/rolling feature-ების წყალობით.
  ეს feature-ები პირდაპირ historical sales pattern-ს "ათვლის", რაც DL მოდელებს
  (DLinear, N-BEATS, TFT) მარტივი seq→pred window-ით არ შეუძლიათ.
- **N-BEATS (1 814) > TFT (1 841) > DLinear (1 883)** DL მოდელებს შორის.
  N-BEATS-ის სიმარტივე (pure FC + residuals) ამ ამოცანაზე TFT-ის კომპლექსურ
  multi-stream დიზაინს სჯობია.
- **TFT tuning-ი ეხმარება** — d_model 64→128-ზე გაზრდა 7.5%-ით (1 991 → 1 841)
  აუმჯობესებს შედეგს. მოდელს სამი input stream-ის ასათვისებლად მეტი სიმძლავრე
  სჭირდება.
- **DLinear Baseline > Tuned** — kernel_size 13→25 არ ეხმარება. 13-კვირიანი
  moving average ყოველკვირეური retail მონაცემებისთვის საკმარისია.
- **Huber loss LightGBM-ში ჩავარდა** — WMAE 15 116-მდე გაიზარდა, რადგან Huber
  სხვა სკალაზე ოპტიმიზებს, ვიდრე WMAE მოითხოვს.

---

## 12. სტატუსი და სამომავლო ნაბიჯები

- [x] LightGBM (Baseline + Tuned_v1 + Tuned_v2 + CV + Inference)
- [x] DLinear (Baseline + Tuned)
- [x] N-BEATS (Baseline + Tuned)
- [x] TFT (Baseline + Tuned)
- [x] XGBoost (10 ვარიანტი), Prophet (5), ARIMA/SARIMA + seasonal-naive
- [x] PatchTST (Deep Learning, Transformer)
- [x] TimesFM foundation model (zero-shot, ბონუსი)
- [x] Model registry + inference → Kaggle submission
- [ ] Lag/rolling feature-ების დამატება DL მოდელებისთვის (LightGBM-ის ძლიერება)
- [ ] ჰიპერპარამეტრების ავტო-tuning (Optuna) და ensembling (LightGBM + N-BEATS)

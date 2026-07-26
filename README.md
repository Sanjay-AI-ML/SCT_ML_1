# House Price Prediction — Linear Regression

**SkillCraft ML Internship — Task 01**

Predicts a home's sale price from its square footage, number of bedrooms, and
number of bathrooms, using Linear Regression trained on the real Kaggle
**House Prices: Advanced Regression Techniques** dataset (Ames Housing, 1,460
homes, Ames, Iowa).

---

## Results at a glance

| Metric | Core model (3 features) | Best benchmark (Gradient Boosting) |
|---|---|---|
| R² (test set) | **0.601** | 0.871 |
| RMSE (test set) | **$46,935** | $26,688 |
| 5-fold CV R² | **0.651 ± 0.042** | — |

The task asked for a 3-feature linear model — that's the primary deliverable.
I also benchmarked it against four stronger models on an extended feature set
to show what's achievable and *why* (see Section 7 of the notebook).

---

## Project structure

```
house_price_project/
├── README.md                          <- you are here
├── data/
│   └── train.csv                      <- Kaggle Ames Housing dataset (1,460 rows)
├── notebooks/
│   └── House_Price_Prediction.ipynb   <- main deliverable: full narrated analysis
├── src/
│   ├── data_preprocessing.py          <- loading, feature engineering, cleaning
│   ├── eda.py                         <- exploratory data analysis + plots
│   └── train_model.py                 <- model training, evaluation, plots
├── outputs/
│   ├── figures/                       <- 11 charts (EDA + diagnostics + comparisons)
│   ├── models/                        <- saved trained models (.joblib)
│   ├── metrics.json                   <- all metrics, machine-readable
│   └── price_predictor.html           <- live interactive demo (open in any browser)
└── build_notebook.py                  <- script that generates the notebook
```

## How to run it

### Option A — just open the notebook
Open `notebooks/House_Price_Prediction.ipynb` in Jupyter. It's already been
executed end-to-end, so all outputs and charts are baked in — you can read it
top to bottom with nothing to install. To re-run it yourself:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib jupyter
jupyter notebook notebooks/House_Price_Prediction.ipynb
```

### Option B — run the pipeline as scripts
```bash
cd src
python data_preprocessing.py   # sanity-check the cleaned dataset
python eda.py                  # regenerate all EDA figures
python train_model.py          # train all models, save metrics + figures
```

### Option C — try the live demo
Just double-click `outputs/price_predictor.html` — no server or install
needed. It runs the model's real learned coefficients directly in the
browser, with a live breakdown of how each input contributes to the price.

---

## What makes this more than the minimum ask

The brief calls for a single linear regression model. I built that model
exactly as specified (see the "core model" throughout), but went further in
a few places, on the theory that a submission should show understanding, not
just a working script:

1. **Feature engineering with reasoning, not just column selection.**
   The raw dataset has no single "bathrooms" column — real-estate listings
   define total bathrooms as `full baths + 0.5 × half baths`, including the
   basement. I engineered that explicitly instead of picking one raw column.

2. **Documented outlier removal.** Two homes with >4,000 sq ft sold at
   unusually low prices (likely distressed sales) — a known quirk of this
   exact dataset. Removing them is explained and visualized, not silently done.

3. **An explained, counter-intuitive result.** The `Bedrooms` coefficient
   comes out **negative** in the fitted model. Section 6 of the notebook
   diagnoses this as multicollinearity (bedrooms correlate with square
   footage, so — holding square footage constant — an extra bedroom implies
   smaller rooms) and backs it up with two supporting plots. This is the
   kind of insight that's easy to miss if you just report the R² and move on.

4. **Full diagnostics, not just accuracy.** Residual plots, actual-vs-predicted
   plots, and 5-fold cross-validation, so the R²/RMSE numbers are trustworthy
   rather than a single lucky train/test split.

5. **A benchmark comparison for context.** Ridge, Lasso, Random Forest, and
   Gradient Boosting were trained on an extended feature set (total finished
   area, overall quality, garage capacity, house age) to show how much the 3
   required features leave on the table (R² 0.60 → 0.87) and which features
   matter most (see the Random Forest feature-importance chart).

6. **A working interactive demo** (`price_predictor.html`) that runs the
   model's actual coefficients client-side — useful for a live walkthrough
   or viva rather than just showing static numbers.

---

## Notes on the model itself

**Learned equation** (core model):

```
SalePrice ≈ 26,342
          + 120.97  × SquareFootage
          − 30,038.83 × Bedrooms
          + 26,254.97 × Bathrooms
```

**How to read the coefficients:**
- Each additional square foot adds about **$121** to predicted price, holding
  bedrooms and bathrooms constant.
- Each additional bathroom adds about **$26,255**.
- Each additional bedroom, *holding square footage fixed*, is associated
  with about **$30,039 lower** predicted price — not because bedrooms are
  bad, but because packing more bedrooms into the same square footage
  usually means smaller, less desirable rooms. This is explained in full,
  with supporting plots, in Section 6 of the notebook.

## Dataset source

Kaggle competition: [House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
(Ames Housing Dataset, compiled by Dean De Cock). `data/train.csv` contains
the original 1,460-row, 81-column training set.

# Diamond Price Prediction with Calibrated Prediction Intervals

![CI](https://github.com/Roudeyna-Bouznif/diamond-price-prediction/actions/workflows/ci.yml/badge.svg)

End-to-end regression project: predict the price of a diamond **and tell how uncertain the
prediction is**. A point estimate alone is rarely enough for a pricing decision, so the model
returns a 90 % prediction interval whose coverage is verified on held-out data.

## Highlights

- **Regression on log-price**: prices are heavy-tailed and errors are multiplicative, so the
  target is `log(price)`; metrics are reported back in dollars.
- **Model comparison with cross-validation**: Ridge (log-log baseline), Random Forest and
  Histogram Gradient Boosting, selected on CV RMSE.
- **Prediction intervals**: quantile gradient boosting (5 % / 95 %), then
  **conformalized quantile regression** (Romano et al., 2019) to correct under-coverage.
  Coverage is checked overall and per carat bucket.
- **Explainability**: permutation importance and partial dependence plots.
- **Serving**: FastAPI REST API (single + batch endpoints, strict input validation), Docker image.
- **Engineering**: unit and API tests with pytest, linting with ruff, GitHub Actions CI.

## Results

<!-- RESULTS:START -->
> Run `python -m diamonds.train` and paste the content of `reports/results.md` here
> (model comparison table + interval coverage table). Figures are saved in `reports/figures/`.
<!-- RESULTS:END -->

![Predictions](reports/figures/predictions.png)
![Intervals](reports/figures/intervals.png)
![Importance](reports/figures/importance.png)

## Dataset

The classic *diamonds* dataset (~54k rows: carat, cut, color, clarity, depth, table, x/y/z, price),
downloaded automatically on the first run from the
[seaborn-data repository](https://github.com/mwaskom/seaborn-data). Cleaning removes duplicates
and physically impossible rows (zero dimensions, extreme outliers).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export PYTHONPATH=src                                   # Windows (cmd): set PYTHONPATH=src

python -m diamonds.train          # downloads data, trains, writes models/ and reports/
pytest                            # tests (use synthetic data, no download needed)
uvicorn diamonds.api:app --reload # API on http://127.0.0.1:8000  (Swagger UI: /docs)
```

Offline smoke test: `python -m diamonds.train --synthetic --no-cv`.

### API example

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
  "depth": 61.8, "table": 57.0, "x": 6.4, "y": 6.4, "z": 3.96 }'
```
```json
{"price_usd": "...", "lower_usd": "...", "upper_usd": "...", "confidence_level": 0.9}
```

Other endpoints: `POST /predict/batch` (up to 1000 diamonds), `GET /model-info`, `GET /health`.

### Docker

```bash
docker build -t diamond-api .      # trains the model inside the image
docker run -p 8000:8000 diamond-api
```

## Method details

| Step | Choice | Why |
|---|---|---|
| Target | `log(price)` | Multiplicative errors, heavy right tail |
| Features | Ordinal codes for cut/color/clarity, `volume = x·y·z` | Quality grades are ordered |
| Split | 80 % train / 20 % test; train further split 75/25 into fit / calibration | Calibration data is never used for fitting or testing |
| Selection | 3-fold CV on log-RMSE (≈ relative error) | Compare on a scale-free metric |
| Intervals | Quantile GBM + conformal correction `q̂` | Finite-sample coverage guarantee, adaptive width |

## Limitations

- Prices come from a single historical snapshot: no time dimension, no market drift.
- The conformal guarantee is *marginal* (average over all diamonds); coverage can differ by
  segment, which is why it is reported per carat bucket.
- Ordinal encoding assumes equal spacing between grades; the tree models do not require it,
  the Ridge baseline does.
- The pickled model is tied to the scikit-learn version used for training (the Docker image
  retrains at build time to avoid mismatches).

## Possible extensions

- Monotonic constraints on carat; SHAP values for per-prediction explanations.
- Hyperparameter search (Optuna) and drift monitoring.
- Streamlit front-end on top of the API.

## Project structure

```
src/diamonds/   features.py · data.py · model.py · train.py · api.py
tests/          features, model (coverage, ordering, monotonicity), API
.github/        CI workflow (ruff + pytest on Python 3.11 / 3.12)
Dockerfile      reproducible serving image
```

## License

MIT

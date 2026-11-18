"""Training pipeline: model comparison (CV), quantile models, conformal calibration,
test-set evaluation, explainability figures.

Usage:
    python -m diamonds.train                  # real data (downloaded on first run)
    python -m diamonds.train --synthetic      # offline smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import DEFAULT_PATH, load_diamonds, make_synthetic
from .features import clean, make_features
from .model import Bundle

ALPHA = 0.10  # 90 % prediction intervals
BEST_DEFAULT = "HistGradientBoosting"


def candidates(seed: int) -> dict:
    return {
        "Ridge (log-log baseline)": make_pipeline(
            FunctionTransformer(np.log1p), StandardScaler(), Ridge(alpha=1.0)
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
        BEST_DEFAULT: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, early_stopping=True, random_state=seed
        ),
    }


def price_metrics(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    y, p = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(p - y) / y
    return {
        "mae_usd": float(mean_absolute_error(y, p)),
        "rmse_usd": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "mape_pct": float(ape.mean() * 100),
        "median_ape_pct": float(np.median(ape) * 100),
    }


def conformal_qhat(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Conformalized quantile regression (Romano et al., 2019)."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def interval_report(lo, hi, y_log, carat) -> dict:
    inside = (y_log >= lo) & (y_log <= hi)
    width = np.exp(hi) - np.exp(lo)
    bins = pd.cut(carat, [0, 0.5, 1.0, 1.5, 2.0, np.inf],
                  labels=["<=0.5", "0.5-1", "1-1.5", "1.5-2", ">2"])
    by_bin = pd.Series(inside).groupby(bins, observed=True).mean().round(4).to_dict()
    return {
        "coverage": float(inside.mean()),
        "mean_width_usd": float(width.mean()),
        "median_width_usd": float(np.median(width)),
        "coverage_by_carat": {str(k): float(v) for k, v in by_bin.items()},
    }


def run_training(df: pd.DataFrame, out_dir: Path, cv: bool = True,
                 figures: bool = True, seed: int = 42) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    df = clean(df)
    X, y = make_features(df), np.log(df["price"].to_numpy())
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, random_state=seed
    )

    # 1) Model comparison: CV on train (log-RMSE ~ relative error) + hold-out test metrics
    comparison, models = [], candidates(seed)
    if cv:
        kf = KFold(3, shuffle=True, random_state=seed)
        for name, est in models.items():
            scores = -cross_val_score(est, X_tr_full, y_tr_full, cv=kf,
                                      scoring="neg_root_mean_squared_error")
            est.fit(X_tr_full, y_tr_full)
            comparison.append({
                "model": name,
                "cv_rmse_log_mean": float(scores.mean()),
                "cv_rmse_log_std": float(scores.std()),
                **price_metrics(y_te, est.predict(X_te)),
            })
        best = min(comparison, key=lambda r: r["cv_rmse_log_mean"])["model"]
    else:
        best = BEST_DEFAULT
        models[best].fit(X_tr_full, y_tr_full)
    point = models[best]

    # 2) Quantile models + conformal calibration on a held-out calibration set
    def quantile_model(q):
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.08, random_state=seed
        ).fit(X_tr, y_tr)

    lower, upper = quantile_model(ALPHA / 2), quantile_model(1 - ALPHA / 2)
    qhat = conformal_qhat(lower.predict(X_cal), upper.predict(X_cal), y_cal, ALPHA)

    bundle = Bundle(point, lower, upper, qhat, ALPHA)

    # 3) Test-set evaluation
    lo_raw, _, hi_raw = bundle.predict_log(X_te, conformal=False)
    lo_c, mid, hi_c = bundle.predict_log(X_te, conformal=True)
    metrics = {
        "selected_model": best,
        "n_rows_after_cleaning": int(len(df)),
        "n_train": int(len(X_tr_full)),
        "n_test": int(len(X_te)),
        "point_metrics_test": price_metrics(y_te, mid),
        "nominal_coverage": 1 - ALPHA,
        "qhat_log": qhat,
        "interval_raw_quantile": interval_report(lo_raw, hi_raw, y_te, X_te["carat"].to_numpy()),
        "interval_conformal": interval_report(lo_c, hi_c, y_te, X_te["carat"].to_numpy()),
        "model_comparison": comparison,
    }

    # 4) Explainability: permutation importance on the test set
    pi = permutation_importance(point, X_te, y_te, n_repeats=5, random_state=seed,
                                scoring="neg_root_mean_squared_error")
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
    metrics["permutation_importance"] = {k: float(v) for k, v in imp.items()}

    bundle.metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "selected_model": best,
        "alpha": ALPHA,
    }
    joblib.dump(bundle, out_dir / "models" / "model.joblib", compress=3)
    (out_dir / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "reports" / "results.md").write_text(results_markdown(metrics))
    if figures:
        make_figures(out_dir / "reports" / "figures", bundle, X_te, y_te, imp)
    return metrics


def results_markdown(m: dict) -> str:
    p, r, c = m["point_metrics_test"], m["interval_raw_quantile"], m["interval_conformal"]
    lines = [f"**Selected model:** {m['selected_model']}  ",
             f"**Test set:** {m['n_test']:,} diamonds\n"]
    if m["model_comparison"]:
        lines += ["| Model | CV RMSE (log) | Test MAE ($) | Test R² | Test MAPE (%) |",
                  "|---|---|---|---|---|"]
        for r_ in m["model_comparison"]:
            lines.append(f"| {r_['model']} | {r_['cv_rmse_log_mean']:.4f} ± "
                         f"{r_['cv_rmse_log_std']:.4f} | {r_['mae_usd']:,.0f} | "
                         f"{r_['r2']:.4f} | {r_['mape_pct']:.2f} |")
        lines.append("")
    lines += [f"Final model: MAE **${p['mae_usd']:,.0f}**, R² **{p['r2']:.4f}**, "
              f"median error **{p['median_ape_pct']:.1f}%**.\n",
              "| 90% interval | Coverage | Mean width ($) |", "|---|---|---|",
              f"| Raw quantile regression | {r['coverage']:.1%} | {r['mean_width_usd']:,.0f} |",
              f"| + Conformal calibration | {c['coverage']:.1%} | {c['mean_width_usd']:,.0f} |"]
    return "\n".join(lines) + "\n"


def make_figures(fig_dir: Path, bundle: Bundle, X_te, y_te, imp: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay

    lo, mid, hi = bundle.predict_log(X_te)
    y, p = np.exp(y_te), np.exp(mid)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y, p, s=3, alpha=0.3)
    lim = [y.min(), y.max()]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set(xscale="log", yscale="log", xlabel="Actual price ($)",
              ylabel="Predicted price ($)", title="Predicted vs actual")
    ax[1].scatter(p, (p - y) / y * 100, s=3, alpha=0.3)
    ax[1].axhline(0, color="r", lw=1)
    ax[1].set(xscale="log", xlabel="Predicted price ($)", ylabel="Relative error (%)",
              title="Relative residuals", ylim=(-60, 60))
    fig.tight_layout(); fig.savefig(fig_dir / "predictions.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    imp.sort_values().plot.barh(ax=ax)
    ax.set(xlabel="Increase in log-RMSE when shuffled", title="Permutation importance (test set)")
    fig.tight_layout(); fig.savefig(fig_dir / "importance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    PartialDependenceDisplay.from_estimator(
        bundle.point, X_te.sample(min(2000, len(X_te)), random_state=0),
        ["carat", "clarity", "color"], ax=ax)
    fig.suptitle("Partial dependence (log price)")
    fig.tight_layout(); fig.savefig(fig_dir / "partial_dependence.png", dpi=130); plt.close(fig)

    order = np.argsort(p)[:: max(1, len(p) // 150)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(order))
    ax.fill_between(xs, np.exp(lo)[order], np.exp(hi)[order], alpha=0.3, label="90% interval")
    ax.plot(xs, y[order], "k.", ms=4, label="Actual price")
    ax.plot(xs, p[order], "r-", lw=1, label="Prediction")
    ax.set(yscale="log", xlabel="Test diamonds (sorted by predicted price)",
           ylabel="Price ($)", title="Conformalized prediction intervals")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "intervals.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--out", default=".", help="Base dir for models/ and reports/")
    ap.add_argument("--synthetic", action="store_true", help="Offline synthetic data")
    ap.add_argument("--no-cv", action="store_true", help="Skip model comparison (fast)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    df = make_synthetic(20000) if a.synthetic else load_diamonds(a.data)
    m = run_training(df, Path(a.out), cv=not a.no_cv, figures=not a.no_figures)
    print((Path(a.out) / "reports" / "results.md").read_text())


if __name__ == "__main__":
    main()

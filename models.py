"""models.py — Machine Learning (Regression) และ Time Series Forecasting"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

FEATURES = {
    "IsWeekend": "สุดสัปดาห์", "Promotion": "โปรโมชัน", "AvgPrice": "ราคาเฉลี่ย",
    "MonthSin": "ฤดูกาล (sin)", "MonthCos": "ฤดูกาล (cos)",
    "Lag1": "ยอดขายเมื่อวาน", "Lag7": "ยอดขายสัปดาห์ก่อน",
}
X_COLS = list(FEATURES)


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    m = dates.dt.month
    return pd.DataFrame({"IsWeekend": (dates.dt.dayofweek >= 5).astype(int),
                         "MonthSin": np.sin(2 * np.pi * m / 12), "MonthCos": np.cos(2 * np.pi * m / 12)})


def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("Date").reset_index(drop=True)
    df = pd.concat([df, calendar_features(df["Date"])], axis=1)
    df["Lag1"], df["Lag7"] = df["Sales"].shift(1), df["Sales"].shift(7)
    return df.dropna().reset_index(drop=True)


def train_regression(daily: pd.DataFrame):
    """เทียบ Ridge กับ Random Forest — แบ่ง train/test ตามเวลา 80/20 (ไม่สุ่ม เพื่อเลี่ยง data leakage)"""
    df = make_features(daily)
    if len(df) < 40:
        return None
    cut = int(len(df) * 0.8)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    candidates = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    rows, test = [], pd.DataFrame({"Date": te["Date"].values, "Actual": te["Sales"].values})
    for name, model in candidates.items():
        model.fit(tr[X_COLS], tr["Sales"])
        pred = model.predict(te[X_COLS])
        test[name] = pred
        rows.append({"Model": name, "MAE": mean_absolute_error(te["Sales"], pred),
                     "RMSE": float(np.sqrt(mean_squared_error(te["Sales"], pred))), "R2": r2_score(te["Sales"], pred)})
    metrics = pd.DataFrame(rows)
    best = metrics.sort_values("MAE").iloc[0]["Model"]

    # ความสำคัญของปัจจัย: Permutation Importance บนชุดทดสอบ (ใช้ได้กับทุกโมเดล)
    pi = permutation_importance(candidates[best], te[X_COLS], te["Sales"], n_repeats=8,
                                random_state=0, scoring="neg_mean_absolute_error")
    imp = pd.Series(np.clip(pi.importances_mean, 0, None), index=X_COLS)
    imp = imp / imp.sum() if imp.sum() > 0 else imp

    final = clone(candidates[best]).fit(df[X_COLS], df["Sales"])  # เทรนใหม่ด้วยข้อมูลทั้งหมดเพื่อพยากรณ์จริง
    return {"best": best, "model": final, "metrics": metrics, "test": test, "importance": imp,
            "corr": df[X_COLS + ["Sales"]].corr()["Sales"].drop("Sales"),
            "naive_mae": mean_absolute_error(te["Sales"], te["Lag7"])}


def regression_forecast(model, daily: pd.DataFrame, horizon: int, promo_start: int = 0, promo_days: int = 0) -> pd.DataFrame:
    """พยากรณ์แบบ recursive: ใช้ค่าที่พยากรณ์เป็น Lag ของวันถัดไป (promo_start = เริ่มโปรโมชันอีกกี่วันนับจากพรุ่งนี้, promo_days = จำนวนวันโปรโมชัน)"""
    hist = list(daily.sort_values("Date")["Sales"])
    avg_price = daily["AvgPrice"].mean()
    dates = pd.date_range(daily["Date"].max() + pd.Timedelta(days=1), periods=horizon)
    cal = calendar_features(pd.Series(dates))
    out = []
    for i in range(horizon):
        row = cal.iloc[[i]].assign(Promotion=int(promo_start <= i < promo_start + promo_days), AvgPrice=avg_price, Lag1=hist[-1], Lag7=hist[-7])
        y = max(0.0, float(model.predict(row[X_COLS])[0]))
        hist.append(y)
        out.append(y)
    return pd.DataFrame({"Date": dates, "Forecast": out})


def _hw(series):
    try:
        return ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7).fit()
    except Exception:
        return ExponentialSmoothing(series, trend="add").fit()


def timeseries_forecast(daily: pd.DataFrame, horizon: int):
    """Holt-Winters (trend + seasonality รายสัปดาห์) พร้อม Backtest 28 วันเทียบ Naive"""
    s = daily.set_index("Date")["Sales"].asfreq("D").ffill()
    backtest = None
    if len(s) > 60:
        train, test = s.iloc[:-28], s.iloc[-28:]
        pred = _hw(train).forecast(28)
        backtest = {"mae": mean_absolute_error(test, pred), "naive_mae": mean_absolute_error(test, [train.iloc[-1]] * 28)}
    model = _hw(s)
    fc, sd = model.forecast(horizon), float(np.std(model.resid))
    out = pd.DataFrame({"Date": fc.index, "Forecast": fc.clip(lower=0).values,
                        "Lower": (fc - 1.96 * sd).clip(lower=0).values, "Upper": (fc + 1.96 * sd).values})
    return out, backtest

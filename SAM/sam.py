import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def run_sam(tickers, first_tau=20151231, last_tau=20201231):
    cal        = load_calendar()
    taus       = get_retraining_dates(cal, first_tau, last_tau)
    etf_cache  = load_etf_cache()
    month_ends = sorted(etf_cache.keys())

    all_predictions = {ticker: {} for ticker in tickers}
    all_skipped     = {}  # 修复5: 记录每个tau跳过的ticker

    for tau_idx, tau in enumerate(taus):
        print(f"\n[{tau_idx+1}/{len(taus)}] tau = {tau}")

        bounds      = get_window_bounds(cal, tau)
        train_start = bounds["train_start"]
        train_end   = bounds["train_end"]
        val_start   = bounds["val_start"]
        val_end     = bounds["val_end"]
        pred_start  = bounds["pred_start"]
        pred_end    = bounds["pred_end"]

        data_start = cal[cal.index(train_start) - 5]
        panel = load_date_range(data_start, pred_end,
                                tickers=tickers + ETF_LIST)

        etf_panel = extract_etf_returns(panel)
        etf_panel = add_market_adjusted_returns(etf_panel)

        skipped_tickers = []

        for ticker in tickers:
            print(f"  Processing {ticker}...")

            train_dates = [d for d in cal if train_start <= d <= train_end]
            val_dates   = [d for d in cal if val_start   <= d <= val_end]

            # Construct training X and y
            X_train, y_train = [], []
            for t in train_dates:
                t_idx  = cal.index(t)
                t_next = cal[t_idx + 1]

                next_day = panel[(panel['ticker'] == ticker) &
                                 (panel['date']   == t_next)]
                if next_day.empty:
                    continue
                y = next_day['pvCLCL'].values[0]

                etf_stl = get_etf_features(etf_cache, month_ends, t, calendar=cal)
                etf_vol  = build_etf_volume_vol_features(
                               etf_panel[etf_panel['date'] == t])
                stock_ft = build_stock_features(panel, ticker, t)

                if not etf_stl or etf_vol is None or stock_ft is None:
                    continue

                x = build_feature_vector(etf_stl, etf_vol, stock_ft)
                if x is None:
                    continue

                X_train.append(x)
                y_train.append(y)

            # Construct validation X and y
            X_val, y_val = [], []
            for t in val_dates:
                t_idx  = cal.index(t)
                t_next = cal[t_idx + 1]
                next_day = panel[(panel['ticker'] == ticker) &
                                 (panel['date']   == t_next)]

                if next_day.empty:
                    continue
                y = next_day['pvCLCL'].values[0]

                etf_stl = get_etf_features(etf_cache, month_ends, t, calendar=cal)
                etf_vol  = build_etf_volume_vol_features(
                               etf_panel[etf_panel['date'] == t])
                stock_ft = build_stock_features(panel, ticker, t)

                if not etf_stl or etf_vol is None or stock_ft is None:
                    continue

                x = build_feature_vector(etf_stl, etf_vol, stock_ft)
                if x is None:
                    continue

                X_val.append(x)
                y_val.append(y)

            if len(X_train) == 0 or len(X_val) == 0:
                skipped_tickers.append(ticker)
                continue

            # 修复1: 固定 feature 顺序，用第一个样本的 keys 作为基准
            feature_names = list(X_train[0].keys())
            X_train_arr = np.array([[x.get(f, np.nan) for f in feature_names]
                                    for x in X_train])
            y_train_arr = np.array(y_train)
            X_val_arr   = np.array([[x.get(f, np.nan) for f in feature_names]
                                    for x in X_val])
            y_val_arr   = np.array(y_val)

            # 修复2: 先删 y 的 NaN，再检查样本数
            train_mask  = ~np.isnan(y_train_arr)
            val_mask    = ~np.isnan(y_val_arr)
            X_train_arr = X_train_arr[train_mask]
            y_train_arr = y_train_arr[train_mask]
            X_val_arr   = X_val_arr[val_mask]
            y_val_arr   = y_val_arr[val_mask]

            # 删完 NaN 之后再检查样本数是否足够
            if len(X_train_arr) < 100 or len(X_val_arr) < 50:
                skipped_tickers.append(ticker)
                continue

            # 检测 X 中 NaN 数量
            nan_count = np.isnan(X_train_arr).sum()
            print(f"  {ticker}: {nan_count} NaN values in X_train "
                  f"({nan_count / X_train_arr.size * 100:.2f}%)")

            # 用 training 数据 fit scaler（deliberately 避免 leakage）
            scaler         = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_arr)
            X_val_scaled   = scaler.transform(X_val_arr)

            # fill NaN with 0 after standardizing
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
            X_val_scaled   = np.nan_to_num(X_val_scaled,   nan=0.0)

            # lambda 候选值
            lambdas = [0.01, 0.1, 1, 10, 100, 1000]

            best_lambda  = None
            best_val_mse = np.inf

            for lam in lambdas:
                model = Ridge(alpha=lam)
                model.fit(X_train_scaled, y_train_arr)
                y_pred_val = model.predict(X_val_scaled)
                mse = np.mean((y_pred_val - y_val_arr) ** 2)
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_lambda  = lam

            print(f"  {ticker}: best_lambda = {best_lambda}, "
                  f"val_mse = {best_val_mse:.6f}")

            # 用 train+val 重新训练最终模型，scaler 保持用 training 期的
            X_refit = np.vstack([X_train_scaled, X_val_scaled])
            y_refit = np.concatenate([y_train_arr, y_val_arr])
            final_model = Ridge(alpha=best_lambda)
            final_model.fit(X_refit, y_refit)

            # 预测窗口
            pred_dates = [d for d in cal if pred_start <= d <= pred_end]

            for t in pred_dates:
                etf_stl = get_etf_features(etf_cache, month_ends, t, calendar=cal)
                etf_vol  = build_etf_volume_vol_features(
                               etf_panel[etf_panel['date'] == t])
                stock_ft = build_stock_features(panel, ticker, t)

                if not etf_stl or etf_vol is None or stock_ft is None:
                    continue

                x = build_feature_vector(etf_stl, etf_vol, stock_ft)
                if x is None:
                    continue

                x_arr    = np.array([x.get(f, np.nan) for f in feature_names]).reshape(1, -1)
                x_scaled = scaler.transform(x_arr)
                x_scaled = np.nan_to_num(x_scaled, nan=0.0)

                y_pred = final_model.predict(x_scaled)[0]
                all_predictions[ticker][t] = y_pred

        # 修复5: 记录并打印每个 tau 跳过的 ticker
        all_skipped[tau] = skipped_tickers
        if skipped_tickers:
            print(f"  Skipped {len(skipped_tickers)} tickers: {skipped_tickers}")

    # 保存预测结果
    results_path = Path(__file__).parent.parent / "results"
    results_path.mkdir(exist_ok=True)

    rows = []
    for ticker, preds in all_predictions.items():
        for date, pred in preds.items():
            rows.append({"ticker": ticker, "date": date, "pred": pred})

    df = pd.DataFrame(rows)
    df.to_csv(results_path / "sam_predictions.csv", index=False)
    print(f"\nSaved predictions to {results_path / 'sam_predictions.csv'}")

    # 修复5: 保存 skipped tickers 记录
    skipped_rows = []
    for tau, tickers_skipped in all_skipped.items():
        for t in tickers_skipped:
            skipped_rows.append({"tau": tau, "ticker": t})
    pd.DataFrame(skipped_rows).to_csv(results_path / "sam_skipped.csv", index=False)
    print(f"Saved skipped tickers to {results_path / 'sam_skipped.csv'}")

    return all_predictions, all_skipped


if __name__ == "__main__":
    vol = pd.read_csv(
        Path.home() / "199project" / "data" /
        "Matrix_Format_SubsetUniverse" / "volume_20000103_20201231.csv"
    )
    tickers = vol["ticker"].tolist()

    # 先只跑5只股票、1个tau测试
    run_sam(tickers, first_tau=20191231, last_tau=20241231)
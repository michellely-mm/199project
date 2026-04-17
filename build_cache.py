from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from utils import (
    load_calendar,
    load_date_range,
    extract_etf_returns,
    add_market_adjusted_returns,
    build_etf_feature_cache,
    save_etf_cache,
    ETF_LIST,
    FEATURES_PATH,
)

if __name__ == "__main__":

    cache_path = FEATURES_PATH / "etf_stl_cache.pkl"

    # 如果cache已经存在就跳过
    if cache_path.exists():
        print(f"Cache already exists at {cache_path}, skipping.")
    else:
        print("=" * 50)
        print("Step 1/5: Loading calendar...")
        cal = load_calendar()
        print(f"  Done. {len(cal)} trading days loaded.")

        print("=" * 50)
        print("Step 2/5: Loading ETF data (2000-2024)...")
        raw = load_date_range(20000103, 20241231, tickers=ETF_LIST)
        print(f"  Done. {len(raw)} rows loaded.")

        print("=" * 50)
        print("Step 3/5: Preparing ETF panel...")
        etf_panel = extract_etf_returns(raw)
        etf_panel = add_market_adjusted_returns(etf_panel)
        print(f"  Done. ETF panel shape: {etf_panel.shape}")

        print("=" * 50)
        print("Step 4/5: Building ETF STL cache...")
        print("  WARNING: This will take a long time.")
        cache = build_etf_feature_cache(etf_panel, cal, 20000103, 20241231)
        print(f"  Done. {len(cache)} month-end entries in cache.")

        print("=" * 50)
        print("Step 5/5: Saving cache...")
        save_etf_cache(cache)

        print("=" * 50)
        print("All done! You can now run SAM/UAM/CAM.")

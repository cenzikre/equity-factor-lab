from __future__ import annotations

from datetime import date
from pathlib import Path

from util.dashboard.market_regime_helpers import (
    validate_environment,
    fetch_fmp_price_data,
    build_market_features,
    save_pdf_report,
    save_snapshot_csvs,
)

from dotenv import load_dotenv


load_dotenv()
REPORT_NAME = "market_regime"


def main() -> None:
    validate_environment()

    tickers = ["SPY", "QQQ", "IWM", "RSP"]
    start_date = "2024-01-01"
    end_date = date.today().strftime("%Y-%m-%d")

    output_dir = Path("reports") / REPORT_NAME / end_date
    output_dir.mkdir(parents=True, exist_ok=True)

    price_df = fetch_fmp_price_data(tickers=tickers, start_date=start_date, end_date=end_date)
    price_path = output_dir / "source_prices.csv"
    price_df.to_csv(price_path, index=False)

    feature_groups = build_market_features(
        price_df,
        price_col="adjClose",
        return_lookbacks=(5, 10, 20, 60, 126),
        dma_windows=(5, 10, 20, 60, 126),
        vdma_windows=(5, 10, 20, 60, 126),
        rv_windows=(5, 10, 20, 60, 126),
        mdd_windows=(20, 60, 126, 252),
        reg_ma_windows=(5, 10, 20, 60, 126),
        reg_window=10,
        z_window=60,
    )

    save_snapshot_csvs(feature_groups, output_dir=output_dir)

    pdf_path = output_dir / f"{REPORT_NAME}_{end_date}.pdf"
    save_pdf_report(feature_groups, output_pdf=pdf_path, title=f"Market Regime Report — {end_date}", tickers=tickers)

    print(f"Saved source prices: {price_path}")
    print(f"Saved report PDF:    {pdf_path}")
    print(f"Saved snapshots to:  {output_dir}")


if __name__ == "__main__":
    main()

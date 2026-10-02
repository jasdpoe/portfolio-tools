import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import yfinance.cache as yf_cache


BASE_DIR = Path(__file__).resolve().parents[1]
PORTFOLIO_DIR = BASE_DIR / "tfsa" / "portfolio_outputs"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / ".yfinance_cache"

BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
RSI_WINDOW = 14


def safe_file_part(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def load_portfolio_symbols(portfolio_dir: Path) -> list[str]:
    symbols: set[str] = set()
    for csv_file in portfolio_dir.glob("*_portfolio.csv"):
        try:
            holdings = pd.read_csv(csv_file, usecols=["symbol"])
        except (ValueError, OSError, pd.errors.EmptyDataError):
            continue

        symbols.update(
            str(symbol).strip().upper()
            for symbol in holdings["symbol"].dropna()
            if str(symbol).strip()
        )
    return sorted(symbols)


def resolve_symbol(asset_name: str, portfolio_dir: Path) -> str:
    requested = asset_name.strip().upper()
    if not requested:
        raise ValueError("Asset name cannot be empty.")

    portfolio_symbols = load_portfolio_symbols(portfolio_dir)
    exact_matches = [symbol for symbol in portfolio_symbols if symbol == requested]
    base_matches = [
        symbol
        for symbol in portfolio_symbols
        if symbol.split(".", maxsplit=1)[0] == requested
    ]

    matches = exact_matches or base_matches
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        joined = ", ".join(matches)
        raise ValueError(f"Asset name '{asset_name}' matches multiple symbols: {joined}")

    return requested


def download_daily_data(symbol: str, period: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data = data.copy()
    data.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["SMA"] = data["Close"].rolling(window=BOLLINGER_WINDOW).mean()
    rolling_std = data["Close"].rolling(window=BOLLINGER_WINDOW).std()
    data["UpperBB"] = data["SMA"] + (BOLLINGER_STD * rolling_std)
    data["LowerBB"] = data["SMA"] - (BOLLINGER_STD * rolling_std)

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.rolling(window=RSI_WINDOW).mean()
    average_loss = loss.rolling(window=RSI_WINDOW).mean()
    relative_strength = average_gain / average_loss
    data["RSI"] = 100 - (100 / (1 + relative_strength))

    return data.dropna(subset=["SMA", "UpperBB", "LowerBB", "RSI"])


def plot_chart(symbol: str, data: pd.DataFrame, output_file: Path) -> None:
    fig, (price_ax, rsi_ax, volume_ax) = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    price_ax.plot(data.index, data["Close"], label="Close", color="#1f77b4", linewidth=1.4)
    price_ax.plot(
        data.index,
        data["SMA"],
        label=f"{BOLLINGER_WINDOW}-day SMA",
        color="#ff7f0e",
        linewidth=1,
    )
    price_ax.plot(data.index, data["UpperBB"], label="Upper Bollinger", color="#6c757d", linewidth=0.9)
    price_ax.plot(data.index, data["LowerBB"], label="Lower Bollinger", color="#6c757d", linewidth=0.9)
    price_ax.fill_between(
        data.index,
        data["LowerBB"].to_numpy(dtype=float),
        data["UpperBB"].to_numpy(dtype=float),
        color="#adb5bd",
        alpha=0.2,
    )
    price_ax.set_title(f"{symbol} Daily Chart")
    price_ax.set_ylabel("Price")
    price_ax.grid(True, alpha=0.25)
    price_ax.legend(loc="upper left", ncols=2)

    rsi_ax.plot(data.index, data["RSI"], color="#7b2cbf", linewidth=1)
    rsi_ax.axhline(70, color="#d62728", linestyle="--", linewidth=0.9)
    rsi_ax.axhline(30, color="#2ca02c", linestyle="--", linewidth=0.9)
    rsi_ax.set_ylim(0, 100)
    rsi_ax.set_ylabel("RSI")
    rsi_ax.grid(True, alpha=0.25)

    volume_ax.bar(data.index, data["Volume"], color="#495057", width=1.0)
    volume_ax.set_ylabel("Volume")
    volume_ax.grid(True, axis="y", alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a daily price chart with Bollinger Bands, RSI, and volume "
            "for one asset from the portfolio output files or a direct ticker."
        )
    )
    parser.add_argument("asset", help="Asset name or ticker, for example AMD, AMD.TO, VFV, or VFV.TO.")
    parser.add_argument("--period", default="6mo", help="Yahoo Finance period to download. Default: 6mo.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path. Default: output/YYYY-MM-DD_SYMBOL_daily_indicators.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf_cache.set_cache_location(str(CACHE_DIR))

    symbol = resolve_symbol(args.asset, PORTFOLIO_DIR)
    print(f"Downloading daily data for {symbol}...")

    data = download_daily_data(symbol, args.period)
    data = add_indicators(data)
    if data.empty:
        raise SystemExit(f"No usable Yahoo Finance data returned for {symbol}.")

    run_date = datetime.now().strftime("%Y-%m-%d")
    output_file = args.output or OUTPUT_DIR / f"{run_date}_{safe_file_part(symbol)}_daily_indicators.png"
    plot_chart(symbol, data, output_file)

    print(f"Saved chart to {output_file}")


if __name__ == "__main__":
    main()

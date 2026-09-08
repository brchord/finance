"""
yf_fred_market_data.py

Retrieves Yahoo Finance SPX time series and FRED treasury yields
to store it into local storage for Monte Carlo path synthesis using
VAR Filtered Block Boostrap.
"""

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


class MarketDataManager:
    """
    Handles local Parquet persistence and incremental fetching for daily S&P 500 
    prices and FRED constant maturity yields (3M Bill, 5Y Note).
    """

    def __init__(self,
                 auto_update = False,
                 cache_filepath: str = "market_levels_cache.parquet",
                 historical_floor_date: str = "1982-01-01"):
        self.auto_update = auto_update
        self.cache_filepath = Path(cache_filepath)
        self.historical_floor_date = pd.Timestamp(historical_floor_date)

    def _fetch_remote_data(self,
                           start_date: pd.Timestamp,
                           end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Queries Yahoo Finance and FRED endpoints for a specified date range.
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # 1. Fetch S&P 500 Historical Close
        spx_raw = yf.download(
            "^GSPC", start=start_str, end=end_str, progress=False)["Close"]
        if isinstance(spx_raw, pd.DataFrame):
            spx_raw = spx_raw["^GSPC"]
        spx_series = spx_raw.rename("spx_close")

        # 2. Fetch FRED Constant Maturity Yields
        fred_3m_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
        fred_5y_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS5"

        tbill_raw = pd.read_csv(
            fred_3m_url, parse_dates=["observation_date"], index_col="observation_date")
        tnote_raw = pd.read_csv(
            fred_5y_url, parse_dates=["observation_date"], index_col="observation_date")

        yield_3m_series = pd.to_numeric(tbill_raw["DGS3MO"], errors="coerce").rename("yield_3m")
        yield_3m_series /= 100.0
        yield_5y_series = pd.to_numeric(tnote_raw["DGS5"], errors="coerce").rename("yield_5y")
        yield_5y_series /= 100.0

        # 3. Combine raw fetched series
        remote_levels = pd.concat([spx_series, yield_3m_series, yield_5y_series],
                                  axis=1)
        return remote_levels.loc[start_date:end_date]

    def load_or_update_market_levels(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Loads cached market levels or fetches incremental updates if local cache is stale.
        """
        today = pd.Timestamp(datetime.date.today())

        if not self.cache_filepath.exists() or force_refresh:
            market_levels = self._fetch_remote_data(
                start_date=self.historical_floor_date, end_date=today)
        else:
            cached_levels = pd.read_parquet(self.cache_filepath)
            max_cached_date = cached_levels.index.max()

            if self.auto_update and max_cached_date < (today - pd.Timedelta(days=1)):
                incremental_start_date = max_cached_date - pd.Timedelta(days=5)
                new_levels = self._fetch_remote_data(
                    start_date=incremental_start_date, end_date=today)

                market_levels = pd.concat([cached_levels, new_levels])
                market_levels = market_levels[~market_levels.index.duplicated(keep="last")]
                market_levels.sort_index(inplace=True)
            else:
                market_levels = cached_levels

        market_levels = market_levels.ffill().dropna()
        market_levels.to_parquet(self.cache_filepath)

        return market_levels

    def get_aligned_data(
            self,
            force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns synchronized market levels alongside daily stationary increments.
        """
        market_levels = self.load_or_update_market_levels(force_refresh=force_refresh)

        market_returns = pd.DataFrame(index=market_levels.index[1:])
        market_returns["spx_log_return"] = np.log(
            market_levels["spx_close"] / market_levels["spx_close"].shift(1))
        market_returns["yield_3m_diff"] = market_levels["yield_3m"].diff()
        market_returns["yield_5y_diff"] = market_levels["yield_5y"].diff()
        market_returns = market_returns.dropna()

        return market_levels, market_returns

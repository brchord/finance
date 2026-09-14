"""
yf_fred_market_data.py

Retrieves Yahoo Finance SPX time series and FRED treasury yields/CPI
to store into local storage and compute real-space monthly return vectors 
for 3D VAR residual bootstrapping.
"""

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


class MarketDataManager:
    """
    Handles local Parquet persistence and incremental fetching for monthly 
    S&P 500 prices, FRED yields, and CPI data to compute upstream real returns.
    """

    def __init__(self,
                 auto_update: bool = False,
                 cache_filepath: str = "market_levels_cache.parquet",
                 historical_floor_date: str = "1982-01-01"):
        self.auto_update = auto_update
        self.cache_filepath = Path(cache_filepath)
        self.historical_floor_date = pd.Timestamp(historical_floor_date)

    def _fetch_remote_data(self,
                           start_date: pd.Timestamp,
                           end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Queries Yahoo Finance and FRED endpoints (Yields + CPI) for a specified date range.
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # 1. Fetch S&P 500 Historical Close
        spx_raw = yf.download(
            "^GSPC", start=start_str, end=end_str, progress=False)["Close"]
        if isinstance(spx_raw, pd.DataFrame):
            spx_raw = spx_raw["^GSPC"]
        spx_series = spx_raw.rename("spx_close")

        # 2. Fetch FRED Series: Yields (Daily) & CPI (Monthly)
        fred_3m_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
        fred_5y_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS5"
        fred_cpi_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"

        tbill_raw = pd.read_csv(
            fred_3m_url, parse_dates=["observation_date"], index_col="observation_date")
        tnote_raw = pd.read_csv(
            fred_5y_url, parse_dates=["observation_date"], index_col="observation_date")
        cpi_raw = pd.read_csv(
            fred_cpi_url, parse_dates=["observation_date"], index_col="observation_date")

        yield_3m_series = pd.to_numeric(tbill_raw["DGS3MO"], errors="coerce").rename("yield_3m") / 100.0
        yield_5y_series = pd.to_numeric(tnote_raw["DGS5"], errors="coerce").rename("yield_5y") / 100.0
        cpi_series = pd.to_numeric(cpi_raw["CPIAUCSL"], errors="coerce").rename("cpi")

        # Combine raw fetched daily levels (CPI will be monthly points filled forward initially)
        remote_levels = pd.concat([spx_series, yield_3m_series, yield_5y_series, cpi_series], axis=1)
        return remote_levels.loc[start_date:end_date]

    def load_or_update_market_levels(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Loads cached market levels or fetches incremental updates if local cache is stale.
        """
        today = pd.Timestamp(datetime.date.today())
        save_file = False

        if not self.cache_filepath.exists() or force_refresh:
            save_file = True
            market_levels = self._fetch_remote_data(
                start_date=self.historical_floor_date, end_date=today)
        else:
            cached_levels = pd.read_parquet(self.cache_filepath)
            max_cached_date = cached_levels.index.max()

            if self.auto_update and max_cached_date < (today - pd.Timedelta(days=1)):
                save_file = True
                incremental_start_date = max_cached_date - pd.Timedelta(days=10)
                new_levels = self._fetch_remote_data(
                    start_date=incremental_start_date, end_date=today)

                market_levels = pd.concat([cached_levels, new_levels])
                market_levels = market_levels[~market_levels.index.duplicated(keep="last")]
                market_levels.sort_index(inplace=True)
            else:
                market_levels = cached_levels

        market_levels = market_levels.ffill().dropna()
        if save_file:
            market_levels.to_parquet(self.cache_filepath)

        return market_levels

    def get_aligned_real_returns(self, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Resamples market levels to monthly frequency and computes 3D real monthly returns:
        [r_spx_real, r_3m_real, r_5y_real].
        """
        raw_levels = self.load_or_update_market_levels(force_refresh=force_refresh)

        # 1. Resample to Month-End frequency to align Daily Financial Data with Monthly CPI
        monthly_levels = pd.DataFrame()
        monthly_levels["spx_close"] = raw_levels["spx_close"].resample("ME").last()
        monthly_levels["cpi"] = raw_levels["cpi"].resample("ME").last().ffill()
        monthly_levels["yield_3m"] = raw_levels["yield_3m"].resample("ME").last()
        monthly_levels["yield_5y"] = raw_levels["yield_5y"].resample("ME").last()
        monthly_levels = monthly_levels.dropna()

        # 2. Compute Nominal Monthly Returns
        # S&P 500 log returns
        spx_log_ret = np.log(monthly_levels["spx_close"] / monthly_levels["spx_close"].shift(1))

        # CPI log inflation rate
        cpi_log_ret = np.log(monthly_levels["cpi"] / monthly_levels["cpi"].shift(1))

        # 3M T-Bill nominal return (1/12th of previous month annualized yield)
        yield3m_diff = monthly_levels["yield_3m"].diff()

        # 5Y Note nominal return: Coupon yield - (Modified Duration * Yield Change)
        # Assuming average modified duration D_5 ~ 4.5 years for 5-year Treasuries
        #duration_5y = 4.5
        yield5y_diff = monthly_levels["yield_5y"].diff()

        market_returns = pd.DataFrame(index=monthly_levels.index[1:])
        market_returns["spx_log_return"] = spx_log_ret
        market_returns["cpi_log_return"] = cpi_log_ret
        market_returns["yield_3m_diff"] = yield3m_diff
        market_returns["yield_5y_diff"] = yield5y_diff
        market_returns = market_returns.dropna()

        return monthly_levels, market_returns
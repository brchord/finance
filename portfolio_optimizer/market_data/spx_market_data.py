"""
spx_market_data.py
Simple module to retrieve basic SPX and VIX market data
from IBKR REST Web API.

Useful to retrieve data needed to run backtests and fit
volatility smiles using real data from recent options
Implied Volatility.
"""
import argparse
import copy
import json
import math
import ssl
import sys
import urllib.error
import urllib.request

from datetime import date
from urllib.parse import urlencode

class IBKRSPXMarketData:
    """
    Retrieves SPX and VIX market data from the IBKR
    REST Web API
    """
    # IBKR Web API base URL
    BASE_URL = "https://localhost:5000/v1/api"
    # Historical data API Endpoint
    HIST_DATA_ENDPOINT = "/iserver/marketdata/history"
    # Contract search endpoint, needed to initialize
    # option chain retrieval.
    CON_SEARCH_ENDPOINT = "/iserver/secdef/search"
    # Option strikes retrieval, needed to initialize
    # option chain retrieval.
    STRIKES_ENDPOINT = "/iserver/secdef/strikes"
    # Base query parameters to retrieve 1yr of daily candle data
    BASE_QUERY_PARAMS = {
            "exchange": "SMART",
            "period": "1y",
            "bar": "1d",
            "outsideRth": "false"
    }
    # Option contract validation
    STRIKE_CHECK_ENDPOINT = "/iserver/secdef/info"
    # Live market data endpoint
    LIVE_MARKET_DATA_ENDPOINT = "/iserver/marketdata/snapshot"
    # SPX IBKR Contract ID
    SPX_CON_ID = 416904
    # VIX IBKR Contract ID
    VIX_CON_ID = 13455763
    # How far from the SPX spot (in %) to filter strikes
    # out of that range to avoid excessive IV curve skew.
    DIST_FROM_SPOT = 0.10

    def __init__(self):
        # Bypass self-signed certificate errors common to the local IBKR gateway.
        self.ssl_context = ssl._create_unverified_context()
        self.spx_contract_data = None
        self.vix_contract_data = None


    def _get_request(self, endpoint: str) -> dict:
        """Performs an out-of-the-box REST GET request against the local IBKR gateway."""
        url = f"{IBKRSPXMarketData.BASE_URL}/{endpoint.lstrip('/')}"

        # Standard spoof header so local gateways accept the incoming connection smoothly
        #headers = {"User-Agent": "Python-urllib"}
        req = urllib.request.Request(url) #, headers=headers)

        try:
            # Pass the unverified context directly to urlopen
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                raw_data = response.read().decode("utf-8")
                return json.loads(raw_data)

        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
            return {}
        except urllib.error.URLError as e:
            print(f"Network / Gateway Connection Error: {e.reason}")
            return {}

    def _get_historical_data(self, conid, end_date: date, period="1y", candle_size="1d"):
        """
        Retrieves a year historical daily candle price data
        for the given contract ID and returns it as a JSON blob.
        end_date: the last day of historical data to retrieve
        return: A JSON list containing structured price data
                according to the IBKR Web API specification.
                https://www.interactivebrokers.com/docs/web-api/v1/endpoints/market-data/historical-market-data
        """
        # Define query parameters
        end_date_str = f"{end_date.year}{end_date.month:02d}{end_date.day:02d}"

        query_params = copy.deepcopy(IBKRSPXMarketData.BASE_QUERY_PARAMS)
        query_params["conId"] = conid
        query_params["period"] = period
        query_params["bar"] = candle_size
        query_params["startTime"] = end_date_str

        # Encode into a query string
        query_string = urlencode(query_params)
        endpoint_url = f"{IBKRSPXMarketData.HIST_DATA_ENDPOINT}?{query_string}-00:00:00"
        return self._get_request(endpoint_url)


    def _initialize_spx(self):
        """
        Retrieves the SPX contract since the IBKR API mandates
        the need to retrieve this to successfully retrieve
        market data.
        """
        if not self.spx_contract_data:
            # Query the SPX contract ID to initialize the IBKR client.
            con_endpoint_url = f"{IBKRSPXMarketData.CON_SEARCH_ENDPOINT}?symbol=SPX"
            self.spx_contract_data = self._get_request(con_endpoint_url)


    def _initialize_vix(self):
        """
        Retrieves the VIX contract since the IBKR API mandates
        the need to retrieve this to successfully retrieve
        market data.
        """
        if not self.vix_contract_data:
            # Query the VIX contract ID to initialize the IBKR client.
            con_endpoint_url = f"{IBKRSPXMarketData.CON_SEARCH_ENDPOINT}?symbol=VIX"
            self.vix_contract_data = self._get_request(con_endpoint_url)


    def spx_historical_data(self, end_date: date):
        """
        Retrieves SPX daily historical market data for 1 year
        end_date: Date from where the data will go back in time.
        """
        self._initialize_spx()
        return self._get_historical_data(IBKRSPXMarketData.SPX_CON_ID, end_date)


    def vix_historial_data(self, end_date: date):
        """
        Retrieves VIX daily historical market data for 1 year
        end_date: Date from where the data will go back in time.
        """
        self._initialize_vix()
        return self._get_historical_data(IBKRSPXMarketData.VIX_CON_ID, end_date)


    def _get_spx_spot(self):
        hist_data = self._get_historical_data(
            IBKRSPXMarketData.SPX_CON_ID, date.today(), "1w", "1w")
        last_candle = hist_data["data"][-1]
        return last_candle["c"]

    def _get_strike_contracts(self, spx_spot, option_type="call"):
        opt_contract_data = [x for x in self.spx_contract_data[0]["sections"]
                             if x["secType"] == "OPT"]
        opt_next_month = opt_contract_data[0]["months"].split(";")[1]
        query_params = {
            "conId": IBKRSPXMarketData.SPX_CON_ID,
            "secType": "OPT",
            "month": opt_next_month
        }
        query_string = urlencode(query_params)
        strike_endpoint_url = f"{IBKRSPXMarketData.STRIKES_ENDPOINT}?{query_string}"
        _ = self._get_request(strike_endpoint_url)
        spx_low = spx_spot * (1 - IBKRSPXMarketData.DIST_FROM_SPOT)
        spx_low = math.floor(spx_low / 100.0) * 100
        spx_hi = spx_spot * (1 + IBKRSPXMarketData.DIST_FROM_SPOT)
        spx_hi = math.ceil(spx_hi)
        curr_strike = spx_low
        option_contract_map = {}
        expiration_set = set()
        # Step 3. Validate option strike contracts
        while curr_strike < spx_hi:
            query_params = {
                "conId": IBKRSPXMarketData.SPX_CON_ID,
                "secType": "OPT",
                "month": opt_next_month,
                "strike": curr_strike,
                "right": "C" if option_type == "call" else "P"
            }
            query_string = urlencode(query_params)
            strike_check_url = f"{IBKRSPXMarketData.STRIKE_CHECK_ENDPOINT}?{query_string}"
            opt_contracts = self._get_request(strike_check_url)
            monthlies = [c for c in opt_contracts if c["tradingClass"] == "SPX"]
            if len(monthlies) > 0:
                c = monthlies[0]
                option_contract_map[curr_strike] = c["conid"]
                expiration_set.add(c["maturityDate"])
            curr_strike += 50.0

        assert len(expiration_set) == 1
        return option_contract_map


    def _get_live_market_data(self, conids, fields):
        """
        Retrieves live market data for the given contract ids
        querying the desired field IDs.
        """
        query_string = f"conids={",".join(conids)}&fields={",".join(fields)}"
        api_url = f"{IBKRSPXMarketData.LIVE_MARKET_DATA_ENDPOINT}?{query_string}"
        data = self._get_request(api_url)
        return data


    def spx_current_option_iv_surface(self, option_type="call"):
        """
        Retrieves the IV smile from the closest monthly SPX options
        chain.

        Useful to fit a Volatility Model to properly model IV smiles.

        Returns a list of pairs of the form (strike, IV%) where the
        strikes are in ascending order.
        """
        opt_type = option_type.lower()
        if option_type not in ["call", "put"]:
            raise ValueError("Invalid option type: " + option_type)

        # Step 1. Initialize the client by retrieving the SPX
        #         contract data
        self._initialize_spx()
        # Step 2. Retrieve SPX spot to filter down strikes.
        spx_close = self._get_spx_spot()
        # Step 2. Query the set of strikes for the monthly option contracts
        #         of next month.
        opt_contracts = self._get_strike_contracts(spx_close, opt_type)
        # Step 3. Retrieve option contract prices
        conids = [str(x) for x in list(opt_contracts.values())]
        conid_map = {value: key for (key, value) in opt_contracts.items()}
        # Step 4. Retrieve each option implied volatility and then build a list
        #         of pairs mapping strike -> IV% in strike ascending order.
        market_data = self._get_live_market_data(conids, ["7633"])
        iv_surface = {}
        for market_row in market_data:
            conid = market_row["conid"]
            iv = market_row["7633"]
            iv_surface[conid_map[conid]] = iv
        sorted_strikes = list(iv_surface.keys())
        sorted_strikes.sort()
        sorted_ivs = [iv_surface[s] for s in sorted_strikes]
        return list(zip(sorted_strikes, sorted_ivs))


def main():
    """
    Quick data retrieval tool for SPX, VIX time series
    or SPX volatility smile.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-spx", help="Retrieve last year daily SPX data",
                        action="store_true")
    parser.add_argument("-vix", help="Retrieve last year daily VIX data",
                        action="store_true")
    parser.add_argument("-spx-iv-surface",
                         help="""Retrieve the closest monthly IV surface,
                         argument can be either put or call""")

    args = parser.parse_args()
    ibkr = IBKRSPXMarketData()

    if args.spx:
        spx_candles = ibkr.spx_historical_data(date.today())
        print(spx_candles)

    if args.vix:
        vix_candles = ibkr.vix_historial_data(date.today())
        print(vix_candles)

    if args.spx_iv_surface:
        option_type = args.spx_iv_surface.lower()
        if option_type not in ["call", "put"]:
            print(f"Invalid option type: '{option_type}'", file=sys.stderr)
            sys.exit(1)

        iv_surface = ibkr.spx_current_option_iv_surface(option_type=option_type)
        print(iv_surface)


if __name__ == "__main__":
    main()

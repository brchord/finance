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
import itertools
import logging
import json
import math
import ssl
import sys
import urllib.error
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from urllib.parse import urlencode

import numpy as np

logger = logging.getLogger(__name__)

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
    # VIX3M IBKR Contract ID
    VIX3M_CON_ID = 47511905
    # How far from the SPX spot (in %) to filter strikes
    # out of that range to avoid excessive IV curve skew.
    DIST_FROM_SPOT = 0.10

    def __init__(self):
        # Bypass self-signed certificate errors common to the local IBKR gateway.
        self.ssl_context = ssl._create_unverified_context()
        self.spx_contract_data = None
        self.vix_contract_data = None
        self.vix3m_contract_data = None


    def _get_request(self, endpoint: str) -> dict:
        """Performs an out-of-the-box REST GET request against the local IBKR gateway."""
        url = f"{IBKRSPXMarketData.BASE_URL}/{endpoint.lstrip('/')}"

        logging.debug("Sending request to %s", url)

        # Standard spoof header so local gateways accept the incoming connection smoothly
        #headers = {"User-Agent": "Python-urllib"}
        req = urllib.request.Request(url) #, headers=headers)

        try:
            # Pass the unverified context directly to urlopen
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                raw_data = response.read().decode("utf-8")
                return json.loads(raw_data)

        except urllib.error.HTTPError as e:
            logging.error("HTTP Error %d: %s", e.code, e.read().decode('utf-8'))
            return {}
        except urllib.error.URLError as e:
            logging.error("Network / Gateway Connection Error: %s", e.reason)
            return {}

    def _get_historical_data(self, conid: str, end_date: date,
                             period="1y", candle_size="1d") -> dict:
        """
        Retrieves a year historical daily candle price data
        for the given contract ID and returns it as a JSON blob.
        end_date: the last day of historical data to retrieve
        return: A python dict containing structured price data
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

    def _initialize_vix3m(self):
        if not self.vix3m_contract_data:
            # Query the VIX contract ID to initialize the IBKR client.
            con_endpoint_url = f"{IBKRSPXMarketData.CON_SEARCH_ENDPOINT}?symbol=VIX3M"
            self.vix3m_contract_data = self._get_request(con_endpoint_url)


    def spx_historical_data(self, end_date: date) -> dict:
        """
        Retrieves SPX daily historical market data for 1 year
        end_date: Date from where the data will go back in time.
        """
        self._initialize_spx()
        return self._get_historical_data(IBKRSPXMarketData.SPX_CON_ID, end_date)


    def vix_historial_data(self, end_date: date) -> dict:
        """
        Retrieves VIX daily historical market data for 1 year
        end_date: Date from where the data will go back in time.
        """
        self._initialize_vix()
        return self._get_historical_data(IBKRSPXMarketData.VIX_CON_ID, end_date)

    def vix3m_historial_data(self, end_date: date) -> dict:
        """
        Retrieves VIX3M daily historical market data for 1 year
        end_date: Date from where the data will go back in time.
        """
        self._initialize_vix3m()
        return self._get_historical_data(IBKRSPXMarketData.VIX3M_CON_ID, end_date)


    def _get_vix_spot(self) -> float:
        hist_data = self._get_historical_data(
            IBKRSPXMarketData.VIX_CON_ID, date.today(), "1w", "1w")
        last_candle = hist_data["data"][-1]
        return last_candle["c"]


    def _get_spx_spot(self) -> float:
        hist_data = self._get_historical_data(
            IBKRSPXMarketData.SPX_CON_ID, date.today(), "1w", "1w")
        last_candle = hist_data["data"][-1]
        return last_candle["c"]


    def _get_live_market_data(self, conids: list[str], fields: list[str]) -> dict:
        """
        Retrieves live market data for the given contract ids
        querying the desired field IDs.
        """
        query_string = f"conids={",".join(conids)}&fields={",".join(fields)}"
        api_url = f"{IBKRSPXMarketData.LIVE_MARKET_DATA_ENDPOINT}?{query_string}"
        data = self._get_request(api_url)
        return data


    def _get_strikes_per_maturity(
            self, spx_spot, maturity, option_type="call") -> dict[int, str]:
        spx_low = spx_spot * (1 - IBKRSPXMarketData.DIST_FROM_SPOT)
        spx_low = math.floor(spx_low / 100.0) * 100
        spx_hi = spx_spot # (1 + IBKRSPXMarketData.DIST_FROM_SPOT)
        spx_hi = math.ceil(spx_hi)
        curr_strike = spx_low
        option_contract_map = {}
        expiration_set = set()
        # Step 1. Validate option strike contracts
        while curr_strike < spx_hi:
            query_params = {
                "conId": IBKRSPXMarketData.SPX_CON_ID,
                "secType": "OPT",
                "month": maturity,
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
        return list(expiration_set)[0], option_contract_map


    def _get_strike_contracts(self, spx_spot, option_type="call") -> dict[int, str]:
        opt_contract_data = [x for x in self.spx_contract_data[0]["sections"]
                             if x["secType"] == "OPT"]
        opt_maturities = opt_contract_data[0]["months"].split(";")[1:4]
        for m in opt_maturities:
            query_params = {
                "conId": IBKRSPXMarketData.SPX_CON_ID,
                "secType": "OPT",
                "month": m
            }
            query_string = urlencode(query_params)
            strike_endpoint_url = f"{IBKRSPXMarketData.STRIKES_ENDPOINT}?{query_string}"
            _ = self._get_request(strike_endpoint_url)
        options_chain = {}
        with ThreadPoolExecutor(max_workers=len(opt_maturities)) as executor:
            results = executor.map(
                self._get_strikes_per_maturity,
                itertools.repeat(spx_spot),
                list(opt_maturities),
                itertools.repeat(option_type))
            for i, r in enumerate(results):
                options_chain[opt_maturities[i]] = r

        return options_chain


    def spx_current_option_iv_surface(
            self, option_type="call") -> dict[str, list[tuple[float, str]]]:
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
        self._initialize_vix()
        # Step 2. Retrieve SPX spot to filter down strikes.
        spx_close = self._get_spx_spot()
        vix_close = self._get_vix_spot()
        # Step 2. Query the set of strikes for the monthly option contracts
        #         of next month.
        options_chain = self._get_strike_contracts(spx_close, opt_type)
        options_data = {}
        # Step 3. Retrieve option contract prices
        for _, chain in options_chain.items():
            maturity_date = chain[0]
            option_contracts = chain[1]
            conid_map = {conid: strike for (strike, conid) in option_contracts.items()}
            # Step 4. Retrieve each option implied volatility and then build a list
            #         of pairs mapping strike -> IV% in strike ascending order.
            retries = 3
            while retries > 0:
                try:
                    market_data = self._get_live_market_data(
                        [str(conid) for (_, conid) in option_contracts.items()],
                        ["7633"])
                    logging.debug("First IV value: %s", market_data[0]["7633"])
                    break
                except Exception as e:
                    logging.warning("Error trying to load IV: %s", str(e))
                finally:
                    retries -= 1

            iv_surface = {}
            for market_row in market_data:
                conid = market_row["conid"]
                iv = market_row["7633"]
                iv_surface[conid_map[conid]] = iv
            sorted_strikes = list(iv_surface.keys())
            sorted_strikes.sort()
            sorted_ivs = [iv_surface[s] for s in sorted_strikes]
            options_data[maturity_date] = list(zip(sorted_strikes, sorted_ivs))
        return spx_close, vix_close, options_data

    @classmethod
    def load_iv_surface(cls, json_file):
        """
        Pulls multi month volatility surface data from the output
        of market_data/spx_market_data.py
        """
        with open(json_file, encoding="utf-8") as f:
            iv_surface = json.load(f)
            # Convert IBKR IV string into a floating point number
            # and separate the zipped time series (Strike, IV) into
            # independent arrays.
            surface_spot = iv_surface["spot_spx"]
            surface_atm_iv = iv_surface["spot_vix"] / 100.0
            today = date.today()
            chain = iv_surface["opt_chain"]
            surface_chain = {}
            for exp_str in chain.keys():
                exp_yr = int(exp_str[0:4])
                exp_m = int(exp_str[4:6])
                exp_d = int(exp_str[6:])
                expiration = date(exp_yr, exp_m, exp_d)
                surface_expiration = (expiration - today).days
                surface_expiration *= 1.0/365.0
                surface_data = chain[exp_str]
                surface_strikes = np.zeros(len(surface_data))
                surface_ivs = np.zeros(len(surface_data))
                for i, pair in enumerate(surface_data):
                    surface_strikes[i] = pair[0]
                    iv = float(pair[1][:-1]) / 100.0
                    surface_ivs[i] = iv
                surface_chain[surface_expiration] = (surface_strikes, surface_ivs)
            return surface_spot, surface_atm_iv, surface_chain


def parse_args():
    "Parse command line arguments"
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--historical-spx-vix-vix3m",
                        help="Retrieve last year daily SPX, VIX and VIX3M data",
                        dest="historicals",
                        action="store_true")
    parser.add_argument("-v", "--spx-iv-surface",
                        help="Retrieve the closest monthly IV surface, "
                              "argument can be either put or call",
                        dest="iv_surface")
    parser.add_argument("-o", "--output-file",
                        help="Output file to stream data in JSON format",
                        dest="output_file",
                        required=True)
    return parser.parse_args()


def main():
    """
    Quick data retrieval tool for SPX, VIX time series
    or SPX volatility smile.
    """
    logging.basicConfig(
    format="%(asctime)s:%(filename)s:"
            "%(lineno)d:%(levelname)s: %(message)s",
    level=logging.DEBUG)
    args = parse_args()
    ibkr = IBKRSPXMarketData()

    if args.historicals and args.iv_surface:
        raise ValueError("Please specify only one option to retrieve data.")

    if args.historicals:
        today = date.today()
        spx_candles = ibkr.spx_historical_data(today)
        vix_candles = ibkr.vix_historial_data(today)
        vix3m_candles = ibkr.vix3m_historial_data(today)
        output = {
            "spx": spx_candles,
            "vix": vix_candles,
            "vix3m": vix3m_candles
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f)
            return 0

    if args.iv_surface:
        option_type = args.iv_surface.lower()
        if option_type not in ["call", "put"]:
            logging.error("Invalid option type: '%s'", option_type)
            sys.exit(1)

        spx, vix, options_chain = \
            ibkr.spx_current_option_iv_surface(option_type=option_type)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump({ "spot_spx": spx,
                        "spot_vix": vix,
                        "opt_chain": options_chain
                        }, f)
            return 0

    return 0

if __name__ == "__main__":
    main()

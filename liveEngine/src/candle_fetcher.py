"""
Dynamic Candle Data Fetcher for SmartAPI
Continuously fetches and appends candle data for specified symbols at any interval.
Supports: 1, 3, 5, 10, 15, 30 minutes, 1 hour, and 1 day intervals.
"""

import os
import sys
import json
import pyotp
import pytz
from logzero import logger
import pandas as pd
import warnings
from SmartApi import SmartConnect
from datetime import datetime, timedelta
import time
import requests

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for keys import
from dotenv import load_dotenv

# Load environment variables - find .env relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(ENV_PATH)

# Configuration
DATA_FOLDER = os.path.join(SCRIPT_DIR, '..', 'data', 'candleData')
TOKEN_CACHE_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'memoryData', 'token_map.csv')

# Ensure data folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(TOKEN_CACHE_FILE), exist_ok=True)

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Interval constants for SmartAPI with their minute values
INTERVALS = {
    "ONE_MINUTE": {"api_name": "ONE_MINUTE", "minutes": 1},
    "THREE_MINUTE": {"api_name": "THREE_MINUTE", "minutes": 3},
    "FIVE_MINUTE": {"api_name": "FIVE_MINUTE", "minutes": 5},
    "TEN_MINUTE": {"api_name": "TEN_MINUTE", "minutes": 10},
    "FIFTEEN_MINUTE": {"api_name": "FIFTEEN_MINUTE", "minutes": 15},
    "THIRTY_MINUTE": {"api_name": "THIRTY_MINUTE", "minutes": 30},
    "ONE_HOUR": {"api_name": "ONE_HOUR", "minutes": 60},
    "ONE_DAY": {"api_name": "ONE_DAY", "minutes": 1440}  # 24 * 60
}

# Shorthand interval names for convenience
INTERVAL_SHORTCUTS = {
    "1m": "ONE_MINUTE",
    "3m": "THREE_MINUTE", 
    "5m": "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
    # Also accept full names
    "ONE_MINUTE": "ONE_MINUTE",
    "THREE_MINUTE": "THREE_MINUTE",
    "FIVE_MINUTE": "FIVE_MINUTE",
    "TEN_MINUTE": "TEN_MINUTE",
    "FIFTEEN_MINUTE": "FIFTEEN_MINUTE",
    "THIRTY_MINUTE": "THIRTY_MINUTE",
    "ONE_HOUR": "ONE_HOUR",
    "ONE_DAY": "ONE_DAY",
}

# Default symbols to track (can be modified)
DEFAULT_SYMBOLS = [
    {"symbol": "NIFTY", "token": "99926000", "exchange": "NSE"},
    {"symbol": "BANKNIFTY", "token": "99926009", "exchange": "NSE"},
    {"symbol": "RELIANCE-EQ", "token": "2885", "exchange": "NSE"},
    {"symbol": "HDFCBANK-EQ", "token": "1333", "exchange": "NSE"},
    {"symbol": "INFY-EQ", "token": "1594", "exchange": "NSE"},
]


class SmartAPIClient:
    """SmartAPI client wrapper for authentication and data fetching."""
    
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.client_code = os.getenv("CLIENT_CODE")
        self.password = os.getenv("PASSWORD")
        self.totp_secret = os.getenv("TOTP_SECRET")
        self.obj = None
        self.auth_token = None
        self.feed_token = None
        self.token_df = None
        
        # Debug: Print loaded credentials (masked)
        logger.info(f"Loaded .env from: {ENV_PATH}")
        logger.info(f"API_KEY: {'*' * (len(self.api_key)-3) + self.api_key[-3:] if self.api_key else 'NOT SET'}")
        logger.info(f"CLIENT_CODE: {self.client_code if self.client_code else 'NOT SET'}")
        logger.info(f"PASSWORD: {'****' if self.password else 'NOT SET'}")
        logger.info(f"TOTP_SECRET: {'*' * (len(self.totp_secret)-4) + self.totp_secret[-4:] if self.totp_secret else 'NOT SET'}")
        
    def login(self):
        """Authenticate with SmartAPI and generate session."""
        try:
            self.obj = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            
            data = self.obj.generateSession(self.client_code, self.password, totp)
            
            if data.get("status"):
                self.auth_token = data['data']['jwtToken']
                self.feed_token = self.obj.getfeedToken()
                logger.info("Login Successful")
                logger.info(f"Auth Token: {self.auth_token[:50]}...")
                return True
            else:
                logger.error(f"Login failed: {data.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def initialize_token_map(self):
        """Download and cache the symbol token map."""
        try:
            url = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'
            response = requests.get(url)
            
            if response.status_code == 200:
                d = response.json()
                self.token_df = pd.DataFrame.from_dict(d)
                self.token_df['expiry'] = pd.to_datetime(self.token_df['expiry'], errors='coerce')
                self.token_df.to_csv(TOKEN_CACHE_FILE, index=False)
                logger.info(f"Token map downloaded: {len(self.token_df)} symbols")
                return True
            else:
                logger.error(f"Failed to download token map: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Token map error: {e}")
            return False
    
    def get_symbol_token(self, symbol: str, exchange: str = "NSE") -> str:
        """Get token for a symbol from the token map."""
        if self.token_df is None:
            # Try to load from cache
            if os.path.exists(TOKEN_CACHE_FILE):
                self.token_df = pd.read_csv(TOKEN_CACHE_FILE)
            else:
                self.initialize_token_map()
        
        if self.token_df is not None:
            mask = (self.token_df['symbol'] == symbol) & (self.token_df['exch_seg'] == exchange)
            result = self.token_df[mask]
            if not result.empty:
                return str(result.iloc[0]['token'])
        
        return None
    
    def fetch_candle_data(self, exchange: str, token: str, from_date: str, 
                          to_date: str, interval: str = "FIVE_MINUTE") -> pd.DataFrame:
        """
        Fetch historical candle data from SmartAPI.
        
        Args:
            exchange: Exchange code (NSE, NFO, BSE, etc.)
            token: Symbol token
            from_date: Start date in "YYYY-MM-DD HH:MM" format
            to_date: End date in "YYYY-MM-DD HH:MM" format
            interval: Candle interval (ONE_MINUTE, FIVE_MINUTE, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            response = self.obj.getCandleData(params)
            
            if response and response.get('status') and response.get('data'):
                columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
                df = pd.DataFrame(response['data'], columns=columns)
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df.set_index('DateTime', inplace=True)
                return df
            else:
                logger.warning(f"No data returned for token {token}: {response}")
                return pd.DataFrame()
                
        except Exception as e:
            if "exceeding access rate" in str(e).lower():
                logger.warning("Rate limit hit, waiting 2 seconds...")
                time.sleep(2)
                return self.fetch_candle_data(exchange, token, from_date, to_date, interval)
            else:
                logger.error(f"Error fetching candle data: {e}")
                return pd.DataFrame()


class CandleDataManager:
    """Manages continuous fetching and storage of candle data."""
    
    def __init__(self, client: SmartAPIClient, symbols: list = None, interval: str = "5m"):
        self.client = client
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.data_cache = {}  # In-memory cache of DataFrames
        
        # Resolve interval shortcut to full name
        self.interval_key = INTERVAL_SHORTCUTS.get(interval.upper() if interval.upper() in INTERVAL_SHORTCUTS else interval, interval)
        if self.interval_key not in INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid options: {list(INTERVAL_SHORTCUTS.keys())}")
        
        self.interval_config = INTERVALS[self.interval_key]
        self.interval_minutes = self.interval_config["minutes"]
        self.interval_api_name = self.interval_config["api_name"]
        
    def get_csv_path(self, symbol: str) -> str:
        """Get CSV file path for a symbol with interval suffix."""
        safe_symbol = symbol.replace("-", "_").replace("&", "_")
        interval_suffix = f"{self.interval_minutes}min" if self.interval_minutes < 60 else (f"{self.interval_minutes // 60}h" if self.interval_minutes < 1440 else "1d")
        return os.path.join(DATA_FOLDER, f"{safe_symbol}_{interval_suffix}.csv")
    
    def load_existing_data(self, symbol: str) -> pd.DataFrame:
        """Load existing data from CSV if available."""
        csv_path = self.get_csv_path(symbol)
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, parse_dates=['DateTime'], index_col='DateTime')
                logger.info(f"Loaded {len(df)} existing candles for {symbol}")
                return df
            except Exception as e:
                logger.error(f"Error loading {csv_path}: {e}")
        
        return pd.DataFrame()
    
    def save_data(self, symbol: str, df: pd.DataFrame):
        """Save DataFrame to CSV."""
        if df.empty:
            return
            
        csv_path = self.get_csv_path(symbol)
        df.to_csv(csv_path)
        logger.info(f"Saved {len(df)} candles for {symbol} to {csv_path}")
    
    def fetch_and_append(self, symbol_info: dict, lookback_candles: int = 5):
        """
        Fetch latest COMPLETED candles and append to existing data.
        Only fetches candles that are fully closed (not the current incomplete candle).
        
        Args:
            symbol_info: Dict with symbol, token, exchange
            lookback_candles: How many candles back to fetch (to catch any missed candles)
        """
        symbol = symbol_info['symbol']
        token = symbol_info['token']
        exchange = symbol_info['exchange']
        
        # Load existing data
        if symbol not in self.data_cache:
            self.data_cache[symbol] = self.load_existing_data(symbol)
        
        existing_df = self.data_cache[symbol]
        
        # Get current time and calculate the last COMPLETED candle based on interval
        now = datetime.now(IST)
        
        # Calculate the last completed candle time based on interval
        last_completed_candle_time = self._get_last_completed_candle_time(now)
        
        to_date = last_completed_candle_time.strftime("%Y-%m-%d %H:%M")
        
        if existing_df.empty:
            # First fetch - get last 2 days of data (or more for daily candles)
            lookback_days = 2 if self.interval_minutes < 1440 else 30
            from_date = (last_completed_candle_time - timedelta(days=lookback_days)).strftime("%Y-%m-%d 09:15")
        else:
            # Get data from last candle time (with some overlap for safety)
            last_candle = existing_df.index.max()
            if last_candle.tzinfo is None:
                last_candle = IST.localize(last_candle)
            lookback_minutes = self.interval_minutes * lookback_candles
            from_date = (last_candle - timedelta(minutes=lookback_minutes)).strftime("%Y-%m-%d %H:%M")
        
        logger.info(f"Fetching {symbol} [{self.interval_api_name}] from {from_date} to {to_date}")
        
        # Fetch new data
        new_df = self.client.fetch_candle_data(
            exchange=exchange,
            token=token,
            from_date=from_date,
            to_date=to_date,
            interval=self.interval_api_name
        )
        
        if new_df.empty:
            logger.warning(f"No new data for {symbol}")
            return
        
        # Merge with existing data (drop duplicates based on index)
        if existing_df.empty:
            merged_df = new_df
        else:
            merged_df = pd.concat([existing_df, new_df])
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            merged_df = merged_df.sort_index()
        
        # Update cache and save
        self.data_cache[symbol] = merged_df
        self.save_data(symbol, merged_df)
        
        new_candles = len(merged_df) - len(existing_df)
        logger.info(f"{symbol}: {new_candles} new candles added, total: {len(merged_df)}")
        
        return merged_df
    
    def _get_last_completed_candle_time(self, now: datetime) -> datetime:
        """
        Calculate the start time of the last fully completed candle.
        Accounts for market hours: 9:15 AM to 3:30 PM IST.
        First candle always starts at 9:15.
        
        Args:
            now: Current datetime
            
        Returns:
            datetime of the last completed candle's start time
        """
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
        
        # Minutes from market open (9:15 = 0)
        market_open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE  # 555 minutes
        market_close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE  # 930 minutes
        
        if self.interval_minutes >= 1440:  # Daily candles
            # Daily candle completes at market close (15:30)
            # Last completed daily candle is today if market closed, else yesterday
            if now >= market_close:
                return market_open  # Today's candle is complete
            else:
                return market_open - timedelta(days=1)  # Yesterday's candle
        
        # For intraday candles, calculate based on market hours
        current_minutes = now.hour * 60 + now.minute
        
        # If before market open, last completed candle was from previous trading day
        if current_minutes < market_open_minutes:
            # Return last candle of previous day
            prev_day_close = market_close - timedelta(days=1)
            return self._get_last_candle_before_close(prev_day_close - timedelta(days=1))
        
        # If after market close, last completed candle is the last one of today
        if current_minutes >= market_close_minutes:
            return self._get_last_candle_before_close(now)
        
        # During market hours - calculate based on interval from market open (9:15)
        minutes_since_market_open = current_minutes - market_open_minutes
        
        # How many complete intervals since market open?
        complete_intervals = minutes_since_market_open // self.interval_minutes
        
        # If we're exactly at an interval boundary and within first 5 seconds, go back one
        if minutes_since_market_open % self.interval_minutes == 0 and now.second < 5:
            complete_intervals = max(0, complete_intervals - 1)
        
        # Last completed candle start time
        last_candle_minutes = market_open_minutes + (complete_intervals * self.interval_minutes)
        last_candle_hour = last_candle_minutes // 60
        last_candle_min = last_candle_minutes % 60
        
        return now.replace(hour=last_candle_hour, minute=last_candle_min, second=0, microsecond=0)
    
    def _get_last_candle_before_close(self, date: datetime) -> datetime:
        """
        Get the start time of the last candle before market close on a given date.
        
        Args:
            date: The date to calculate for
            
        Returns:
            datetime of the last candle's start time
        """
        market_open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE  # 555
        market_close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE  # 930
        
        # Total trading minutes
        trading_minutes = market_close_minutes - market_open_minutes  # 375 minutes
        
        # Number of complete candles in a day
        complete_candles = trading_minutes // self.interval_minutes
        
        # Last candle starts at
        last_candle_offset = (complete_candles - 1) * self.interval_minutes
        last_candle_minutes = market_open_minutes + last_candle_offset
        
        last_candle_hour = last_candle_minutes // 60
        last_candle_min = last_candle_minutes % 60
        
        return date.replace(hour=last_candle_hour, minute=last_candle_min, second=0, microsecond=0)
    
    def _seconds_until_next_candle(self) -> int:
        """
        Calculate seconds until the next candle completes.
        Accounts for market hours: 9:15 AM to 3:30 PM IST.
        
        Returns:
            Seconds to wait (including 5-second buffer), or -1 if market is closed
        """
        now = datetime.now(IST)
        
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
        
        market_open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
        market_close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE
        current_minutes = now.hour * 60 + now.minute
        
        if self.interval_minutes >= 1440:  # Daily candles
            # Next candle completes at 15:30 (market close)
            if now >= market_close:
                # Market closed, wait until next day's market close
                next_close = market_close + timedelta(days=1)
                return int((next_close - now).total_seconds()) + 5
            return int((market_close - now).total_seconds()) + 5
        
        # Before market open - wait until first candle completes
        if current_minutes < market_open_minutes:
            # First candle completes at 9:15 + interval
            first_candle_end_minutes = market_open_minutes + self.interval_minutes
            first_candle_end = now.replace(
                hour=first_candle_end_minutes // 60,
                minute=first_candle_end_minutes % 60,
                second=0, microsecond=0
            )
            return int((first_candle_end - now).total_seconds()) + 5
        
        # After market close - wait until next day's first candle completes
        if current_minutes >= market_close_minutes:
            tomorrow_open = market_open + timedelta(days=1)
            first_candle_end_minutes = market_open_minutes + self.interval_minutes
            first_candle_end = tomorrow_open.replace(
                hour=first_candle_end_minutes // 60,
                minute=first_candle_end_minutes % 60,
                second=0, microsecond=0
            )
            return int((first_candle_end - now).total_seconds()) + 5
        
        # During market hours - calculate next candle boundary from 9:15
        minutes_since_market_open = current_minutes - market_open_minutes
        current_candle_number = minutes_since_market_open // self.interval_minutes
        next_candle_end_offset = (current_candle_number + 1) * self.interval_minutes
        next_candle_end_minutes = market_open_minutes + next_candle_end_offset
        
        # If next candle would be after market close, return -1 (will trigger end-of-day logic)
        if next_candle_end_minutes > market_close_minutes:
            # Wait for market close to get the last candle
            return int((market_close - now).total_seconds()) + 5
        
        next_candle_end = now.replace(
            hour=next_candle_end_minutes // 60,
            minute=next_candle_end_minutes % 60,
            second=0, microsecond=0
        )
        
        return int((next_candle_end - now).total_seconds()) + 5
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(IST)
        current_minutes = now.hour * 60 + now.minute
        market_open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
        market_close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE
        return market_open_minutes <= current_minutes < market_close_minutes
    
    def run_continuous(self):
        """
        Continuously fetch and append COMPLETED candle data for all symbols.
        Waits for each candle interval boundary to ensure candles are complete.
        Respects market hours: 9:15 AM to 3:30 PM IST.
        """
        interval_str = f"{self.interval_minutes}min" if self.interval_minutes < 60 else (f"{self.interval_minutes // 60}h" if self.interval_minutes < 1440 else "1d")
        
        logger.info(f"Starting continuous candle fetcher for {len(self.symbols)} symbols")
        logger.info(f"Interval: {self.interval_api_name} ({interval_str})")
        logger.info(f"Market hours: {MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE:02d} - {MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE:02d} IST")
        logger.info(f"Data folder: {DATA_FOLDER}")
        logger.info(f"Will fetch only COMPLETED {interval_str} candles")
        
        cycle_count = 0
        
        while True:
            now = datetime.now(IST)
            market_status = "OPEN" if self._is_market_open() else "CLOSED"
            
            # Calculate seconds until next candle completes
            wait_seconds = self._seconds_until_next_candle()
            
            if wait_seconds > 10:  # Only wait if more than 10 seconds away
                wait_mins = wait_seconds // 60
                wait_secs = wait_seconds % 60
                if wait_mins > 60:
                    logger.info(f"Market {market_status}. Waiting {wait_mins // 60}h {wait_mins % 60}m for next {interval_str} candle...")
                elif wait_mins > 0:
                    logger.info(f"Market {market_status}. Waiting {wait_mins}m {wait_secs}s for next {interval_str} candle...")
                else:
                    logger.info(f"Market {market_status}. Waiting {wait_secs}s for next {interval_str} candle...")
                time.sleep(wait_seconds)
            
            cycle_count += 1
            cycle_start = datetime.now(IST)
            logger.info(f"\n{'='*50}")
            logger.info(f"Cycle {cycle_count} started at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
            
            for symbol_info in self.symbols:
                try:
                    self.fetch_and_append(symbol_info)
                    time.sleep(1)  # Rate limit: 1 request per second
                except Exception as e:
                    logger.error(f"Error processing {symbol_info['symbol']}: {e}")
            
            cycle_duration = (datetime.now(IST) - cycle_start).total_seconds()
            logger.info(f"Cycle {cycle_count} completed in {cycle_duration:.1f}s")
            
            # Small sleep to avoid tight loop if something goes wrong
            time.sleep(5)


def main():
    """Main entry point."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dynamic Candle Data Fetcher for SmartAPI")
    parser.add_argument(
        "-i", "--interval",
        type=str,
        default="5m",
        help="Candle interval: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 1d (default: 5m)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch once and exit (don't run continuously)"
    )
    args = parser.parse_args()
    
    interval_str = args.interval
    
    logger.info("=" * 60)
    logger.info("Dynamic Candle Data Fetcher")
    logger.info(f"Interval: {interval_str}")
    logger.info("=" * 60)
    
    # Initialize client
    client = SmartAPIClient()
    
    # Login
    if not client.login():
        logger.error("Failed to login. Exiting.")
        return
    
    # Initialize token map (for symbol lookups)
    client.initialize_token_map()
    
    # Custom symbols (modify as needed)
    # You can add more symbols here with their token and exchange
    symbols = [
        {"symbol": "NIFTY", "token": "99926000", "exchange": "NSE"},
        {"symbol": "BANKNIFTY", "token": "99926009", "exchange": "NSE"},
        {"symbol": "RELIANCE-EQ", "token": "2885", "exchange": "NSE"},
        {"symbol": "HDFCBANK-EQ", "token": "1333", "exchange": "NSE"},
        {"symbol": "INFY-EQ", "token": "1594", "exchange": "NSE"},
        {"symbol": "TCS-EQ", "token": "11536", "exchange": "NSE"},
        {"symbol": "ICICIBANK-EQ", "token": "4963", "exchange": "NSE"},
    ]
    
    # Initialize data manager with chosen interval
    manager = CandleDataManager(client, symbols, interval=interval_str)
    
    if args.once:
        # Fetch once and exit
        logger.info("Fetching data once...")
        for symbol_info in symbols:
            try:
                manager.fetch_and_append(symbol_info)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing {symbol_info['symbol']}: {e}")
        logger.info("Done!")
    else:
        # Run continuous fetcher
        manager.run_continuous()


if __name__ == "__main__":
    main()

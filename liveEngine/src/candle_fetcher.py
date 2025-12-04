"""
5-Minute Candle Data Fetcher for SmartAPI
Continuously fetches and appends 5-minute candle data for specified symbols.
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

# Interval constants for SmartAPI
INTERVALS = {
    "ONE_MINUTE": "ONE_MINUTE",
    "THREE_MINUTE": "THREE_MINUTE",
    "FIVE_MINUTE": "FIVE_MINUTE",
    "TEN_MINUTE": "TEN_MINUTE",
    "FIFTEEN_MINUTE": "FIFTEEN_MINUTE",
    "THIRTY_MINUTE": "THIRTY_MINUTE",
    "ONE_HOUR": "ONE_HOUR",
    "ONE_DAY": "ONE_DAY"
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
    
    def __init__(self, client: SmartAPIClient, symbols: list = None):
        self.client = client
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.data_cache = {}  # In-memory cache of DataFrames
        
    def get_csv_path(self, symbol: str) -> str:
        """Get CSV file path for a symbol."""
        safe_symbol = symbol.replace("-", "_").replace("&", "_")
        return os.path.join(DATA_FOLDER, f"{safe_symbol}_5min.csv")
    
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
    
    def fetch_and_append(self, symbol_info: dict, lookback_minutes: int = 30):
        """
        Fetch latest candles and append to existing data.
        
        Args:
            symbol_info: Dict with symbol, token, exchange
            lookback_minutes: How far back to fetch (to catch any missed candles)
        """
        symbol = symbol_info['symbol']
        token = symbol_info['token']
        exchange = symbol_info['exchange']
        
        # Load existing data
        if symbol not in self.data_cache:
            self.data_cache[symbol] = self.load_existing_data(symbol)
        
        existing_df = self.data_cache[symbol]
        
        # Determine fetch window
        now = datetime.now(IST)
        
        if existing_df.empty:
            # First fetch - get last 2 days of data
            from_date = (now - timedelta(days=2)).strftime("%Y-%m-%d 09:15")
        else:
            # Get data from last candle time (with some overlap for safety)
            last_candle = existing_df.index.max()
            if last_candle.tzinfo is None:
                last_candle = IST.localize(last_candle)
            from_date = (last_candle - timedelta(minutes=lookback_minutes)).strftime("%Y-%m-%d %H:%M")
        
        to_date = now.strftime("%Y-%m-%d %H:%M")
        
        logger.info(f"Fetching {symbol} from {from_date} to {to_date}")
        
        # Fetch new data
        new_df = self.client.fetch_candle_data(
            exchange=exchange,
            token=token,
            from_date=from_date,
            to_date=to_date,
            interval="FIVE_MINUTE"
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
    
    def run_continuous(self, interval_seconds: int = 60):
        """
        Continuously fetch and append data for all symbols.
        
        Args:
            interval_seconds: Seconds between fetch cycles (default 60 for 5-min candles)
        """
        logger.info(f"Starting continuous candle fetcher for {len(self.symbols)} symbols")
        logger.info(f"Fetch interval: {interval_seconds} seconds")
        logger.info(f"Data folder: {DATA_FOLDER}")
        
        cycle_count = 0
        
        while True:
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
            sleep_time = max(0, interval_seconds - cycle_duration)
            
            logger.info(f"Cycle {cycle_count} completed in {cycle_duration:.1f}s")
            logger.info(f"Sleeping for {sleep_time:.1f}s until next cycle")
            
            time.sleep(sleep_time)


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("5-Minute Candle Data Fetcher")
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
    
    # Initialize data manager
    manager = CandleDataManager(client, symbols)
    
    # Run continuous fetcher
    # Fetch every 60 seconds (adjust based on your needs)
    # For 5-minute candles, fetching every 60 seconds ensures you catch each new candle
    manager.run_continuous(interval_seconds=60)


if __name__ == "__main__":
    main()

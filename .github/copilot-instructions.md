# Copilot Instructions for OptionsIV-Dashboard

## Project Overview

Options IV (Implied Volatility) Dashboard for Indian stock markets. Uses **Angel Broking SmartAPI** to fetch real-time option Greeks data and display via Flask web interface.

### Architecture

```
┌──────────────────┐      ┌──────────────────┐
│   liveEngine/    │      │   dashboard/     │
│   src/main.py    │      │   app.py         │
│   (Standalone    │      │   (Flask + Data  │
│    Data Fetcher) │      │    Fetcher)      │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └────────┬────────────────┘
                  ▼
         SmartAPI (Angel Broking)
         - optionGreek() endpoint
         - Rate limited (1 req/sec)
```

**Two entry points:**
- `liveEngine/src/main.py` - Standalone data collector, saves to `data/IVjsons/`
- `dashboard/algoDashboard/app.py` - Flask app with integrated data fetching (port 5006)

## Critical Dependencies

```python
# Core packages (not in requirements.txt - manually install)
SmartApi       # Angel Broking API client
pyotp          # TOTP 2FA authentication
logzero        # Logging
pandas         # Data manipulation
flask          # Web framework
python-dotenv  # Environment variables
```

## Authentication Pattern

Credentials in `.env` file (gitignored), loaded via `keys.py`:

```python
# dashboard/algoDashboard/keys.py pattern
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")
client_code = os.getenv("CLIENT_CODE")
password = os.getenv("PASSWORD")
totp_secret = os.getenv("TOTP_SECRET")
```

**SmartAPI login flow:**
```python
obj = SmartConnect(api_key=api_key)
token = pyotp.TOTP(totp_secret).now()  # Time-based OTP
data = obj.generateSession(client_code, password, token)
```

## Key Data Flow

1. **Symbol Token Map** - Fetched from Angel Broking OpenAPI:
   ```python
   url = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'
   ```
   Filtered by `name_list` (predefined stocks), saved to `data/memoryData/token_filtered.csv`

2. **Option Greeks** - Fetched per stock/expiry:
   ```python
   params = {"name": stock, "expirydate": expiry.strftime("%d%b%Y").upper()}
   obj.optionGreek(params)  # Returns delta, gamma, theta, vega, IV
   ```

3. **Rate Limiting** - 1 second delay between API calls; retry on "exceeding access rate" errors

## File Conventions

| Path | Purpose |
|------|---------|
| `data/IVjsons/{STOCK}.json` | Cached option Greeks per stock |
| `data/memoryData/token_filtered.csv` | Symbol mapping cache |
| `.env` | API credentials (never commit) |
| `liveEngine/config/config.ini` | Placeholder for config (currently unused) |

## Stock List Management

Stocks are hardcoded in both `main.py` and `app.py` as `name_list`. Dashboard version has extended list including indices:
- NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY, NIFTYNXT50 (indices - weekly expiries)
- Individual stocks (monthly expiries only - last Thursday of month)

## Flask Dashboard Routes

- `GET /` - Main UI with stock/expiry selectors
- `GET /get_data` - JSON API returning filtered Greeks for selected stock/expiry

## Development Workflow

```bash
# Setup
pip install smartapi-python pyotp logzero pandas flask python-dotenv requests

# Run dashboard (includes data fetching)
cd dashboard/algoDashboard
python app.py  # Serves on http://localhost:5006

# Run standalone data collector
cd liveEngine/src
python main.py  # Fetches all stocks continuously
```

## Code Patterns to Follow

- Use `logzero.logger` for all logging
- Suppress warnings: `warnings.filterwarnings('ignore')`
- Handle rate limits with exponential backoff (currently 2s retry)
- Store fetched data as JSON with indent=4
- Use pandas for DataFrame operations on token map

## Known Issues

- `requirements.txt` is empty - dependencies not documented
- `liveEngine/src/main.py` has hardcoded credentials (should use keys.py pattern)
- `config/config.ini` and `docs/setup_guide.md` are empty placeholders
- `backtesting/` notebooks are empty - intended for future strategy testing

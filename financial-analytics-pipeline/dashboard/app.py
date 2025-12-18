import streamlit as st
import os
import sqlite3
import pandas as pd
from pathlib import Path

def db_explorer_launcher():
    st.subheader("Database Explorer")
    st.info("The DB Explorer now runs as a separate app for security and isolation.")
    if st.button("Open DB Explorer in New Tab"):
        db_url = "http://localhost:8502"  # Adjust if you run DB Explorer elsewhere
        js = f"window.open('{db_url}', '_blank')"
        st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)

def db_explorer_page():
    st.title("🗄️ Database Explorer")
    db_path = str(DB_PATH)
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}")
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    db_explorer_launcher()

import streamlit as st


# Main dashboard logic
# (Removed obsolete block that referenced undefined 'page'. Navigation is handled in main().)
from typing import Optional
import sqlite3
import pandas as pd
import numpy as np
import os
from pathlib import Path
import requests

# --- Asset Name/Symbol Resolution ---
def resolve_asset_symbol(asset_query: str, asset_type: str) -> Optional[str]:
    # Hardcoded mapping for common US stocks and cryptos
    STOCK_NAME_MAP = {
        'microsoft': 'MSFT', 'msft': 'MSFT',
        'apple': 'AAPL', 'aapl': 'AAPL',
        'nvidia': 'NVDA', 'nvda': 'NVDA',
        'google': 'GOOGL', 'googl': 'GOOGL',
        'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'amzn': 'AMZN',
        'meta': 'META', 'facebook': 'META', 'fb': 'META',
        'tesla': 'TSLA', 'tsla': 'TSLA',
        'berkshire': 'BRK.B', 'brk.b': 'BRK.B',
        'unitedhealth': 'UNH', 'unh': 'UNH',
        'visa': 'V', 'v': 'V',
        'jpmorgan': 'JPM', 'jpm': 'JPM',
        'johnson & johnson': 'JNJ', 'jnj': 'JNJ',
        'walmart': 'WMT', 'wmt': 'WMT',
        'procter & gamble': 'PG', 'pg': 'PG',
        'mastercard': 'MA', 'ma': 'MA',
        'exxon': 'XOM', 'xom': 'XOM',
        'costco': 'COST', 'cost': 'COST',
        'pepsico': 'PEP', 'pep': 'PEP',
        'coca-cola': 'KO', 'ko': 'KO',
        'adobe': 'ADBE', 'adbe': 'ADBE',
        'cisco': 'CSCO', 'csco': 'CSCO',
        'oracle': 'ORCL', 'orcl': 'ORCL',
        'intel': 'INTC', 'intc': 'INTC',
        'netflix': 'NFLX', 'nflx': 'NFLX',
        'paypal': 'PYPL', 'pypl': 'PYPL',
        'starbucks': 'SBUX', 'sbux': 'SBUX',
        'mcdonalds': 'MCD', 'mcd': 'MCD',
        'boeing': 'BA', 'ba': 'BA',
        'chevron': 'CVX', 'cvx': 'CVX',
        'ibm': 'IBM', 'ibm': 'IBM',
        'ford': 'F', 'f': 'F',
        'pfizer': 'PFE', 'pfe': 'PFE',
        'at&t': 'T', 't': 'T',
        'verizon': 'VZ', 'vz': 'VZ',
        'abbott': 'ABT', 'abt': 'ABT',
        'broadcom': 'AVGO', 'avgo': 'AVGO',
        'qualcomm': 'QCOM', 'qcom': 'QCOM',
        'salesforce': 'CRM', 'crm': 'CRM',
        'wells fargo': 'WFC', 'wfc': 'WFC',
        'goldman sachs': 'GS', 'gs': 'GS',
        'blackrock': 'BLK', 'blk': 'BLK',
        'caterpillar': 'CAT', 'cat': 'CAT',
        'lockheed martin': 'LMT', 'lmt': 'LMT',
        'general electric': 'GE', 'ge': 'GE',
        'american express': 'AXP', 'axp': 'AXP',
        'nike': 'NKE', 'nke': 'NKE',
        'united parcel': 'UPS', 'ups': 'UPS',
        '3m': 'MMM', 'mmm': 'MMM',
        'dow': 'DOW', 'dow': 'DOW',
        'dupont': 'DD', 'dd': 'DD',
        'altria': 'MO', 'mo': 'MO',
        'cvs': 'CVS', 'cvs': 'CVS',
        'gilead': 'GILD', 'gild': 'GILD',
        'moderna': 'MRNA', 'mrna': 'MRNA',
        'biontech': 'BNTX', 'bntx': 'BNTX',
        'regeneron': 'REGN', 'regn': 'REGN',
        'biogen': 'BIIB', 'biib': 'BIIB',
        'amgen': 'AMGN', 'amgn': 'AMGN',
        'eli lilly': 'LLY', 'lly': 'LLY',
        'merck': 'MRK', 'mrk': 'MRK',
        'bristol-myers': 'BMY', 'bmy': 'BMY',
        'johnson controls': 'JCI', 'jci': 'JCI',
        'honeywell': 'HON', 'hon': 'HON',
        'prologis': 'PLD', 'pld': 'PLD',
        'cigna': 'CI', 'ci': 'CI',
        'anthem': 'ELV', 'elv': 'ELV',
        'centene': 'CNC', 'cnc': 'CNC',
        'humana': 'HUM', 'hum': 'HUM',
        'united rentals': 'URI', 'uri': 'URI',
        'paccar': 'PCAR', 'pcar': 'PCAR',
        'palo alto': 'PANW', 'panw': 'PANW',
        'snowflake': 'SNOW', 'snow': 'SNOW',
        'datadog': 'DDOG', 'ddog': 'DDOG',
        'zoom': 'ZM', 'zm': 'ZM',
        'block': 'SQ', 'sq': 'SQ',
        'coinbase': 'COIN', 'coin': 'COIN',
        'robinhood': 'HOOD', 'hood': 'HOOD',
        'shopify': 'SHOP', 'shop': 'SHOP',
        'spotify': 'SPOT', 'spot': 'SPOT',
        'twilio': 'TWLO', 'twlo': 'TWLO',
        'uber': 'UBER', 'uber': 'UBER',
        'lyft': 'LYFT', 'lyft': 'LYFT',
        'airbnb': 'ABNB', 'abnb': 'ABNB',
        'palantir': 'PLTR', 'pltr': 'PLTR',
        'zillow': 'ZG', 'zg': 'ZG',
        'zillow group': 'ZG',
    }
    CRYPTO_NAME_MAP = {
        'bitcoin': 'BTC', 'btc': 'BTC',
        'ethereum': 'ETH', 'eth': 'ETH',
        'binance': 'BNB', 'bnb': 'BNB',
        'solana': 'SOL', 'sol': 'SOL',
        'cardano': 'ADA', 'ada': 'ADA',
        'ripple': 'XRP', 'xrp': 'XRP',
        'polkadot': 'DOT', 'dot': 'DOT',
        'dogecoin': 'DOGE', 'doge': 'DOGE',
        'avalanche': 'AVAX', 'avax': 'AVAX',
        'matic': 'MATIC', 'polygon': 'MATIC',
        'chainlink': 'LINK', 'link': 'LINK',
        'uniswap': 'UNI', 'uni': 'UNI',
        'litecoin': 'LTC', 'ltc': 'LTC',
        'cosmos': 'ATOM', 'atom': 'ATOM',
        'stellar': 'XLM', 'xlm': 'XLM',
        'algorand': 'ALGO', 'algo': 'ALGO',
        'vechain': 'VET', 'vet': 'VET',
        'internet computer': 'ICP', 'icp': 'ICP',
        'filecoin': 'FIL', 'fil': 'FIL',
        'tron': 'TRX', 'trx': 'TRX',
    }
    query_lower = asset_query.strip().lower()
    if asset_type == "stock" and query_lower in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[query_lower]
    if asset_type == "crypto" and query_lower in CRYPTO_NAME_MAP:
        return CRYPTO_NAME_MAP[query_lower]

    """Resolve asset name or symbol to a valid symbol/ID for price lookup."""
    if asset_type == "crypto":
        # Try CoinGecko search endpoint
        url = f"{COINGECKO_API}/search"
        params = {"query": asset_query}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Prefer exact symbol match, else take first coin result
                for coin in data.get("coins", []):
                    if coin["symbol"].upper() == asset_query.upper():
                        return coin["id"]
                if data.get("coins"):
                    return data["coins"][0]["id"]
        except Exception:
            pass
    else:
        # Try TwelveData symbol search
        url = f"{TWELVEDATA_API}/symbol_search"
        params = {"symbol": asset_query, "apikey": TWELVEDATA_API_KEY}
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"]:
                    # Only accept NASDAQ/NYSE tickers and prefer exact or well-known matches
                    query_upper = asset_query.upper()
                    us_tickers = {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "BRK.B", "UNH", "V", "JPM", "JNJ", "WMT", "PG", "MA", "HD", "XOM", "LLY", "ABBV", "COST", "AVGO", "MRK", "PEP", "KO", "ADBE", "CSCO", "CRM", "TMO", "ABT", "MCD", "ACN", "DHR", "NKE", "LIN", "WFC", "AMD", "TXN", "NEE", "PM", "BMY", "AMGN", "UNP", "LOW", "QCOM", "MS", "INTC", "HON", "SBUX", "AMAT", "CVX", "GS", "RTX", "CAT", "GE", "ISRG", "MDT", "SPGI", "BLK", "SCHW", "ZTS", "PLD", "LMT", "T", "SYK", "AXP", "NOW", "DE", "C", "BKNG", "GILD", "MO", "ADP", "CB", "USB", "MMC", "ELV", "CI", "SO", "PGR", "TGT", "DUK", "BDX", "CL", "FISV", "REGN", "SHW", "ITW", "ICE", "GM", "FDX", "AON", "APD", "NSC", "EW", "PSA", "VRTX", "EMR", "ETN", "AIG", "HUM", "CME", "EOG", "CSX", "MCO", "AEP", "KMB", "D", "ORLY", "ADSK", "MCK", "SRE", "TRV", "IDXX", "ROST", "CMG", "MAR", "F"}
                    # 1. Exact symbol match on US exchange
                    for item in data["data"]:
                        symbol = item.get("symbol", "")
                        exchange = item.get("exchange", "")
                        if symbol.upper() == query_upper and exchange in ("NASDAQ", "NYSE"):
                            return symbol
                    # 2. Well-known US ticker on US exchange
                    for item in data["data"]:
                        symbol = item.get("symbol", "")
                        exchange = item.get("exchange", "")
                        if symbol.upper() in us_tickers and exchange in ("NASDAQ", "NYSE"):
                            return symbol
                    # 3. Fallback: nothing found
                    return None
        except Exception:
            pass
    return None
from datetime import datetime, timedelta
import requests
import time
import json
from typing import Optional, Dict, List, Tuple

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Investment Portfolio Tracker",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Path Configuration
# ---------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

DB_PATH = project_root / "data" / "database" / "finance_data.db"
PIPELINE_LOG = project_root / "data" / "logs" / "pipeline.log"

# ---------------------------
# API Configuration
# ---------------------------
API_PORT = 5000
API_URL = f"http://localhost:{API_PORT}"

# n8n Webhooks
N8N_BASE_URL = "https://n8n.datastagke.com"
N8N_AI_ADVISOR_URL = f"{N8N_BASE_URL}/webhook/ai_asset_advisor"
N8N_ASSET_LOOKUP_URL = f"{N8N_BASE_URL}/webhook/asset_lookup"
N8N_ASSET_INGEST_URL = f"{N8N_BASE_URL}/webhook/asset_ingest"

# External APIs
COINGECKO_API = "https://api.coingecko.com/api/v3"
TWELVEDATA_API_KEY = "c8b06cafbbdd4f5aa28e912167284452"
TWELVEDATA_API = "https://api.twelvedata.com"
NEWSAPI_KEY = "c9389cd7ed544a5eb95cde877f283b57"

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
    /* Increase KPI label font size */
    div[data-testid="stMetricLabel"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    /* Reduce spacing after metrics */
    div[data-testid="stMetric"] {
        margin-bottom: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Database Utilities with Retry Logic
# ---------------------------
def execute_with_retry(func, max_retries=3, retry_delay=0.5):
    """Execute function with retry logic"""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    
    # If we get here, all retries failed
    if last_error is not None:
        error_msg = f"Database operation failed after {max_retries} attempts: {str(last_error)}"
        st.error(f"❌ {error_msg}")
        st.exception(last_error)
        raise last_error
    else:
        error_msg = f"Database operation failed after {max_retries} attempts"
        st.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

def run_query(sql: str, params=None, use_fallback=True) -> pd.DataFrame:
    """
    Run SQL query with retry logic and Flask API fallback
    
    Args:
        sql: SQL query string
        params: Query parameters
        use_fallback: Whether to use Flask API as fallback
    
    Returns:
        DataFrame with results
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            df = pd.read_sql_query(sql, conn, params=params or [])
            return df
        finally:
            conn.close()
    except Exception as e:
        st.error(f"❌ Database query failed: {str(e)}")
        st.code(sql)
        if params:
            st.write(f"Parameters: {params}")
        st.exception(e)
        
        if use_fallback:
            st.warning("⚠️ Attempting API fallback...")
            return pd.DataFrame()
        raise

def execute_query(sql: str, params=None):
    """Execute SQL without returning data (with retry)"""
    def exec_db():
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params or [])
            conn.commit()
        finally:
            conn.close()
    
    execute_with_retry(exec_db)

# ---------------------------
# Data Fetching Functions
# ---------------------------
@st.cache_data(ttl=60)
def get_portfolio_holdings() -> pd.DataFrame:
    """Get all portfolio holdings from database"""
    sql = """
        SELECT 
            id, asset_symbol, asset_type, quantity_owned, purchase_price,
            purchase_date, total_cost, fees, notes, created_at, updated_at
        FROM portfolio_holdings
        ORDER BY purchase_date DESC
    """
    return run_query(sql)

@st.cache_data(ttl=30)
def get_latest_crypto_price(asset: str) -> Optional[float]:
    """Get latest crypto price with retry logic"""
    sql = """
        SELECT price FROM crypto_prices
        WHERE LOWER(asset) = LOWER(?)
        ORDER BY timestamp DESC LIMIT 1
    """
    df = run_query(sql, params=[asset])
    return df['price'].iloc[0] if not df.empty else None

@st.cache_data(ttl=30)
def get_latest_stock_price(symbol: str) -> Optional[float]:
    """Get latest stock price with retry logic"""
    sql = """
        SELECT close FROM stock_prices
        WHERE UPPER(symbol) = UPPER(?)
        ORDER BY timestamp DESC LIMIT 1
    """
    df = run_query(sql, params=[symbol])
    return df['close'].iloc[0] if not df.empty else None

def fetch_current_price_for_holding(asset_symbol: str, asset_type: str) -> Optional[float]:
    """Fetch current live price from external APIs (CoinGecko/TwelveData) for new holdings"""
    
    # CoinGecko symbol to ID mapping for common cryptos
    COINGECKO_SYMBOL_MAP = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin', 'SOL': 'solana',
        'ADA': 'cardano', 'XRP': 'ripple', 'DOT': 'polkadot', 'DOGE': 'dogecoin',
        'AVAX': 'avalanche-2', 'MATIC': 'matic-network', 'LINK': 'chainlink',
        'UNI': 'uniswap', 'LTC': 'litecoin', 'ATOM': 'cosmos', 'XLM': 'stellar',
        'ALGO': 'algorand', 'VET': 'vechain', 'ICP': 'internet-computer',
        'FIL': 'filecoin', 'TRX': 'tron'
    }
    
    try:
        if asset_type == "crypto":
            # Map symbol to CoinGecko ID
            coingecko_id = COINGECKO_SYMBOL_MAP.get(asset_symbol.upper(), asset_symbol.lower())
            
            # Call CoinGecko API directly
            url = f"{COINGECKO_API}/simple/price"
            params = {'ids': coingecko_id, 'vs_currencies': 'usd'}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if coingecko_id in data:
                    price = data[coingecko_id].get('usd')
                    if price:
                        return float(price)
                st.error(f"❌ Asset '{asset_symbol}' not found on CoinGecko. Try full name (e.g., 'binancecoin' for BNB).")
            else:
                st.error(f"❌ CoinGecko API error (Status {response.status_code})")
        
        else:  # stock
            # Call TwelveData API directly
            url = f"{TWELVEDATA_API}/price"
            params = {'symbol': asset_symbol.upper(), 'apikey': TWELVEDATA_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    return float(data['price'])
                elif 'message' in data:
                    st.error(f"❌ TwelveData: {data['message']}")
                else:
                    st.error(f"❌ Stock '{asset_symbol}' not found. Verify the symbol.")
            else:
                st.error(f"❌ TwelveData API error (Status {response.status_code})")
    
    except requests.exceptions.Timeout:
        st.error("❌ API request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("❌ Network connection error. Check your internet.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
    
    return None

def get_crypto_history(asset: str, days: int = 30) -> pd.DataFrame:
    """Get crypto price history"""
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    sql = """
        SELECT timestamp as date, price
        FROM crypto_prices
        WHERE LOWER(asset) = LOWER(?) AND timestamp >= ?
        ORDER BY timestamp ASC
    """
    return run_query(sql, params=[asset, cutoff_date])

def get_stock_history(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get stock price history"""
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    sql = """
        SELECT timestamp as date, close as price
        FROM stock_prices
        WHERE UPPER(symbol) = UPPER(?) AND timestamp >= ?
        ORDER BY timestamp ASC
    """
    return run_query(sql, params=[symbol, cutoff_date])

# ---------------------------
# Portfolio Analytics Functions
# ---------------------------
def calculate_portfolio_value(holdings_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive portfolio metrics"""
    if holdings_df.empty:
        return {
            'total_value': 0,
            'total_cost': 0,
            'total_gain_loss': 0,
            'total_gain_loss_pct': 0,
            'holdings': []
        }
    
    holdings_data = []
    total_current_value = 0
    total_cost = holdings_df['total_cost'].sum()
    
    for _, holding in holdings_df.iterrows():
        # Get current price
        if holding['asset_type'] == 'crypto':
            current_price = get_latest_crypto_price(holding['asset_symbol'])
        else:
            current_price = get_latest_stock_price(holding['asset_symbol'])
        
        if current_price:
            current_value = holding['quantity_owned'] * current_price
            gain_loss = current_value - holding['total_cost']
            gain_loss_pct = (gain_loss / holding['total_cost']) * 100 if holding['total_cost'] > 0 else 0
            
            holdings_data.append({
                'symbol': holding['asset_symbol'],
                'type': holding['asset_type'],
                'quantity': holding['quantity_owned'],
                'purchase_price': holding['purchase_price'],
                'current_price': current_price,
                'total_cost': holding['total_cost'],
                'current_value': current_value,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct,
                'allocation_pct': 0  # Will calculate after total
            })
            
            total_current_value += current_value
    
    # Calculate allocation percentages
    for h in holdings_data:
        h['allocation_pct'] = (h['current_value'] / total_current_value * 100) if total_current_value > 0 else 0
    
    total_gain_loss = total_current_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'total_value': total_current_value,
        'total_cost': total_cost,
        'total_gain_loss': total_gain_loss,
        'total_gain_loss_pct': total_gain_loss_pct,
        'holdings': holdings_data
    }

def calculate_portfolio_history(holdings_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Calculate historical portfolio value for performance graph"""
    if holdings_df.empty:
        return pd.DataFrame()
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    portfolio_values = []
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        daily_value = 0
        
        for _, holding in holdings_df.iterrows():
            # Get historical price for this date
            if holding['asset_type'] == 'crypto':
                sql = """
                    SELECT price FROM crypto_prices
                    WHERE LOWER(asset) = LOWER(?)
                    AND DATE(timestamp) = ?
                    ORDER BY timestamp DESC LIMIT 1
                """
                df = run_query(sql, params=[holding['asset_symbol'], date_str])
            else:
                sql = """
                    SELECT close as price FROM stock_prices
                    WHERE UPPER(symbol) = UPPER(?)
                    AND DATE(timestamp) = ?
                    ORDER BY timestamp DESC LIMIT 1
                """
                df = run_query(sql, params=[holding['asset_symbol'], date_str])
            
            if not df.empty:
                price = df['price'].iloc[0]
                daily_value += holding['quantity_owned'] * price
        
        portfolio_values.append({
            'date': date,
            'value': daily_value
        })
    
    return pd.DataFrame(portfolio_values)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe Ratio"""
    if len(returns) < 2:
        return 0
    excess_returns = returns - (risk_free_rate / 252)
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0

def calculate_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility"""
    if len(returns) < 2:
        return 0
    return returns.std() * np.sqrt(252) * 100

def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate maximum drawdown percentage"""
    if len(portfolio_values) < 2:
        return 0
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax * 100
    return drawdown.min()

# ---------------------------
# API Integration Functions
# ---------------------------
def add_holding_via_n8n(asset_data: Dict) -> Tuple[bool, str]:
    """Add holding through n8n workflow -> ETL server (replaces direct Flask POST)"""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Prepare payload for n8n workflow
        payload = {
            'data_type': 'holding',
            'data': [asset_data]  # n8n expects array format
        }
        # Send to n8n webhook which will forward to ETL server
        response = requests.post(
            N8N_ASSET_INGEST_URL,
            json=payload,
            timeout=15,
            verify=False
        )
        if response.status_code in [200, 201]:
            result = response.json()
            # Standardize: if result is a list, convert to dict
            if isinstance(result, list):
                # Try to find a dict with 'success' or 'error' keys
                for item in result:
                    if isinstance(item, dict) and ('success' in item or 'error' in item):
                        result = item
                        break
                else:
                    # If no suitable dict found, wrap as error
                    return False, f"n8n returned a list: {result}"
            if isinstance(result, dict):
                if result.get('success'):
                    return True, "Holding added successfully via n8n workflow!"
                else:
                    return False, result.get('error', 'Unknown error from n8n')
            else:
                return False, f"n8n returned unexpected response: {result}"
        else:
            return False, f"n8n workflow error: {response.status_code}"
    except Exception as e:
        return False, f"Failed to trigger n8n workflow: {str(e)}"

def delete_holding_via_api(holding_id: int) -> Tuple[bool, str]:
    """Delete holding through Flask API"""
    try:
        response = requests.delete(
            f"{API_URL}/api/portfolio/holdings/{holding_id}",
            timeout=10
        )
        if response.status_code == 200:
            return True, "Holding deleted successfully!"
        else:
            return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"Failed to delete holding: {str(e)}"

def trigger_etl_workflow(asset_symbol: str, asset_type: str) -> Dict:
    """
    Trigger ETL workflow for new asset using n8n webhooks
    Steps: Asset lookup → Registration → Historical data ingestion
    """
    workflow_status = {
        'lookup': {'status': 'pending', 'message': ''},
        'register': {'status': 'pending', 'message': ''},
        'ingest': {'status': 'pending', 'message': ''}
    }
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Step 1: Asset Lookup via n8n workflow
    with st.spinner('🔍 Step 1/3: Looking up asset via n8n...'):
        try:
            lookup_payload = {
                'query': asset_symbol,
                'asset_type': asset_type
            }
            response = requests.post(N8N_ASSET_LOOKUP_URL, json=lookup_payload, timeout=15, verify=False)
            
            if response.status_code in [200, 201]:
                lookup_result = response.json()
                workflow_status['lookup'] = {'status': 'success', 'message': f'Found: {lookup_result.get("name", asset_symbol)}'}
            else:
                workflow_status['lookup'] = {'status': 'warning', 'message': f'Lookup returned {response.status_code}, continuing...'}
        except Exception as e:
            workflow_status['lookup'] = {'status': 'warning', 'message': f'Lookup failed: {str(e)[:50]}, continuing...'}
    
    # Step 2: Check if already registered
    with st.spinner('📝 Step 2/3: Checking registration status...'):
        sql = "SELECT id FROM registered_assets WHERE LOWER(symbol) = LOWER(?)"
        df = run_query(sql, params=[asset_symbol])
        
        if not df.empty:
            workflow_status['register'] = {'status': 'skipped', 'message': 'Asset already registered'}
        else:
            try:
                # Use INSERT OR IGNORE to handle race conditions
                insert_sql = """
                    INSERT OR IGNORE INTO registered_assets (symbol, asset_type, source)
                    VALUES (?, ?, ?)
                """
                execute_query(insert_sql, params=[asset_symbol, asset_type, 'n8n_workflow'])
                workflow_status['register'] = {'status': 'success', 'message': 'Asset registered successfully'}
            except Exception as e:
                # If still fails, it's already there - that's okay
                if 'UNIQUE constraint' in str(e):
                    workflow_status['register'] = {'status': 'skipped', 'message': 'Asset already exists'}
                else:
                    workflow_status['register'] = {'status': 'error', 'message': str(e)}
                    return workflow_status
    
    # Step 3: Trigger data ingestion via n8n workflow
    with st.spinner('📊 Step 3/3: Triggering data ingestion via n8n...'):
        try:
            ingest_payload = {
                'symbol': asset_symbol,
                'asset_type': asset_type,
                'days': 30
            }
            response = requests.post(N8N_ASSET_INGEST_URL, json=ingest_payload, timeout=30, verify=False)
            
            if response.status_code in [200, 201]:
                ingest_result = response.json()
                records_added = ingest_result.get('records_added', 'unknown')
                workflow_status['ingest'] = {'status': 'success', 'message': f'Ingested {records_added} records via n8n'}
            else:
                # Fallback to direct API fetch if n8n unavailable
                workflow_status['ingest'] = {'status': 'warning', 'message': f'n8n returned {response.status_code}, using direct fetch...'}
                
                if asset_type == 'crypto':
                    url = f"{COINGECKO_API}/coins/{asset_symbol.lower()}/market_chart"
                    params = {'vs_currency': 'usd', 'days': 30}
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        prices = data.get('prices', [])
                        
                        for timestamp_ms, price in prices:
                            timestamp = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
                            insert_sql = "INSERT OR IGNORE INTO crypto_prices (asset, price, timestamp) VALUES (?, ?, ?)"
                            execute_query(insert_sql, params=[asset_symbol, price, timestamp])
                        
                        workflow_status['ingest'] = {'status': 'success', 'message': f'{len(prices)} records added (fallback)'}
                else:
                    url = f"{TWELVEDATA_API}/time_series"
                    params = {'symbol': asset_symbol, 'interval': '1day', 'outputsize': 30, 'apikey': TWELVEDATA_API_KEY}
                    response = requests.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        values = data.get('values', [])
                        
                        for record in values:
                            insert_sql = """
                                INSERT OR IGNORE INTO stock_prices 
                                (symbol, open, high, low, close, volume, timestamp)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """
                            execute_query(insert_sql, params=[
                                asset_symbol, float(record['open']), float(record['high']),
                                float(record['low']), float(record['close']), int(record['volume']),
                                record['datetime']
                            ])
                        
                        workflow_status['ingest'] = {'status': 'success', 'message': f'{len(values)} records added (fallback)'}
        
        except Exception as e:
            workflow_status['ingest'] = {'status': 'error', 'message': str(e)}
    
    return workflow_status

def get_ai_recommendation(asset_symbol: str, asset_type: str, current_price: float, 
                         quantity: float, total_cost: float) -> Optional[str]:
    """Get AI recommendation from n8n workflow"""
    try:
        payload = {
            'asset_symbol': asset_symbol,
            'asset_type': asset_type,
            'current_price': current_price,
            'quantity': quantity,
            'total_cost': total_cost,
            'timestamp': datetime.now().isoformat()
        }
        # Disable SSL verification for ngrok webhooks
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.post(N8N_AI_ADVISOR_URL, json=payload, timeout=30, verify=False)
        if response.status_code == 200:
            result = response.json()
            # Try to extract the main AI output (text/answer) from the webhook response
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, str) and v.strip():
                        return v
                return 'No AI output found.'
            elif isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    for v in first.values():
                        if isinstance(v, str) and v.strip():
                            return v
                    return 'No AI output found.'
                elif isinstance(first, str) and first.strip():
                    return first
                else:
                    return 'No AI output found.'
            elif isinstance(result, str) and result.strip():
                return result
            else:
                return 'No AI output found.'
        else:
            st.warning(f"⚠️ n8n returned status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        st.error(f"❌ AI Advisor error: {str(e)}")
    # Fallback: run ai_portfolio_advisor.py and return its output
    try:
        import subprocess
        import sys
        script_path = str(project_root / "ai_portfolio_advisor.py")
        result = subprocess.run([sys.executable, script_path, asset_symbol, asset_type, str(current_price), str(quantity), str(total_cost)], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            return "No AI output available (n8n and backup failed)."
    except Exception as e:
        return f"No AI output available (n8n and backup failed): {str(e)}"

def get_portfolio_recommendations(portfolio_summary: Dict, holdings: List[Dict]) -> List[Dict]:
    """
    Generate portfolio-level recommendations using fundamental analysis
    Rules: Don't recommend buying assets already held unless suggesting sell/hold
    """
    try:
        # Try n8n AI Advisor for portfolio-level recommendations
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        import requests
        import json
        payload = {
            'portfolio_summary': portfolio_summary,
            'holdings': holdings
        }
        N8N_PORTFOLIO_ADVISOR_URL = N8N_AI_ADVISOR_URL  # Use same endpoint for now
        response = requests.post(N8N_PORTFOLIO_ADVISOR_URL, json=payload, timeout=30, verify=False)
        if response.status_code == 200:
            result = response.json()
            # Try to extract the main AI output (text/answer) from the webhook response
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, str) and v.strip():
                        return [{'type': 'ai', 'title': 'AI Portfolio Recommendation', 'message': v}]
            elif isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    for v in first.values():
                        if isinstance(v, str) and v.strip():
                            return [{'type': 'ai', 'title': 'AI Portfolio Recommendation', 'message': v}]
                elif isinstance(first, str) and first.strip():
                    return [{'type': 'ai', 'title': 'AI Portfolio Recommendation', 'message': first}]
            elif isinstance(result, str) and result.strip():
                return [{'type': 'ai', 'title': 'AI Portfolio Recommendation', 'message': result}]
    except Exception as e:
        st.warning(f"AI Portfolio Advisor (n8n) failed: {str(e)}")
    # Fallback: run ai_portfolio_advisor.py and return its output
    try:
        import subprocess
        import sys
        script_path = str(project_root / "ai_portfolio_advisor.py")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return [{'type': 'backup', 'title': 'Backup Portfolio Recommendation', 'message': result.stdout.strip()}]
        else:
            return [{'type': 'error', 'title': 'No AI Output', 'message': 'No AI output available (n8n and backup failed).'}]
    except Exception as e:
        return [{'type': 'error', 'title': 'No AI Output', 'message': f'No AI output available (n8n and backup failed): {str(e)}'}]

# ---------------------------
# Visualization Functions
# ---------------------------
def plot_portfolio_performance(history_df: pd.DataFrame, days: int):
    """Create interactive portfolio performance chart with zoom"""
    if history_df.empty:
        st.info("No historical data available yet.")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df['date'],
        y=history_df['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title=f'Portfolio Performance - Last {days} Days',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=500,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        ),
        yaxis=dict(
            fixedrange=False
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_allocation_pie(holdings: List[Dict]):
    """Plot portfolio allocation pie chart"""
    if not holdings:
        return
    
    df = pd.DataFrame(holdings)
    
    fig = px.pie(
        df,
        values='current_value',
        names='symbol',
        title='Portfolio Allocation',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)  # Match other chart sizes
    st.plotly_chart(fig, use_container_width=True)

def plot_gain_loss_chart(holdings: List[Dict]):
    """Plot gain/loss bar chart for each holding"""
    if not holdings:
        return
    
    df = pd.DataFrame(holdings)
    df['color'] = df['gain_loss'].apply(lambda x: 'green' if x >= 0 else 'red')
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['symbol'],
            y=df['gain_loss'],
            marker_color=df['color'],
            text=df['gain_loss'].apply(lambda x: f'${x:.2f}'),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Gain/Loss by Asset',
        xaxis_title='Asset',
        yaxis_title='Gain/Loss ($)',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_asset_history(asset_symbol: str, asset_type: str, days: int, unique_id=None):
    """Plot individual asset price history"""
    if asset_type == 'crypto':
        df = get_crypto_history(asset_symbol, days)
    else:
        df = get_stock_history(asset_symbol, days)
    
    if df.empty:
        st.warning(f"No historical data available for {asset_symbol}")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name=asset_symbol,
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'{asset_symbol} Price - Last {days} Days',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=300,
        hovermode='x'
    )
    
    key = f"{asset_symbol}_{asset_type}_plot"
    if unique_id is not None:
        key += f"_{unique_id}"
    st.plotly_chart(fig, use_container_width=True, key=key)

# ---------------------------
# Platform Status Functions
# ---------------------------
def check_platform_status() -> Dict:
    """Check status of all platform components"""
    status = {
        'database': {'status': 'unknown', 'message': ''},
        'flask_api': {'status': 'unknown', 'message': ''},
        'coingecko': {'status': 'unknown', 'message': ''},
        'twelvedata': {'status': 'unknown', 'message': ''},
        'n8n': {'status': 'unknown', 'message': ''}
    }
    
    # Check Database
    try:
        df = run_query("SELECT COUNT(*) as count FROM portfolio_holdings", use_fallback=False)
        status['database'] = {'status': 'healthy', 'message': f'{df["count"].iloc[0]} holdings'}
    except Exception as e:
        status['database'] = {'status': 'error', 'message': str(e)}
    
    # Check Flask API
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            status['flask_api'] = {'status': 'healthy', 'message': 'Responding'}
        else:
            status['flask_api'] = {'status': 'error', 'message': f'Status {response.status_code}'}
    except:
        status['flask_api'] = {'status': 'error', 'message': 'Not responding'}
    
    # Check CoinGecko
    try:
        response = requests.get(f"{COINGECKO_API}/ping", timeout=3)
        if response.status_code == 200:
            status['coingecko'] = {'status': 'healthy', 'message': 'API active'}
        else:
            status['coingecko'] = {'status': 'error', 'message': 'API error'}
    except:
        status['coingecko'] = {'status': 'error', 'message': 'Not reachable'}
    
    # Check TwelveData
    try:
        url = f"{TWELVEDATA_API}/time_series"
        params = {'symbol': 'AAPL', 'interval': '1day', 'outputsize': 1, 'apikey': TWELVEDATA_API_KEY}
        response = requests.get(url, params=params, timeout=3)
        if response.status_code == 200:
            status['twelvedata'] = {'status': 'healthy', 'message': 'API active'}
        else:
            status['twelvedata'] = {'status': 'error', 'message': 'API error'}
    except:
        status['twelvedata'] = {'status': 'error', 'message': 'Not reachable'}
    
    # Check n8n workflows
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response = requests.get(N8N_BASE_URL, timeout=3, verify=False)
        if response.status_code in [200, 404]:  # 404 is ok, means server is up
            status['n8n'] = {'status': 'healthy', 'message': '3 workflows ready (AI, Lookup, Ingest)'}
        else:
            status['n8n'] = {'status': 'error', 'message': f'Status {response.status_code}'}
    except:
        status['n8n'] = {'status': 'error', 'message': 'Not reachable'}
    
    return status

def get_last_data_update() -> Dict:
    """Get timestamps of last data updates"""
    updates = {}
    
    # Last crypto price update
    sql = "SELECT MAX(timestamp) as last_update FROM crypto_prices"
    df = run_query(sql)
    updates['crypto'] = df['last_update'].iloc[0] if not df.empty else 'Never'
    
    # Last stock price update
    sql = "SELECT MAX(timestamp) as last_update FROM stock_prices"
    df = run_query(sql)
    updates['stock'] = df['last_update'].iloc[0] if not df.empty else 'Never'
    
    return updates

def fetch_market_news(max_results: int = 5) -> List[Dict]:
    """Fetch latest market news from NewsAPI"""
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            'apiKey': NEWSAPI_KEY,
            'category': 'business',
            'language': 'en',
            'pageSize': max_results
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('articles', [])
        else:
            return []
    except Exception as e:
        st.error(f"Failed to fetch news: {str(e)}")
        return []

# ---------------------------
# Main Dashboard Pages
# ---------------------------
def page_overview():
    """Main overview page with portfolio summary and performance"""
    st.markdown('<h1 style="font-size:4rem; font-weight:bold; color:#1f77b4; text-align:center; margin-bottom:1.5rem;">Investment Portfolio Dashboard</h1>', unsafe_allow_html=True)
    
    # Get portfolio data
    holdings_df = get_portfolio_holdings()
    
    if holdings_df.empty:
        st.info("👋 Welcome! Add your first holding to get started.")
        return
    
    # Calculate portfolio metrics
    portfolio = calculate_portfolio_value(holdings_df)
    
    # Main content area and news column (2:1 ratio so news gets 1/3 of width)
    main_col, news_col = st.columns([2, 1])
    
    with main_col:
        # Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Value",
                value=f"${portfolio['total_value']:,.2f}",
                delta=f"${portfolio['total_gain_loss']:,.2f}"
            )
        
        with col2:
            st.metric(
                label="Total Invested",
                value=f"${portfolio['total_cost']:,.2f}"
            )
        
        with col3:
            delta_color = "normal" if portfolio['total_gain_loss'] >= 0 else "inverse"
            st.metric(
                label="Profit/Loss",
                value=f"${portfolio['total_gain_loss']:,.2f}",
                delta=f"{portfolio['total_gain_loss_pct']:.2f}%"
            )
        
        with col4:
            st.metric(
                label="Assets",
                value=len(holdings_df)
            )
    
        # Portfolio Performance Graph (inside main_col to be next to news)
        st.subheader("Portfolio Performance")
        
        # Time range selector
        col1, col2 = st.columns([3, 1])
        with col2:
            time_range = st.selectbox(
                "Time Range",
                options=[7, 30, 90, 180, 365],
                index=1,  # Default to 30 days
                format_func=lambda x: f"{x} days"
            )
        
        history_df = calculate_portfolio_history(holdings_df, days=time_range)
        plot_portfolio_performance(history_df, time_range)
    
    with news_col:
        st.markdown('<h3 style="text-align: center; color:#1f77b4; font-size:1.5rem; font-weight:700; margin-bottom:10px;">📰 Market News</h3>', unsafe_allow_html=True)
        
        with st.spinner("Loading news..."):
            news_articles = fetch_market_news(max_results=10)
        
        if news_articles:
            # Create a single bordered, scrollable container
            st.markdown("""
            <style>
            .news-box {
                border: 2px solid rgba(31, 119, 180, 0.3);
                border-radius: 12px;
                padding: 20px;
                height: 700px;
                overflow-y: auto;
                background-color: rgba(255, 255, 255, 0.05);
                margin-bottom: 20px;
            }
            .news-box img {
                max-width: 100%;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Build all news content as HTML
            news_html = '<div class="news-box">'
            
            for i, article in enumerate(news_articles):
                title = article.get('title', 'No title')
                description = article.get('description', '')
                url = article.get('url', '#')
                image_url = article.get('urlToImage')
                published = article.get('publishedAt', '')
                
                # Add image if available
                if image_url:
                    news_html += f'<img src="{image_url}" alt="News image">'
                
                # Add title
                news_html += f'<strong>{title}</strong><br>'
                
                # Add description
                if description:
                    desc_text = description[:150] + "..." if len(description) > 150 else description
                    news_html += f'<p style="color: #888; font-size: 0.9em; margin: 5px 0;">{desc_text}</p>'
                
                # Add link
                news_html += f'<a href="{url}" target="_blank" style="color: #1f77b4;">Read more →</a><br>'
                
                # Add date
                if published:
                    try:
                        pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                        date_str = pub_date.strftime('%b %d, %Y')
                        news_html += f'<p style="color: #666; font-size: 0.85em; margin: 5px 0;">📅 {date_str}</p>'
                    except:
                        pass
                
                # Add divider between articles
                if i < len(news_articles) - 1:
                    news_html += '<hr style="margin: 15px 0; border: 0; border-top: 1px solid #ddd;">'
            
            news_html += '</div>'
            
            # Render the entire news box as one HTML block
            st.markdown(news_html, unsafe_allow_html=True)
        else:
            st.info("No news available at the moment.")
    
    # AI Recommendations (below performance chart and news)
    st.divider()
    st.subheader("AI Portfolio Recommendations")
    
    # AI Disclaimer
    st.caption("⚠️ **Disclaimer:** These recommendations are generated by AI analysis and should not be considered financial advice. Always do your own research and consult with a licensed financial advisor before making investment decisions.")
    
    # Fetch recommendations from Flask API
    try:
        response = requests.get(f"{API_URL}/api/portfolio/recommendations", timeout=10)
        if response.status_code == 200:
            result = response.json()
            recommendations = result.get('data', [])
            
            if recommendations:
                # Initialize state
                if 'rec_index' not in st.session_state:
                    st.session_state.rec_index = 0
                
                # Display current recommendation
                rec = recommendations[st.session_state.rec_index]
                
                # Show recommendation
                if rec['type'] == 'success':
                    st.success(f"**{rec['title']}**\n\n{rec['message']}")
                elif rec['type'] == 'warning':
                    st.warning(f"**{rec['title']}**\n\n{rec['message']}")
                elif rec['type'] == 'error':
                    st.error(f"**{rec['title']}**\n\n{rec['message']}")
                else:
                    st.info(f"**{rec['title']}**\n\n{rec['message']}")
                
                # Navigation buttons below recommendation
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                
                with nav_col1:
                    if st.button("◀ Previous", key="rec_prev", use_container_width=True):
                        st.session_state.rec_index = (st.session_state.rec_index - 1) % len(recommendations)
                        st.rerun()
                
                with nav_col2:
                    st.markdown(f"<p style='text-align: center; padding-top: 8px;'>{st.session_state.rec_index + 1} of {len(recommendations)}</p>", unsafe_allow_html=True)
                
                with nav_col3:
                    if st.button("Next ▶", key="rec_next", use_container_width=True):
                        st.session_state.rec_index = (st.session_state.rec_index + 1) % len(recommendations)
                        st.rerun()
            else:
                st.info("✅ Your portfolio looks well-balanced!")
        else:
            st.warning("Unable to load AI recommendations. Flask API may be offline.")
    except requests.exceptions.RequestException:
        st.warning("Unable to connect to recommendation service. Please ensure Flask API is running.")
    
    st.divider()
    
    # Calculate advanced metrics if we have history
    if not history_df.empty and len(history_df) > 1:
        st.subheader("Advanced Analytics")
        
        returns = history_df['value'].pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)
        volatility = calculate_volatility(returns)
        max_dd = calculate_max_drawdown(history_df['value'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.caption("Risk-adjusted returns (>1 is good)")
        
        with col2:
            st.metric("Volatility", f"{volatility:.2f}%")
            st.caption("Annualized price variation")
        
        with col3:
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
            st.caption("Largest peak-to-trough decline")
    
    st.divider()
    
    # Holdings Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Allocation")
        plot_allocation_pie(portfolio['holdings'])
    
    with col2:
        st.subheader("Gain/Loss by Asset")
        plot_gain_loss_chart(portfolio['holdings'])
    
    st.divider()

def page_holdings():
    """Holdings management page with CRUD operations"""
    st.header("📋 Manage Holdings")
    
    # Get current holdings
    holdings_df = get_portfolio_holdings()
    portfolio = calculate_portfolio_value(holdings_df)
    
    # Holdings Display
    if not holdings_df.empty:
        # Individual Holdings Details with AI Recommendations
        st.subheader("📊 Holdings Details & AI Insights")
        for idx, holding in enumerate(portfolio['holdings']):
            # Expand first 3 holdings, fold the rest
            with st.expander(f"{holding['symbol']} - {holding['type'].title()}", expanded=(idx < 3)):
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Price history chart
                    plot_asset_history(holding['symbol'], holding['type'], 30, idx)
                    # Metrics
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("Quantity", f"{holding['quantity']:.4f}")
                    with mcol2:
                        st.metric("Current Value", f"${holding['current_value']:.2f}")
                    with mcol3:
                        color_class = "success-card" if holding['gain_loss'] >= 0 else "warning-card"
                        st.markdown(f'<div class="{color_class}">', unsafe_allow_html=True)
                        st.metric("Gain/Loss", f"${holding['gain_loss']:.2f}", 
                                f"{holding['gain_loss_pct']:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("### 🤖 AI Recommendation")
                    if st.button(f"Get AI Advice for {holding['symbol']}", key=f"ai_{idx}"):
                        with st.spinner("Consulting AI advisor..."):
                            recommendation = get_ai_recommendation(
                                holding['symbol'],
                                holding['type'],
                                holding['current_price'],
                                holding['quantity'],
                                holding['total_cost']
                            )
                            if recommendation:
                                st.info(recommendation)
                            else:
                                st.warning("⚠️ AI Advisor unavailable. Please try again later.")
                    # Delete button
                    st.markdown("---")
                    if st.button(f"🗑️ Delete {holding['symbol']}", key=f"del_{idx}", type="secondary"):
                        # Find holding ID from original dataframe
                        holding_id = holdings_df[holdings_df['asset_symbol'] == holding['symbol']]['id'].iloc[0]
                        success, message = delete_holding_via_api(holding_id)
                        if success:
                            st.toast(f"✅ {message}", icon="✅")
                            # Clear all caches
                            get_portfolio_holdings.clear()
                            get_latest_crypto_price.clear()
                            get_latest_stock_price.clear()
                            st.rerun()
                        else:
                            st.error(message)
        st.divider()
        # Current Holdings Table (moved below details)
        st.subheader("Current Holdings Table")
        # Create display dataframe with current values
        display_data = []
        for holding in portfolio['holdings']:
            display_data.append({
                'Symbol': holding['symbol'],
                'Type': holding['type'].title(),
                'Quantity': f"{holding['quantity']:.4f}",
                'Purchase Price': f"${holding['purchase_price']:.2f}",
                'Current Price': f"${holding['current_price']:.2f}",
                'Total Cost': f"${holding['total_cost']:.2f}",
                'Current Value': f"${holding['current_value']:.2f}",
                'Gain/Loss': f"${holding['gain_loss']:.2f}",
                'Gain/Loss %': f"{holding['gain_loss_pct']:.2f}%",
                'Allocation': f"{holding['allocation_pct']:.1f}%"
            })
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.divider()
        
        st.subheader("🔍 Search Any Asset")
        st.caption("⚠️ **Disclaimer:** AI recommendations are for informational purposes only and do not constitute financial advice. Always conduct your own research.")
        
        # Asset Search Form
        with st.form("asset_search_form"):
            scol1, scol2 = st.columns([2, 1])

            with scol1:
                search_query = st.text_input(
                    "Search Asset",
                    placeholder="Enter crypto name or stock symbol (e.g., ethereum, TSLA)",
                    help="Search for any asset worldwide to see current price and AI recommendation",
                    key="search_asset_query"
                )

            with scol2:
                search_type = st.selectbox("Type", options=["crypto", "stock"], key="search_type_search")

            search_button = st.form_submit_button("🔎 Search", type="primary", use_container_width=True, key="search_button")
            
            if search_button and search_query:
                with st.spinner(f"Searching for {search_query}..."):
                    # Always resolve name to symbol/ID
                    resolved_symbol = resolve_asset_symbol(search_query, search_type)
                    display_name = search_query
                    if resolved_symbol:
                        st.info(f"Resolved '{search_query}' to symbol: {resolved_symbol.upper()}")
                    else:
                        resolved_symbol = search_query  # fallback to user input
                    # Fetch price using resolved symbol
                    current_price = fetch_current_price_for_holding(resolved_symbol, search_type)
                    if current_price:
                        st.success(f"**{display_name} ({resolved_symbol.upper()})** current price: **${current_price:,.2f}**")
                        # Get AI recommendation via n8n
                        st.subheader("🤖 AI Recommendation")
                        with st.spinner("Consulting AI advisor..."):
                            recommendation = get_ai_recommendation(
                                resolved_symbol,
                                search_type,
                                current_price,
                                quantity=1,  # Hypothetical 1 unit
                                total_cost=current_price
                            )
                            if recommendation:
                                st.info(recommendation)
                                st.caption("💡 This is an AI-generated suggestion based on current market data. Not financial advice.")
                            else:
                                st.warning("⚠️ AI Advisor unavailable. Please try again later.")
                    else:
                        st.error(f"❌ Could not find price data for {display_name}. Verify the symbol/name is correct.")
        
        st.divider()
        
        st.subheader("➕ Add New Holding")
        
        with st.form("add_holding_form"):
            # Asset selection in form
            col1, col2 = st.columns(2)
            # Asset type and symbol/name input with lookup
            col1, col2 = st.columns(2)
            with col1:
                asset_type = st.selectbox("Asset Type", options=["crypto", "stock"], key="asset_type_form_add")
            with col2:
                asset_symbol_raw = st.text_input(
                    "Asset Symbol or Name", 
                    placeholder="BTC, ETH, AAPL, MSFT, Bitcoin, Apple Inc.", 
                    key="asset_symbol_form_add",
                    help="Enter symbol or asset name. Price and symbol will auto-fetch as you type."
                )
                asset_symbol = asset_symbol_raw.strip().upper() if asset_symbol_raw else ""
                resolved_symbol = resolve_asset_symbol(asset_symbol_raw, asset_type) if asset_symbol_raw else None
                display_name = asset_symbol_raw
                if resolved_symbol:
                    # Prefer US tickers for stocks (already handled in resolve_asset_symbol)
                    st.info(f"Resolved '{asset_symbol_raw}' to symbol: {resolved_symbol.upper()}")
                    asset_symbol = resolved_symbol.upper()

            # Asset lookup: try to resolve symbol using local price fetch only
            # If the user enters a name instead of a symbol, fetch_current_price_for_holding should handle resolution

            # Auto-fetch price when symbol is entered
            fetched_price = None
            if asset_symbol and len(asset_symbol) >= 2:
                current_key = f"{asset_symbol}_{asset_type}"
                last_fetched = st.session_state.get('last_fetched_key', '')
                if last_fetched != current_key:
                    with st.spinner(f"Fetching {asset_type} price for {asset_symbol}..."):
                        current_price = fetch_current_price_for_holding(asset_symbol, asset_type)
                        if current_price:
                            st.session_state['fetched_price'] = current_price
                            st.session_state['last_fetched_key'] = current_key
                            fetched_price = current_price
                            st.toast(f"💰 {display_name} ({asset_symbol}) Price: ${current_price:,.2f}", icon="💰")
                        else:
                            st.toast(f"⚠️ Could not fetch price for {display_name} ({asset_symbol})", icon="⚠️")
                            st.session_state.pop('fetched_price', None)
                elif 'fetched_price' in st.session_state:
                    fetched_price = st.session_state['fetched_price']
                    st.toast(f"💰 {display_name} ({asset_symbol}) Price: ${fetched_price:,.2f}", icon="💰")

            
            # Smart validation hints
            if asset_symbol:
                validation_hints = []
                if len(asset_symbol) > 10:
                    validation_hints.append("⚠️ Symbol too long (max 10 characters)")
                if not asset_symbol.replace('-', '').replace('.', '').isalnum():
                    validation_hints.append("⚠️ Symbol contains invalid characters")
                
                # Suggest asset type based on symbol
                common_cryptos = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC', 'LINK', 'UNI', 'LTC', 'ATOM', 'XLM', 'ALGO', 'VET', 'ICP', 'FIL', 'TRX']
                if asset_type == "stock" and asset_symbol in common_cryptos:
                    st.warning(f"💡 **{asset_symbol}** looks like a crypto. Did you mean to select 'crypto'?")
                elif asset_type == "crypto" and asset_symbol.isalpha() and len(asset_symbol) <= 5 and asset_symbol not in common_cryptos:
                    st.info(f"💡 **{asset_symbol}** might be a stock ticker. Verify Asset Type is correct.")
                
                for hint in validation_hints:
                    st.warning(hint)
            col1, col2 = st.columns(2)
            
            with col1:
                # Quantity is now auto-calculated and not editable
                quantity = None  # Will be set after amount/purchase_price
                purchase_date = st.date_input(
                    "Purchase Date", 
                    value=datetime.now(),
                    max_value=datetime.now(),
                    help="Date of purchase (cannot be future)"
                )
            
            with col2:
                # Always use fetched price as default if available
                default_price = float(fetched_price) if fetched_price is not None else 0.01
                purchase_price = st.number_input(
                    "Purchase Price ($)", 
                    min_value=0.01, 
                    step=0.01, 
                    value=default_price,
                    help="Price per unit at purchase (auto-filled if available)"
                )
                amount = st.number_input(
                    "Amount ($)", 
                    min_value=0.0, 
                    step=0.01, 
                    value=0.0,
                    help="How much you spent in total (auto-calculates quantity)"
                )
                # Quantity is auto-calculated from amount and purchase price
                if amount > 0 and purchase_price > 0:
                    quantity = round(amount / purchase_price, 8)
                else:
                    quantity = 0.0
                st.markdown(f"**Quantity (auto-calculated):** {quantity:.8f}")
                fees = st.number_input(
                    "Fees ($)", 
                    min_value=0.0, 
                    step=0.01, 
                    value=0.0,
                    help="Transaction fees, commissions, etc."
                )
            
            notes = st.text_area("Notes (optional)", placeholder="Strategy, source, or other details...")
            
            # Only one submit button inside the form
            submit_button = st.form_submit_button("Add Holding", type="primary", use_container_width=True)
            
            if submit_button:
                # Comprehensive validation
                errors = []
                # Required fields: asset_symbol, asset_type, quantity, purchase_price, purchase_date
                if not asset_symbol:
                    errors.append("❌ Asset symbol is required")
                elif len(asset_symbol) > 10:
                    errors.append("❌ Asset symbol too long (max 10 characters)")
                elif not asset_symbol.replace('-', '').replace('.', '').isalnum():
                    errors.append("❌ Asset symbol contains invalid characters")
                if not asset_type:
                    errors.append("❌ Asset type is required")
                if quantity is None or quantity <= 0:
                    errors.append("❌ Quantity must be greater than 0")
                if purchase_price is None or purchase_price <= 0:
                    errors.append("❌ Purchase price must be greater than 0")
                if not purchase_date:
                    errors.append("❌ Purchase date is required")
                elif purchase_date > datetime.now().date():
                    errors.append("❌ Purchase date cannot be in the future")
                if fees is not None and fees < 0:
                    errors.append("❌ Fees cannot be negative")
                # Check for duplicate holdings (first form)
                existing_holdings = get_portfolio_holdings()
                if not existing_holdings.empty:
                    duplicate = existing_holdings[
                        (existing_holdings['asset_symbol'].str.upper() == asset_symbol.upper()) & 
                        (existing_holdings['asset_type'] == asset_type)
                    ]
                    if not duplicate.empty:
                        st.toast(f"⚠️ You already have {asset_symbol}. This will add to your position.", icon="⚠️")
                if errors:
                    for error in errors:
                        st.toast(error, icon="❌")
                else:
                    # Auto-fetch current price if not already fetched
                    if purchase_price == 0.01:
                        with st.spinner(f"Fetching current {asset_type} price for {asset_symbol}..."):
                            current_price = fetch_current_price_for_holding(asset_symbol, asset_type)
                            if current_price:
                                purchase_price = current_price
                                st.success(f"💰 Auto-fetched price: ${current_price:,.2f}")
                    # Calculate total cost
                    total_cost = (quantity * purchase_price) + (fees if fees is not None else 0.0)
                    # Prepare holding data
                    holding_data = {
                        'asset_symbol': asset_symbol,
                        'asset_type': asset_type,
                        'quantity_owned': quantity,
                        'purchase_price': purchase_price,
                        'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                        'total_cost': total_cost,
                        'fees': fees if fees is not None else 0.0,
                        'notes': notes
                    }
                    # Only trigger ETL workflow and add holding if Add Holding button was clicked
                    if submit_button:
                        st.toast("🔄 Initiating ETL workflow...", icon="🔄")
                        workflow_result = trigger_etl_workflow(asset_symbol.upper(), asset_type)
                        # Display workflow results
                        for step, result in workflow_result.items():
                            if result['status'] == 'success':
                                st.toast(f"✅ {step.title()}: {result['message']}", icon="✅")
                            elif result['status'] == 'error':
                                st.toast(f"❌ {step.title()}: {result['message']}", icon="❌")
                            elif result['status'] == 'skipped':
                                st.toast(f"ℹ️ {step.title()}: {result['message']}", icon="ℹ️")
                        # Add holding via API
                        success, message = add_holding_via_n8n(holding_data)
                    if success:
                        st.toast(f"✅ {message}", icon="✅")
                        st.toast("🔄 Click the refresh button to see your new holding!", icon="🔄")
                        # Clear caches
                        get_portfolio_holdings.clear()
                        get_latest_crypto_price.clear()
                        get_latest_stock_price.clear()
                    else:
                        st.toast(f"❌ {message}", icon="❌")
    
    else:
        st.toast("👋 No holdings yet. Add your first holding below!", icon="👋")
        st.divider()
        
        st.subheader("➕ Add New Holding")
        
        with st.form("add_holding_form_empty"):
            # Asset selection in form
            col1, col2 = st.columns(2)
            with col1:
                asset_type = st.selectbox("Asset Type", options=["crypto", "stock"], key="asset_type_form_empty")
            with col2:
                asset_symbol_raw = st.text_input(
                    "Asset Symbol", 
                    placeholder="BTC, ETH, AAPL, MSFT", 
                    key="asset_symbol_form_empty",
                    help="Price will auto-fetch as you type"
                )
                asset_symbol = asset_symbol_raw.strip().upper() if asset_symbol_raw else ""
            
            # Auto-fetch price when symbol is entered
            if asset_symbol and len(asset_symbol) >= 2 and asset_symbol.replace('-', '').replace('.', '').isalnum():
                current_key = f"{asset_symbol}_{asset_type}_empty"
                last_fetched = st.session_state.get('last_fetched_key_empty', '')
                
                if last_fetched != current_key:
                    with st.spinner(f"Fetching {asset_type} price for {asset_symbol}..."):
                        current_price = fetch_current_price_for_holding(asset_symbol, asset_type)
                        if current_price:
                            st.session_state['fetched_price_empty'] = current_price
                            st.session_state['last_fetched_key_empty'] = current_key
                            st.toast(f"💰 Current Price: ${current_price:,.2f}", icon="💰")
                        else:
                            st.toast(f"⚠️ Could not fetch price for {asset_symbol}", icon="⚠️")
                            st.session_state.pop('fetched_price_empty', None)
                elif 'fetched_price_empty' in st.session_state:
                    st.toast(f"💰 Current Price: ${st.session_state['fetched_price_empty']:,.2f}", icon="💰")
            
            # Smart validation hints
            if asset_symbol:
                validation_hints = []
                if len(asset_symbol) > 10:
                    validation_hints.append("⚠️ Symbol too long (max 10 characters)")
                if not asset_symbol.replace('-', '').replace('.', '').isalnum():
                    validation_hints.append("⚠️ Symbol contains invalid characters")
                
                # Suggest asset type based on symbol
                common_cryptos = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC', 'LINK', 'UNI', 'LTC', 'ATOM', 'XLM', 'ALGO', 'VET', 'ICP', 'FIL', 'TRX']
                if asset_type == "stock" and asset_symbol in common_cryptos:
                    st.warning(f"💡 **{asset_symbol}** looks like a crypto. Did you mean to select 'crypto'?")
                elif asset_type == "crypto" and asset_symbol.isalpha() and len(asset_symbol) <= 5 and asset_symbol not in common_cryptos:
                    st.info(f"💡 **{asset_symbol}** might be a stock ticker. Verify Asset Type is correct.")
                
                for hint in validation_hints:
                    st.warning(hint)
            col1, col2 = st.columns(2)
            
            with col1:
                quantity = st.number_input(
                    "Quantity", 
                    min_value=0.0001, 
                    step=0.1, 
                    format="%.4f",
                    help="Number of units/shares to add"
                )
                purchase_date = st.date_input(
                    "Purchase Date", 
                    value=datetime.now(),
                    max_value=datetime.now(),
                    help="Date of purchase (cannot be future)"
                )
            
            with col2:
                # Use fetched price if available and matches current selection
                fetched_price = st.session_state.get('fetched_price', 0.01)
                fetched_symbol = st.session_state.get('fetched_symbol', '')
                
                if fetched_symbol == asset_symbol and 'fetched_price' in st.session_state:
                    default_price = float(fetched_price)
                else:
                    default_price = 0.01
                
                purchase_price = st.number_input(
                    "Purchase Price ($)", 
                    min_value=0.01, 
                    step=0.01, 
                    value=default_price,
                    help="Price per unit at purchase"
                )
                fees = st.number_input(
                    "Fees ($)", 
                    min_value=0.0, 
                    step=0.01, 
                    value=0.0,
                    help="Transaction fees, commissions, etc."
                )
            
            notes = st.text_area("Notes (optional)", placeholder="Strategy, source, or other details...")
            
            # Submit and refresh buttons at 1/3 width each
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
            with col_btn1:
                submit_button = st.form_submit_button("Add Holding", type="primary", use_container_width=True)
            with col_btn2:
                clear_button = st.form_submit_button("�� Clear Form", use_container_width=True)
            
            if clear_button:
                # Clear form by clearing session state
                for key in list(st.session_state.keys()):
                    if "fetched" in str(key) or "last_fetched" in str(key):
                        del st.session_state[key]
                st.rerun()
            
            if submit_button:
                # Comprehensive validation
                errors = []
                
                if not asset_symbol:
                    errors.append("❌ Asset symbol is required")
                elif len(asset_symbol) > 10:
                    errors.append("❌ Asset symbol too long (max 10 characters)")
                elif not asset_symbol.replace('-', '').replace('.', '').isalnum():
                    errors.append("❌ Asset symbol contains invalid characters")
                
                if quantity <= 0:
                    errors.append("❌ Quantity must be greater than 0")
                
                if purchase_price <= 0:
                    errors.append("❌ Purchase price must be greater than 0")
                
                if purchase_date > datetime.now().date():
                    errors.append("❌ Purchase date cannot be in the future")
                
                if fees < 0:
                    errors.append("❌ Fees cannot be negative")
                
                if errors:
                    for error in errors:
                        st.toast(error, icon="❌")
                else:
                    # Auto-fetch current price if not already fetched
                    if purchase_price == 0.01:
                        with st.spinner(f"Fetching current {asset_type} price for {asset_symbol}..."):
                            current_price = fetch_current_price_for_holding(asset_symbol, asset_type)
                            if current_price:
                                purchase_price = current_price
                                st.success(f"💰 Auto-fetched price: ${current_price:,.2f}")
                    
                    # Calculate total cost
                    total_cost = (quantity * purchase_price) + fees
                    
                    # Prepare holding data
                    holding_data = {
                        'asset_symbol': asset_symbol,
                        'asset_type': asset_type,
                        'quantity_owned': quantity,
                        'purchase_price': purchase_price,
                        'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                        'total_cost': total_cost,
                        'fees': fees,
                        'notes': notes
                    }
                    
                    # Trigger ETL workflow (async with loading)
                    st.toast("🔄 Initiating ETL workflow...", icon="🔄")
                    workflow_result = trigger_etl_workflow(asset_symbol.upper(), asset_type)
                    
                    # Display workflow results
                    for step, result in workflow_result.items():
                        if result['status'] == 'success':
                            st.toast(f"✅ {step.title()}: {result['message']}", icon="✅")
                        elif result['status'] == 'error':
                            st.toast(f"❌ {step.title()}: {result['message']}", icon="❌")
                        elif result['status'] == 'skipped':
                            st.toast(f"ℹ️ {step.title()}: {result['message']}", icon="ℹ️")
                    
                    # Add holding via API
                    success, message = add_holding_via_n8n(holding_data)
                    
                    if success:
                        st.toast(f"✅ {message}", icon="✅")
                        st.toast("🔄 Click the refresh button to see your new holding!", icon="🔄")
                        # Clear caches
                        get_portfolio_holdings.clear()
                        get_latest_crypto_price.clear()
                        get_latest_stock_price.clear()
                    else:
                        st.toast(f"❌ {message}", icon="❌")

def page_analytics():
    """Advanced analytics page"""
    st.header("📊 Advanced Analytics")
    
    holdings_df = get_portfolio_holdings()
    
    if holdings_df.empty:
        st.info("Add holdings to view analytics.")
        return
    
    portfolio = calculate_portfolio_value(holdings_df)
    
    # Asset Type Breakdown
    st.subheader("🏦 Asset Type Distribution")
    
    crypto_holdings = [h for h in portfolio['holdings'] if h['type'] == 'crypto']
    stock_holdings = [h for h in portfolio['holdings'] if h['type'] == 'stock']
    
    col1, col2 = st.columns(2)
    
    with col1:
        crypto_value = sum(h['current_value'] for h in crypto_holdings)
        stock_value = sum(h['current_value'] for h in stock_holdings)
        
        fig = go.Figure(data=[go.Pie(
            labels=['Cryptocurrency', 'Stocks'],
            values=[crypto_value, stock_value],
            hole=0.3,
            marker_colors=['#f7931a', '#0066cc']
        )])
        fig.update_layout(title="Value by Asset Type", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Statistics")
        st.metric("Crypto Assets", len(crypto_holdings))
        st.metric("Stock Assets", len(stock_holdings))
        st.metric("Total Assets", len(portfolio['holdings']))
        
        if crypto_value + stock_value > 0:
            crypto_pct = crypto_value / (crypto_value + stock_value) * 100
            st.metric("Crypto Allocation", f"{crypto_pct:.1f}%")
    
    st.divider()
    
    # Performance Comparison
    st.subheader("📊 Individual Asset Performance")
    
    performance_data = []
    for h in portfolio['holdings']:
        performance_data.append({
            'Asset': h['symbol'],
            'Type': h['type'],
            'Return %': h['gain_loss_pct'],
            'Value': h['current_value']
        })
    
    perf_df = pd.DataFrame(performance_data).sort_values('Return %', ascending=False)
    
    fig = px.bar(
        perf_df,
        x='Asset',
        y='Return %',
        color='Return %',
        color_continuous_scale=['red', 'yellow', 'green'],
        title='Returns by Asset'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Risk Analysis
    st.subheader("⚠️ Risk Analysis")
    
    # Concentration risk
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Holdings by Value")
        top_holdings = sorted(portfolio['holdings'], key=lambda x: x['current_value'], reverse=True)[:5]
        for h in top_holdings:
            st.write(f"**{h['symbol']}**: ${h['current_value']:.2f} ({h['allocation_pct']:.1f}%)")
    
    with col2:
        st.markdown("### Risk Indicators")
        
        # Check for concentration
        max_allocation = max(h['allocation_pct'] for h in portfolio['holdings'])
        if max_allocation > 40:
            st.warning(f"⚠️ High concentration: {max_allocation:.1f}% in single asset")
        else:
            st.success("✅ Well diversified portfolio")
        
        # Check for underperformers
        underperformers = [h for h in portfolio['holdings'] if h['gain_loss_pct'] < -15]
        if underperformers:
            st.warning(f"⚠️ {len(underperformers)} asset(s) down >15%")
        else:
            st.success("✅ No significant underperformers")

def page_platform_status():
    """Platform health and status page"""
    st.header("🔧 Platform Status")
    
    # System Status
    st.subheader("🖥️ System Health")
    
    status = check_platform_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status['database']['status'] == 'healthy':
            st.success(f"✅ Database: {status['database']['message']}")
        else:
            st.error(f"❌ Database: {status['database']['message']}")
        
        if status['flask_api']['status'] == 'healthy':
            st.success(f"✅ Flask API: {status['flask_api']['message']}")
        else:
            st.error(f"❌ Flask API: {status['flask_api']['message']}")
    
    with col2:
        if status['coingecko']['status'] == 'healthy':
            st.success(f"✅ CoinGecko: {status['coingecko']['message']}")
        else:
            st.error(f"❌ CoinGecko: {status['coingecko']['message']}")
        
        if status['twelvedata']['status'] == 'healthy':
            st.success(f"✅ TwelveData: {status['twelvedata']['message']}")
        else:
            st.error(f"❌ TwelveData: {status['twelvedata']['message']}")
    
    with col3:
        if status['n8n']['status'] == 'healthy':
            st.success(f"✅ n8n: {status['n8n']['message']}")
        else:
            st.error(f"❌ n8n: {status['n8n']['message']}")
    
    st.divider()
    
    # Last Data Updates
    st.subheader("🕒 Last Data Updates")
    
    updates = get_last_data_update()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Crypto Prices**: {updates['crypto']}")
    with col2:
        st.info(f"**Stock Prices**: {updates['stock']}")
    
    st.divider()
    
    # Database Statistics
    st.subheader("📊 Database Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        count_sql = "SELECT COUNT(*) as count FROM portfolio_holdings"
        df = run_query(count_sql)
        st.metric("Holdings", df['count'].iloc[0])
    
    with col2:
        count_sql = "SELECT COUNT(*) as count FROM registered_assets"
        df = run_query(count_sql)
        st.metric("Registered Assets", df['count'].iloc[0])
    
    with col3:
        count_sql = "SELECT COUNT(*) as count FROM crypto_prices"
        df = run_query(count_sql)
        st.metric("Crypto Price Records", df['count'].iloc[0])
    
    with col4:
        count_sql = "SELECT COUNT(*) as count FROM stock_prices"
        df = run_query(count_sql)
        st.metric("Stock Price Records", df['count'].iloc[0])
    
    st.divider()
    
    # Action Buttons
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    with col2:
        if st.button("📊 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data refreshed!")
            st.rerun()
    
    with col3:
        if st.button("🧪 Test APIs", use_container_width=True):
            with st.spinner("Testing APIs..."):
                # Test CoinGecko
                try:
                    response = requests.get(f"{COINGECKO_API}/simple/price?ids=bitcoin&vs_currencies=usd", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ CoinGecko working")
                    else:
                        st.error("❌ CoinGecko failed")
                except:
                    st.error("❌ CoinGecko unreachable")
                
                # Test TwelveData
                try:
                    url = f"{TWELVEDATA_API}/time_series"
                    params = {'symbol': 'AAPL', 'interval': '1day', 'outputsize': 1, 'apikey': TWELVEDATA_API_KEY}
                    response = requests.get(url, params=params, timeout=5)
                    if response.status_code == 200:
                        st.success("✅ TwelveData working")
                    else:
                        st.error("❌ TwelveData failed")
                except:
                    st.error("❌ TwelveData unreachable")

# ---------------------------
# Main App Navigation
# ---------------------------
def main():
    """Main application"""
    
    # CSS for toast notifications positioning
    st.markdown("""
    <style>
    [data-testid="stToast"] {
        bottom: 50px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("📊 Navigation")
        
        # Initialize session state for page navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Overview"
        
        # Navigation buttons
        if st.button("📊 Overview", use_container_width=True, type="primary" if st.session_state.current_page == "Overview" else "secondary"):
            st.session_state.current_page = "Overview"
            st.rerun()
        
        if st.button("📋 Holdings", use_container_width=True, type="primary" if st.session_state.current_page == "Holdings" else "secondary"):
            st.session_state.current_page = "Holdings"
            st.rerun()
        
        if st.button("📈 Analytics", use_container_width=True, type="primary" if st.session_state.current_page == "Analytics" else "secondary"):
            st.session_state.current_page = "Analytics"
            st.rerun()
        
        if st.button("🔧 Platform Status", use_container_width=True, type="primary" if st.session_state.current_page == "Platform Status" else "secondary"):
            st.session_state.current_page = "Platform Status"
            st.rerun()
        
        page = st.session_state.current_page
        
        st.divider()
        
        st.markdown("### ℹ️ Quick Info")
        st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Quick portfolio summary in sidebar
        holdings_df = get_portfolio_holdings()
        if not holdings_df.empty:
            portfolio = calculate_portfolio_value(holdings_df)
            st.metric("Portfolio Value", f"${portfolio['total_value']:,.2f}")
            st.metric("Total Gain/Loss", f"${portfolio['total_gain_loss']:,.2f}")
        
        st.divider()
        
        # Refresh button at bottom
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            get_portfolio_holdings.clear()
            get_latest_crypto_price.clear()
            get_latest_stock_price.clear()
            st.toast("✅ All data refreshed!", icon="🔄")
            st.rerun()
    
    # Route to selected page
    if page == "Overview":
        page_overview()
    elif page == "Holdings":
        page_holdings()
    elif page == "Analytics":
        page_analytics()
    elif page == "Platform Status":
        page_platform_status()

if __name__ == "__main__":
    main()



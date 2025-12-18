# backfill_historical.py - Historical Price Data Backfill Script
"""
Fetches historical price data (>=30 days) for new assets and sends to ETL server.
Auto-triggered when new asset added or when check_historical_data() detects insufficient data.
"""

import requests
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Configuration
DB_PATH = Path(__file__).resolve().parent / "data" / "database" / "finance_data.db"
ETL_SERVER_URL = "http://localhost:5001/api/etl/ingest"
COINGECKO_URL = "https://api.coingecko.com/api/v3"
TWELVEDATA_API_KEY = "c8b06cafbbdd4f5aa28e912167284452"
TWELVEDATA_URL = "https://api.twelvedata.com"

# Historical data requirements
MIN_DAYS_REQUIRED = 30
MIN_DATAPOINTS = 20


def get_db_connection():
    """Create database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def check_if_backfill_needed(asset_symbol: str, asset_type: str) -> bool:
    """
    Check if asset needs historical backfill.
    
    Args:
        asset_symbol: Asset symbol (e.g., 'bitcoin', 'AAPL')
        asset_type: 'crypto' or 'stock'
    
    Returns:
        True if backfill needed, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if asset_type == 'crypto':
            cursor.execute('''
                SELECT COUNT(*) as count, MIN(timestamp) as oldest_date
                FROM crypto_prices
                WHERE LOWER(asset) = LOWER(?)
            ''', (asset_symbol,))
        else:
            cursor.execute('''
                SELECT COUNT(*) as count, MIN(timestamp) as oldest_date
                FROM stock_prices
                WHERE UPPER(symbol) = UPPER(?)
            ''', (asset_symbol,))
        
        row = cursor.fetchone()
        count = row['count']
        oldest_date = row['oldest_date']
        
        conn.close()
        
        # Check if we have enough datapoints
        if count < MIN_DATAPOINTS:
            return True
        
        # Check if oldest data is at least 30 days old
        if oldest_date:
            oldest = datetime.fromisoformat(oldest_date.replace('Z', '+00:00'))
            days_old = (datetime.now() - oldest).days
            if days_old < MIN_DAYS_REQUIRED:
                return True
        
        return False
    
    except Exception as e:
        print(f"Error checking backfill need: {e}")
        return True  # Default to backfill on error


def fetch_crypto_historical(asset_symbol: str, days: int = 90) -> Optional[List[Dict]]:
    """
    Fetch historical crypto prices from CoinGecko.
    
    Args:
        asset_symbol: Crypto asset name (e.g., 'bitcoin')
        days: Number of days to fetch (default 90 for safety margin)
    
    Returns:
        List of price dicts or None on error
    """
    try:
        url = f"{COINGECKO_URL}/coins/{asset_symbol.lower()}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        prices = data.get('prices', [])
        
        # Convert to our format
        historical_data = []
        for timestamp_ms, price in prices:
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            historical_data.append({
                'asset': asset_symbol.lower(),
                'price': price,
                'timestamp': dt.isoformat() + 'Z'
            })
        
        return historical_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching crypto historical data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def fetch_stock_historical(symbol: str, days: int = 90) -> Optional[List[Dict]]:
    """
    Fetch historical stock prices from TwelveData.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        days: Number of days to fetch (default 90 for safety margin)
    
    Returns:
        List of OHLCV dicts or None on error
    """
    try:
        url = f"{TWELVEDATA_URL}/time_series"
        params = {
            'symbol': symbol.upper(),
            'interval': '1day',
            'outputsize': days,
            'apikey': TWELVEDATA_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if 'values' not in data:
            print(f"No values returned for {symbol}: {data}")
            return None
        
        # Convert to our format
        historical_data = []
        for item in data['values']:
            try:
                dt = datetime.strptime(item['datetime'], '%Y-%m-%d')
                historical_data.append({
                    'symbol': symbol.upper(),
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': int(item['volume']),
                    'timestamp': dt.isoformat() + 'Z'
                })
            except (KeyError, ValueError) as e:
                print(f"Skipping malformed data point: {e}")
                continue
        
        return historical_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stock historical data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def send_to_etl(data_type: str, data: List[Dict]) -> bool:
    """
    Send historical data to ETL server for processing.
    
    Args:
        data_type: 'crypto_prices' or 'stock_prices'
        data: List of price records
    
    Returns:
        True if successful, False otherwise
    """
    try:
        payload = {
            'data_type': data_type,
            'data': data
        }
        
        response = requests.post(ETL_SERVER_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get('success'):
            print(f"Successfully sent {len(data)} {data_type} records to ETL")
            return True
        else:
            print(f"ETL rejected data: {result.get('error')}")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"Error sending to ETL: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def backfill_asset(asset_symbol: str, asset_type: str, force: bool = False) -> bool:
    """
    Backfill historical data for a single asset.
    
    Args:
        asset_symbol: Asset symbol
        asset_type: 'crypto' or 'stock'
        force: Force backfill even if data exists
    
    Returns:
        True if successful or not needed, False on error
    """
    print(f"\n{'='*60}")
    print(f"Backfill Check: {asset_symbol} ({asset_type})")
    print(f"{'='*60}")
    
    # Check if backfill needed
    if not force:
        needs_backfill = check_if_backfill_needed(asset_symbol, asset_type)
        if not needs_backfill:
            print(f"✓ {asset_symbol} has sufficient historical data (>={MIN_DAYS_REQUIRED} days)")
            return True
    
    print(f"→ Fetching historical data for {asset_symbol}...")
    
    # Fetch historical data
    if asset_type == 'crypto':
        historical_data = fetch_crypto_historical(asset_symbol)
        data_type = 'crypto_prices'
    else:
        historical_data = fetch_stock_historical(asset_symbol)
        data_type = 'stock_prices'
    
    if not historical_data:
        print(f"✗ Failed to fetch historical data for {asset_symbol}")
        return False
    
    print(f"✓ Fetched {len(historical_data)} historical records")
    
    # Send to ETL
    print(f"→ Sending to ETL server...")
    success = send_to_etl(data_type, historical_data)
    
    if success:
        print(f"✓ Backfill complete for {asset_symbol}")
        # Ensure asset is present in portfolio_holdings with quantity 0 if missing
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM portfolio_holdings WHERE asset_symbol = ? AND asset_type = ?', (asset_symbol, asset_type))
            exists = cursor.fetchone()[0]
            if not exists:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute('''
                    INSERT INTO portfolio_holdings (
                        asset_symbol, asset_type, quantity_owned, purchase_price, purchase_date, total_cost, fees, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    asset_symbol,
                    asset_type,
                    0.0,
                    0.0,
                    now,
                    0.0,
                    0.0,
                    'Auto-added by backfill',
                    now,
                    now
                ))
                conn.commit()
                print(f"✓ Added {asset_symbol} ({asset_type}) to portfolio_holdings with quantity 0")
            conn.close()
        except Exception as e:
            print(f"Warning: Could not add {asset_symbol} to portfolio_holdings: {e}")
        return True
    else:
        print(f"✗ Failed to send data to ETL for {asset_symbol}")
        return False


def backfill_all_registered_assets() -> Dict[str, int]:
    """
    Backfill all registered assets that need historical data.
    
    Returns:
        Dict with success/failure counts
    """
    print("\n" + "="*60)
    print("AUTOMATIC HISTORICAL DATA BACKFILL")
    print("="*60)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all registered assets
        cursor.execute('''
            SELECT DISTINCT symbol, asset_type
            FROM registered_assets
        ''')
        assets = cursor.fetchall()
        
        # Also get assets from holdings
        cursor.execute('''
            SELECT DISTINCT asset_symbol as symbol, asset_type
            FROM portfolio_holdings
        ''')
        holdings_assets = cursor.fetchall()
        
        conn.close()
        
        # Combine and deduplicate
        all_assets = {}
        for row in assets:
            all_assets[(row['symbol'], row['asset_type'])] = True
        for row in holdings_assets:
            all_assets[(row['symbol'], row['asset_type'])] = True
        
        total = len(all_assets)
        print(f"\nFound {total} unique assets to check")
        
        results = {
            'total': total,
            'success': 0,
            'skipped': 0,
            'failed': 0
        }
        
        for i, (symbol, asset_type) in enumerate(all_assets.keys(), 1):
            print(f"\n[{i}/{total}] Processing {symbol} ({asset_type})...")
            
            success = backfill_asset(symbol, asset_type, force=False)
            
            if success:
                # Check if actually needed backfill
                if check_if_backfill_needed(symbol, asset_type):
                    results['failed'] += 1
                else:
                    results['success'] += 1
            else:
                results['failed'] += 1
            
            # Rate limiting
            time.sleep(1)
        
        print("\n" + "="*60)
        print("BACKFILL SUMMARY")
        print("="*60)
        print(f"Total Assets: {results['total']}")
        print(f"Successful: {results['success']}")
        print(f"Failed: {results['failed']}")
        print("="*60)
        
        return results
    
    except Exception as e:
        print(f"Error during backfill: {e}")
        return {'total': 0, 'success': 0, 'failed': 0}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Manual backfill specific asset
        if len(sys.argv) >= 3:
            asset = sys.argv[1]
            asset_type = sys.argv[2]
            force = '--force' in sys.argv
            
            print(f"Manual backfill: {asset} ({asset_type})")
            success = backfill_asset(asset, asset_type, force=force)
            sys.exit(0 if success else 1)
        else:
            print("Usage: python backfill_historical.py <symbol> <crypto|stock> [--force]")
            print("   or: python backfill_historical.py  (to backfill all)")
            sys.exit(1)
    else:
        # Backfill all registered assets
        results = backfill_all_registered_assets()
        sys.exit(0 if results['failed'] == 0 else 1)

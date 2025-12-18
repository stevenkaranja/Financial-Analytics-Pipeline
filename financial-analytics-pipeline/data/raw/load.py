# load.py - Data Loading Module
"""
Loads transformed data into SQLite database.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Database path
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "database" / "finance_data.db"


def get_db_connection():
    """Create database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_crypto_prices(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Load crypto price data into database.
    
    Args:
        df: DataFrame with columns [asset, price, timestamp]
    
    Returns:
        Dict with success status and row count
    """
    if df.empty:
        return {"success": True, "rows_inserted": 0, "message": "No data to insert"}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        rows_inserted = 0
        for _, row in df.iterrows():
            # Convert timestamp to string format for SQLite
            timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['timestamp']) else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO crypto_prices (asset, price, timestamp)
                VALUES (?, ?, ?)
            ''', (row['asset'], row['price'], timestamp_str))
            rows_inserted += 1
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "rows_inserted": rows_inserted,
            "message": f"Successfully loaded {rows_inserted} crypto price records"
        }
    
    except Exception as e:
        return {
            "success": False,
            "rows_inserted": 0,
            "message": f"Error loading crypto prices: {str(e)}"
        }


def load_stock_prices(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Load stock price data into database.
    
    Args:
        df: DataFrame with columns [symbol, open, high, low, close, volume, timestamp]
    
    Returns:
        Dict with success status and row count
    """
    if df.empty:
        return {"success": True, "rows_inserted": 0, "message": "No data to insert"}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        rows_inserted = 0
        for _, row in df.iterrows():
            # Convert timestamp to string format for SQLite
            timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['timestamp']) else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO stock_prices 
                (symbol, open, high, low, close, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['symbol'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                timestamp_str
            ))
            rows_inserted += 1
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "rows_inserted": rows_inserted,
            "message": f"Successfully loaded {rows_inserted} stock price records"
        }
    
    except Exception as e:
        return {
            "success": False,
            "rows_inserted": 0,
            "message": f"Error loading stock prices: {str(e)}"
        }


def load_holding(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load holding data into database.
    
    Args:
        data: Dict with holding information
    
    Returns:
        Dict with success status and holding ID
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO portfolio_holdings 
            (asset_symbol, asset_type, quantity_owned, purchase_price, purchase_date, total_cost, fees, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['asset_symbol'],
            data['asset_type'],
            data['quantity_owned'],
            data['purchase_price'],
            data['purchase_date'],
            data['total_cost'],
            data['fees'],
            data['notes']
        ))
        
        holding_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "holding_id": holding_id,
            "message": f"Successfully added holding: {data['asset_symbol']}"
        }
    
    except sqlite3.IntegrityError as e:
        return {
            "success": False,
            "holding_id": None,
            "message": f"Holding already exists or constraint violation: {str(e)}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "holding_id": None,
            "message": f"Error adding holding: {str(e)}"
        }


def load_registered_asset(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load registered asset into database.
    
    Args:
        data: Dict with asset information
    
    Returns:
        Dict with success status and asset ID
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if asset already exists
        cursor.execute('''
            SELECT id FROM registered_assets 
            WHERE symbol = ? AND asset_type = ?
        ''', (data['symbol'], data['asset_type']))
        
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return {
                "success": True,
                "asset_id": existing[0],
                "message": f"Asset {data['symbol']} already registered",
                "already_exists": True
            }
        
        cursor.execute('''
            INSERT INTO registered_assets (symbol, asset_type, name, source)
            VALUES (?, ?, ?, ?)
        ''', (
            data['symbol'],
            data['asset_type'],
            data['name'],
            data['source']
        ))
        
        asset_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "asset_id": asset_id,
            "message": f"Successfully registered asset: {data['symbol']}",
            "already_exists": False
        }
    
    except Exception as e:
        return {
            "success": False,
            "asset_id": None,
            "message": f"Error registering asset: {str(e)}",
            "already_exists": False
        }


def check_historical_data(symbol: str, asset_type: str, days_required: int = 30) -> bool:
    """
    Check if asset has sufficient historical data.
    
    Args:
        symbol: Asset symbol
        asset_type: 'crypto' or 'stock'
        days_required: Minimum days of historical data required
    
    Returns:
        True if sufficient data exists, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if asset_type == 'crypto':
            cursor.execute('''
                SELECT COUNT(*) 
                FROM crypto_prices 
                WHERE LOWER(asset) = LOWER(?)
                AND timestamp >= datetime('now', '-' || ? || ' days')
            ''', (symbol, days_required))
        else:
            cursor.execute('''
                SELECT COUNT(*) 
                FROM stock_prices 
                WHERE UPPER(symbol) = UPPER(?)
                AND timestamp >= datetime('now', '-' || ? || ' days')
            ''', (symbol, days_required))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        # Require at least 20 data points (roughly 20 days for daily data)
        return count >= 20
    
    except Exception as e:
        print(f"Error checking historical data: {e}")
        return False


if __name__ == "__main__":
    # Test database connection
    print("🧪 Testing load.py...")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM crypto_prices")
        crypto_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        stock_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"\n✅ Database connected successfully")
        print(f"   Crypto prices: {crypto_count} rows")
        print(f"   Stock prices: {stock_count} rows")
        
        # Test historical data check
        has_data = check_historical_data("BTC", "crypto", 30)
        print(f"\n✅ Historical data check: BTC has sufficient data = {has_data}")
        
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")

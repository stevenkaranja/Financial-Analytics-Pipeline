# transform.py - Data Transformation Module
"""
Transforms raw price data from n8n into clean format for database storage.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Any


def transform_crypto_data(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transform raw crypto data from n8n into clean DataFrame.
    
    Expected input format from n8n:
    [
        {
            "asset": "bitcoin",
            "price": 45000.50,
            "timestamp": "2025-12-12T10:00:00Z"
        },
        ...
    ]
    """
    if not raw_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(raw_data)
    
    # Ensure required columns exist
    required_cols = ['asset', 'price', 'timestamp']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Clean asset names (lowercase, strip whitespace)
    df['asset'] = df['asset'].str.lower().str.strip()
    
    # Ensure price is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with invalid data
    df = df.dropna(subset=['asset', 'price', 'timestamp'])
    
    # Remove duplicates (keep latest)
    df = df.sort_values('timestamp').drop_duplicates(subset=['asset', 'timestamp'], keep='last')
    
    return df


def transform_stock_data(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transform raw stock data from n8n into clean DataFrame.
    
    Expected input format from n8n:
    [
        {
            "symbol": "AAPL",
            "open": 180.50,
            "high": 182.00,
            "low": 179.50,
            "close": 181.25,
            "volume": 50000000,
            "timestamp": "2025-12-12T10:00:00Z"
        },
        ...
    ]
    """
    if not raw_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(raw_data)
    
    # Ensure required columns exist
    required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Clean symbols (uppercase, strip whitespace)
    df['symbol'] = df['symbol'].str.upper().str.strip()
    
    # Ensure numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with invalid data
    df = df.dropna(subset=['symbol', 'close', 'timestamp'])
    
    # Remove duplicates (keep latest)
    df = df.sort_values('timestamp').drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
    
    return df


def transform_holding_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw holding data for database insertion.
    
    Expected input format:
    {
        "asset_symbol": "BTC",
        "asset_type": "crypto",
        "quantity_owned": 0.5,
        "purchase_price": 45000.00,
        "purchase_date": "2025-12-01",
        "fees": 10.00,
        "notes": "Initial purchase"
    }
    """
    # Clean and validate
    clean_data = {
        'asset_symbol': str(raw_data.get('asset_symbol', '')).upper().strip(),
        'asset_type': str(raw_data.get('asset_type', '')).lower().strip(),
        'quantity_owned': float(raw_data.get('quantity_owned', 0)),
        'purchase_price': float(raw_data.get('purchase_price', 0)),
        'purchase_date': raw_data.get('purchase_date'),
        'fees': float(raw_data.get('fees', 0)),
        'notes': str(raw_data.get('notes', '')).strip()
    }
    
    # Calculate total cost
    clean_data['total_cost'] = (clean_data['quantity_owned'] * clean_data['purchase_price']) + clean_data['fees']
    
    # Validate asset type
    if clean_data['asset_type'] not in ['crypto', 'stock']:
        raise ValueError(f"Invalid asset_type: {clean_data['asset_type']}")
    
    # Validate numeric values
    if clean_data['quantity_owned'] <= 0:
        raise ValueError("quantity_owned must be greater than 0")
    
    if clean_data['purchase_price'] <= 0:
        raise ValueError("purchase_price must be greater than 0")
    
    return clean_data


def transform_registered_asset(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw registered asset data for database insertion.
    
    Expected input format:
    {
        "symbol": "BTC",
        "asset_type": "crypto",
        "name": "Bitcoin",
        "source": "coingecko"
    }
    """
    clean_data = {
        'symbol': str(raw_data.get('symbol', '')).upper().strip(),
        'asset_type': str(raw_data.get('asset_type', '')).lower().strip(),
        'name': str(raw_data.get('name', '')).strip(),
        'source': str(raw_data.get('source', '')).lower().strip()
    }
    
    # Validate asset type
    if clean_data['asset_type'] not in ['crypto', 'stock']:
        raise ValueError(f"Invalid asset_type: {clean_data['asset_type']}")
    
    if not clean_data['symbol']:
        raise ValueError("symbol is required")
    
    return clean_data


if __name__ == "__main__":
    # Test transformations
    print("🧪 Testing transform.py...")
    
    # Test crypto data
    test_crypto = [
        {"asset": "bitcoin", "price": 45000.50, "timestamp": "2025-12-12T10:00:00Z"},
        {"asset": " ETHEREUM ", "price": 3000.25, "timestamp": "2025-12-12T10:00:00Z"}
    ]
    crypto_df = transform_crypto_data(test_crypto)
    print(f"\n✅ Crypto transformation: {len(crypto_df)} rows")
    print(crypto_df.head())
    
    # Test stock data
    test_stock = [
        {
            "symbol": "AAPL",
            "open": 180.50,
            "high": 182.00,
            "low": 179.50,
            "close": 181.25,
            "volume": 50000000,
            "timestamp": "2025-12-12T10:00:00Z"
        }
    ]
    stock_df = transform_stock_data(test_stock)
    print(f"\n✅ Stock transformation: {len(stock_df)} rows")
    print(stock_df.head())
    
    print("\n✅ All tests passed!")
